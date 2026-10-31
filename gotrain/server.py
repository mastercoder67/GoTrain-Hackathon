"""`gotrain ui`: a chat front end for the agent, using only the standard library.

The page asks for a task and a budget in plain words, starts `gotrain agent` as a
separate process (so the compute lock and logs behave exactly as on the CLI),
and polls the session's diary state. It binds to localhost; on SageMaker open it
through JupyterLab's proxy at <jupyterlab-url>/proxy/<port>/.
"""
import json
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import diary
from .config import TASKS, runs_root

TASK_WORDS = [
    ("m-pv4ger-seg", r"m-pv4ger-seg"), ("m-pv4ger", r"m-pv4ger\b"), ("m-nz-cattle", r"m-nz-cattle"),
    ("m-eurosat", r"m-eurosat"), ("m-nz-cattle", r"\b(cattle|cows?|livestock)\b"),
    ("m-eurosat", r"\b(eurosat|land[- ]?cover|land[- ]?use|land types?)\b"),
]
SEGMENT_WORDS = r"\b(segment\w*|where|pixels?|masks?|outlines?|locat\w*|map)\b"
CLASSIFY_WORDS = r"\b(classif\w*|is there|whether|present|presence|yes/no|yes or no|contains?)\b"


def parse_request(text):
    """Task, minutes and experiments from a plain-language request, or a follow-up question."""
    lower = text.lower()
    task = next((t for t, pattern in TASK_WORDS if re.search(pattern, lower)), None)
    if task is None and re.search(r"\b(solar|panels?|pv|photovoltaic)\b", lower):
        if re.search(SEGMENT_WORDS, lower):
            task = "m-pv4ger-seg"
        elif re.search(CLASSIFY_WORDS, lower):
            task = "m-pv4ger"
        else:
            return {"question": "Should GoTrain find where the solar panels are, or say whether an image has one?",
                    "detail": "Try: \"segment solar panels in 30 minutes\" or \"classify solar panels in 30 minutes\"."}
    if task is None:
        return {"question": "Which dataset should GoTrain train on?",
                "detail": "Supported: EuroSAT land cover, solar panels (classification or segmentation), "
                          "and cattle segmentation."}
    minutes = 30.0
    if match := re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hrs?|hours?)\b", lower):
        minutes = float(match.group(1)) * 60
    elif match := re.search(r"(\d+(?:\.\d+)?)\s*(?:m|mins?|minutes?)\b", lower):
        minutes = float(match.group(1))
    experiments = 3
    if match := re.search(r"(\d+)\s*(?:experiments?|tries|attempts|runs)\b", lower):
        experiments = int(match.group(1))
    return {"task": task, "minutes": min(max(minutes, 1.0), 480.0), "experiments": min(max(experiments, 1), 12)}


class Launcher:
    """At most one agent process at a time, started by this UI."""

    def __init__(self):
        self.lock = threading.Lock()
        self.process = self.task = self.started_at = self.log = None
        self.session_id = None

    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, plan, request):
        with self.lock:
            if self.running():
                return False
            logs = runs_root() / "ui-logs"
            logs.mkdir(parents=True, exist_ok=True)
            self.log = logs / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_{plan['task']}.log")
            self.task, self.session_id = plan["task"], None
            self.started_at = time.time()
            command = [sys.executable, "-m", "gotrain", "agent", plan["task"], "--experiments", str(plan["experiments"]),
                       "--minutes", str(plan["minutes"]), "--request", request]
            with self.log.open("w") as out:
                self.process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT)
            return True

    def session(self):
        """The session directory this launch created, once the agent has logged its start."""
        if self.session_id or self.started_at is None:
            return self.session_id
        for row in diary.sessions():
            created = datetime.fromisoformat(row["started_at"]).timestamp()
            if row["task"] == self.task and created >= self.started_at - 5:
                self.session_id = row["session_id"]
                break
        return self.session_id

    def error(self):
        if self.process is None or self.process.poll() in (None, 0) or self.session():
            return None
        return self.log.read_text()[-2000:] if self.log and self.log.exists() else "Agent process exited"


LAUNCHER = Launcher()


def state_for(session_id):
    if not session_id or not re.fullmatch(r"session_[0-9a-f]+", session_id):
        return None
    directory = runs_root() / "sessions" / session_id
    return diary.state(directory) if (directory / "events.jsonl").exists() else None


class Handler(BaseHTTPRequestHandler):
    def reply(self, body, status=200, kind="application/json"):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path.endswith("/api/state"):
            session = parse_qs(url.query).get("session", [None])[0] or LAUNCHER.session()
            self.reply({"state": state_for(session), "sessions": diary.sessions()})
        elif url.path.endswith("/api/registry"):
            from .registry import registry
            self.reply({**registry(), "sessions": diary.sessions()})
        elif url.path.endswith("/api/launch"):
            self.reply({"session_id": LAUNCHER.session(), "running": LAUNCHER.running(), "error": LAUNCHER.error()})
        else:
            self.reply(diary.page(), kind="text/html")

    def do_POST(self):
        path = urlparse(self.path).path
        length = min(int(self.headers.get("Content-Length") or 0), 10_000)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self.reply({"error": "invalid JSON"}, 400)
        if path.endswith("/api/registry"):
            from .registry import act
            try:
                result = act(str(body.get("action")), str(body.get("task")), body.get("run_id") or None)
            except Exception as error:
                return self.reply({"ok": False, "error": str(error)}, 400)
            return self.reply({"ok": True, "result": result})
        if not path.endswith("/api/start"):
            return self.reply({"error": "not found"}, 404)
        text = str(body.get("message", "")).strip()[:500]
        plan = parse_request(text)
        if "question" in plan:
            return self.reply({"started": False, "reply": plan["question"], "detail": plan["detail"]})
        if not LAUNCHER.start(plan, text):
            return self.reply({"started": False, "reply": f"GoTrain is still busy with {LAUNCHER.task}.",
                               "detail": "The team GPU trains one request at a time. Open it from the sidebar to watch."})
        return self.reply({"started": True, **plan})

    def log_message(self, *args):
        pass  # Keep the terminal for the agent's own output.


def serve(port=8765):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"GoTrain chat is running on port {port}.\n"
          f"  On SageMaker: open your JupyterLab address, replace the end of it with /proxy/{port}/\n"
          f"                (for example https://…/jupyterlab/default/proxy/{port}/)\n"
          f"  On this machine: http://127.0.0.1:{port}/\n"
          f"Supported tasks: {', '.join(TASKS)}. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
