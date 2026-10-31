"""Build an offline demo from exported evidence, without loading models or data splits."""
import argparse
import base64
from html.parser import HTMLParser
import json
from pathlib import Path

from .config import TASKS


class ReportSamples(HTMLParser):
    def __init__(self):
        super().__init__()
        self.samples = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.current = {"images": [], "text": []}
        if self.current is not None and tag == "img":
            source = dict(attrs).get("src", "")
            if source.startswith("data:image/png;base64,"):
                self.current["images"].append(source)

    def handle_data(self, data):
        if self.current is not None and data.strip():
            self.current["text"].append(data.strip())

    def handle_endtag(self, tag):
        if tag == "article" and self.current is not None:
            self.samples.append(self.current)
            self.current = None


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def read_json_lines(path):
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def image_uri(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or path.suffix != ".png":
        raise ValueError("Example image must be a PNG inside the showcase directory")
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def session_steps(directory):
    steps = []
    path = directory / "events.jsonl"
    if not path.is_file():
        return steps
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        result = row.get("result") or {}
        if row.get("event") == "experiment_requested":
            steps.append({"title": f"Experiment {row.get('attempt', '?')}: hypothesis",
                          "detail": row.get("hypothesis", ""), "time": row.get("time"), "kind": "hypothesis"})
        elif row.get("event") == "tool_result":
            tool = row.get("tool")
            if tool == "inspect_dataset":
                steps.append({"title": "Inspect the dataset", "kind": "inspect", "time": row.get("time"),
                              "detail": f"{result.get('train_samples')} training images, {result.get('valid_samples')} validation images, {result.get('bands')} bands."})
            elif tool == "run_experiment":
                complete = result.get("status") == "completed"
                metric = result.get("selection_metric", "score").replace("_", " ")
                detail = (f"Validation {metric}: {result['selection_score']:.4f}. Best epoch: {result.get('best_epoch')}."
                          if complete else str(result.get("error", "Experiment did not complete.")))
                steps.append({"title": "Measured validation result" if complete else "Experiment failed",
                              "detail": detail, "time": row.get("time"), "kind": "result" if complete else "failed"})
            elif tool in {"publish_best_model", "verify_model_download"}:
                verified = result.get("verified") is True
                steps.append({"title": "Verify model restore" if tool == "verify_model_download" else "Publish the selected model",
                              "kind": "artifact", "time": row.get("time"),
                              "detail": "Restored model passed checksum verification." if verified else
                              "Recorded artifact operation. See the original session log for the full tool result."})
    return steps


def build_payload(submission):
    submission = Path(submission).expanduser().resolve()
    scores = read(submission / "scores.json")
    if not scores:
        raise ValueError("Choose an exported submission directory containing scores.json")
    final = {r["task"]: r for r in scores.get("results", [])}
    verification = read(submission / "verification.json", {})
    latest_demo_checks = read(submission.parent / "demo-verification.json", {})
    checks = {}
    for record in (verification, latest_demo_checks):
        for row in record.get("checks", []):
            command = row.get("command", [])
            if "pytest" in command or "smoke" in command:
                name = "Software tests" if "pytest" in command else "Synthetic model check"
                checks[name] = {"name": name, "passed": row.get("returncode") == 0,
                                "output": row.get("stdout", "")}
    sessions = []
    for directory in (submission / "agent-sessions").glob("*/"):
        summary = read(directory / "session.json", {})
        if summary.get("completed_runs"):
            sessions.append((summary, directory))
    tasks = []
    labels = {"m-eurosat": "Land cover", "m-pv4ger": "Solar-panel classification",
              "m-pv4ger-seg": "Solar-panel segmentation", "m-nz-cattle": "Cattle segmentation"}
    for task, info in TASKS.items():
        directory = submission / task
        summaries = [read(p) for p in sorted(directory.glob("*/summary.json"))]
        completed = [r for r in summaries if r.get("status") == "completed"]
        best = max(completed, key=lambda r: r["selection_score"], default=None)
        runs = [{k: r.get(k) for k in ("run_id", "status", "hypothesis", "selection_score", "selection_metric",
                                      "experiment_type", "best_epoch", "elapsed_seconds", "error")} for r in summaries]
        parser = ReportSamples()
        report = directory / "report.html"
        if report.is_file():
            parser.feed(report.read_text(encoding="utf-8"))
        showcase_path = directory / "showcase"
        showcase = read(showcase_path / "showcase.json")
        if showcase:
            showcase = dict(showcase)
            showcase["examples"] = [{**e, "src": image_uri(showcase_path, e["image"])} for e in showcase["examples"]]
        candidates = sorted([(s, p) for s, p in sessions if s.get("task") == task],
                            key=lambda pair: pair[0].get("finished_at", ""), reverse=True)
        session = candidates[0] if candidates else None
        tasks.append({"id": task, "label": labels[task], "kind": info["kind"], "final": final.get(task),
                      "best": best and {k: best.get(k) for k in ("selection_score", "selection_metric", "run_id")},
                      "runs": runs, "samples": parser.samples, "showcase": showcase,
                      "steps": session_steps(session[1]) if session else [],
                      "session_id": session[0].get("session_id") if session else None,
                      "diagnosis": read(directory / final[task]["run_id"] / "diagnosis.json", {}) if task in final else {}})
    return {"tasks": tasks, "complete": bool(scores.get("complete")), "missing_tasks": scores.get("missing_tasks", []),
            "checks": list(checks.values()), "storage": read(submission / "storage-verification.json", {}),
            "audit": read(submission / "results-audit.json", {}),
            "memory": read_json_lines(submission / "lessons.jsonl"),
            "registry": read(submission / "registry-state.json", {})}


def build_demo(submission, output):
    payload = build_payload(submission)
    template = Path(__file__).with_name("demo.html").read_text(encoding="utf-8")
    # JSON is embedded as data, never interpreted as markup from an evidence file.
    encoded = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c").replace("&", "\\u0026")
    document = template.replace("__GOTRAIN_EVIDENCE__", encoded)
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return {"demo": str(output), "tasks": len(payload["tasks"]), "complete": payload["complete"],
            "mode": "Saved evidence only; no model execution or benchmark split access"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", default="~/gotrain-release-verification/submission")
    parser.add_argument("--out", default="~/gotrain-demo.html")
    args = parser.parse_args()
    print(json.dumps(build_demo(args.submission, args.out), indent=2))


if __name__ == "__main__":
    main()
