"""Immutable run records, checksums, and atomic writes; no cloud dependencies."""
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import runs_root, task_info


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str))
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def append_event(path, event, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"time": now(), "event": event, **fields}, default=str) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def task_dir(task):
    task_info(task)
    return runs_root() / task


def run_dir(task, run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid run ID")
    return task_dir(task) / run_id


def new_run(task):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    path = run_dir(task, run_id)
    path.mkdir(parents=True, exist_ok=False)
    return run_id, path


def created_at(path):
    config = path.parent / "config.json"
    return read_json(config).get("created_at", "") if config.exists() else ""


def list_runs(task):
    # Chronological by recorded creation time; run IDs only have one-second resolution.
    paths = sorted(task_dir(task).glob("*/summary.json"), key=lambda p: (created_at(p), p.parent.name))
    return [read_json(path) for path in paths]


def best_run(task):
    rows = [r for r in list_runs(task) if r.get("status") == "completed"]
    if not rows:
        raise ValueError("No completed validation experiments yet")
    # Ties favour the first recorded run; test results never affect this selection.
    return max(rows, key=lambda r: r["selection_score"])


def assert_unfrozen(task):
    if (task_dir(task) / "final-selection.json").exists():
        raise RuntimeError("Task is frozen for final evaluation; training is disabled")


def claim_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(payload, f, indent=2)
