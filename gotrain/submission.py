import shutil
from pathlib import Path

from .config import TASKS, runs_root
from .store import read_json, task_dir, write_json


def export(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    results, missing = [], []
    for task in TASKS:
        source = task_dir(task)
        if not (source / "test-result.json").exists():
            missing.append(task)
        else:
            results.append(read_json(source / "test-result.json"))
        if source.exists():
            shutil.copytree(source, destination / task, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.pt", "*.tmp", "compute.lock"))
    sessions = runs_root() / "sessions"
    if sessions.exists():
        shutil.copytree(sessions, destination / "agent-sessions", dirs_exist_ok=True)
    write_json(destination / "scores.json", {"results": results, "missing_tasks": missing,
                                            "complete": not missing})
    return {"path": str(destination.resolve()), "complete": not missing, "missing_tasks": missing,
            "note": "Checkpoints are separate versioned artifacts, not embedded in this log bundle."}
