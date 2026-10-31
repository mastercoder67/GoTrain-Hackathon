"""Model and dataset registry view: every run is a version; storage actions are logged.

Built on artifacts.py (filesystem plus optional S3, checksummed manifests). Every
publish/restore is appended to registry-events.jsonl so the history is auditable.
"""
import json
import os

from . import artifacts
from .config import TASKS, artifacts_root, data_root, runs_root
from .store import append_event, list_runs, read_json, run_dir, task_dir

ACTIONS = {"publish-model", "restore-model", "publish-data", "restore-data"}


def log_path():
    return runs_root() / "registry-events.jsonl"


def activity(limit=None):
    rows = []
    if log_path().exists():
        for line in log_path().read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows[-limit:][::-1] if limit else rows


def bucket():
    return os.environ.get("GOTRAIN_S3_BUCKET") or None


def versions(task):
    rows = [r for r in list_runs(task) if "recipe" in r]
    completed = [r for r in rows if r.get("status") == "completed"]
    best = max(completed, key=lambda r: r["selection_score"])["run_id"] if completed else None
    history = activity()
    out = []
    for run in rows:
        config_path = run_dir(task, run["run_id"]) / "config.json"
        config = read_json(config_path) if config_path.exists() else {}
        verified = [e for e in history if e.get("task") == task and e.get("run_id") == run["run_id"]
                    and e.get("action") == "restore-model" and e.get("ok")]
        recipe = run["recipe"]
        out.append({"run_id": run["run_id"], "status": run.get("status"), "created_at": config.get("created_at"),
                    "score": run.get("selection_score"), "metric": run.get("selection_metric"),
                    "hypothesis": run.get("hypothesis") or config.get("hypothesis"),
                    "recipe": f"{recipe.get('model')} · {recipe.get('image_size')}px · lr {recipe.get('learning_rate'):g}"
                              + (" · focus crops" if recipe.get("focus_crops") else "")
                              + (" · TTA" if recipe.get("tta") else ""),
                    "parent": recipe.get("warm_start") or config.get("parent_run_id"),
                    "sha256": run.get("checkpoint_sha256"), "best": run["run_id"] == best,
                    "published": (artifacts_root() / "models" / task / run["run_id"] / "manifest.json").exists(),
                    "verified_at": verified[-1]["time"] if verified else None})
    return out


def registry():
    tasks = []
    for task, info in TASKS.items():
        directory = task_dir(task)
        manifest = artifacts_root() / "datasets" / task / "manifest.json"
        test = directory / "test-result.json"
        tasks.append({"task": task, "kind": info["kind"],
                      "metric": "accuracy" if info["kind"] == "classification" else "mean_iou",
                      "dataset_ready": (data_root() / info["benchmark"] / f"{task}.zip").exists(),
                      "dataset_sha256": read_json(manifest)["sha256"] if manifest.exists() else None,
                      "frozen": (directory / "final-selection.json").exists(),
                      "test_score": read_json(test)["metrics"][read_json(test)["selection_metric"]] if test.exists() else None,
                      "versions": versions(task)})
    return {"storage": {"filesystem": str(artifacts_root()), "s3_bucket": bucket()},
            "tasks": tasks, "activity": activity(limit=12)}


def act(action, task, run_id=None):
    """Run one storage action and record its outcome, success or failure."""
    if action not in ACTIONS or task not in TASKS:
        raise ValueError("Unknown action or task")
    if action.endswith("model"):
        run_dir(task, run_id or "")  # validates the identifier
        if not run_id:
            raise ValueError("A run ID is required")
    calls = {"publish-model": lambda: artifacts.publish_model(task, run_id, bucket()),
             "restore-model": lambda: artifacts.restore_model(task, run_id, bucket()),
             "publish-data": lambda: artifacts.publish_dataset(task, bucket()),
             "restore-data": lambda: artifacts.restore_dataset(task, bucket())}
    try:
        result = calls[action]()
    except Exception as error:
        append_event(log_path(), "registry", action=action, task=task, run_id=run_id, ok=False, error=str(error))
        raise
    append_event(log_path(), "registry", action=action, task=task, run_id=run_id, ok=True,
                 location=result.get("path"), s3_uri=result.get("s3_uri"))
    return result
