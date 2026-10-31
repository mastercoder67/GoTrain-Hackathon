"""Cross-task memory: what each experiment changed, what it measured, and what the agent concluded.

lessons.jsonl is append-only. Experiment lessons are derived mechanically from
run records (recipe difference and validation score change), so they cannot be
invented. Insight lessons are the agent's own conclusions and must cite runs.
Lessons from other tasks are hints to re-test, never proof for a new dataset.
"""
import json
from collections import defaultdict

from .config import TASKS, runs_root, task_info
from .store import append_event, list_runs, read_json, run_dir

# Settings that change speed or bookkeeping, not what the model learns.
IGNORED = {"max_minutes", "workers", "cache", "amp", "stats_samples", "seed"}


def path():
    return runs_root() / "lessons.jsonl"


def load():
    rows = []
    if path().exists():
        for line in path().read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def recipe_change(before, after):
    keys = sorted((set(before) | set(after)) - IGNORED)
    return {k: [before.get(k), after.get(k)] for k in keys if before.get(k) != after.get(k)}


def change_label(change):
    parts = []
    for key, (old, new) in change.items():
        if key == "warm_start":
            parts.append("fine-tune previous best" if new else "train from pretrained weights")
        elif isinstance(new, bool) or isinstance(old, bool):
            parts.append(f"{key} {'on' if new else 'off'}")
        else:
            parts.append(f"{key} {old}→{new}")
    return " + ".join(parts) or "same recipe"


def describe(lesson):
    if lesson["type"] == "insight":
        return f"{lesson['task']}: {lesson['text']}"
    label = change_label(lesson["change"])
    if lesson["verdict"] == "failed":
        return f"{lesson['task']}: {label} failed ({lesson.get('error', '')[:120]})"
    return (f"{lesson['task']}: {label} · {lesson['metric']} {lesson['before']:.4f} → {lesson['after']:.4f} "
            f"({lesson['delta']:+.4f}) · {lesson['verdict']}")


def experiment_lesson(task, run, previous, session_id=None):
    """Lesson comparing a finished or failed run with the best run it had to beat."""
    lesson = {"type": "experiment", "task": task, "kind": task_info(task)["kind"], "run_id": run["run_id"],
              "compared_to": previous["run_id"], "change": recipe_change(previous["recipe"], run["recipe"]),
              "metric": previous["selection_metric"], "before": previous["selection_score"],
              "session_id": session_id}
    config_path = run_dir(task, run["run_id"]) / "config.json"
    if config_path.exists():
        lesson["hypothesis"] = read_json(config_path).get("hypothesis")
    diagnosis = run_dir(task, previous["run_id"]) / "diagnosis.json"
    if diagnosis.exists():
        lesson["context"] = read_json(diagnosis)["observations"]
    if run.get("status") == "completed":
        lesson.update(after=run["selection_score"], delta=run["selection_score"] - previous["selection_score"],
                      verdict="kept" if run["selection_score"] > previous["selection_score"] else "rejected")
    else:
        lesson.update(verdict="failed", error=str(run.get("error", ""))[:300])
    return lesson


def add(lesson):
    known = {(r["task"], r.get("run_id")) for r in load() if r["type"] == "experiment"}
    if lesson["type"] == "experiment" and (lesson["task"], lesson["run_id"]) in known:
        return None
    append_event(path(), "lesson", **lesson)
    return lesson


def learn_from_history(task):
    """Backfill lessons from a task's existing runs in time order, against the best-so-far."""
    added, best = [], None
    for run in list_runs(task):
        if "recipe" not in run:
            continue
        if best is not None:
            lesson = add(experiment_lesson(task, run, best))
            if lesson:
                added.append(describe(lesson))
        if run.get("status") == "completed" and (best is None or run["selection_score"] > best["selection_score"]):
            best = run
    return {"task": task, "lessons_added": len(added), "lessons": added}


def scoreboard(rows=None):
    """Per lever: how often it was tried, kept, and its mean validation change."""
    groups = defaultdict(list)
    for row in rows if rows is not None else load():
        if row["type"] == "experiment" and row["change"]:
            groups[change_label(row["change"])].append(row)
    board = []
    for label, items in groups.items():
        measured = [r["delta"] for r in items if r["verdict"] != "failed"]
        board.append({"change": label, "tried": len(items), "kept": sum(r["verdict"] == "kept" for r in items),
                      "failed": sum(r["verdict"] == "failed" for r in items),
                      "mean_delta": sum(measured) / len(measured) if measured else None,
                      "tasks": sorted({r["task"] for r in items})})
    return sorted(board, key=lambda b: (-b["kept"], -(b["mean_delta"] or -1)))


def best_recipes(kind, exclude):
    """Winning validation recipe of every other task of the same kind; warm starts do not transfer."""
    recipes = []
    for other, info in TASKS.items():
        if other == exclude or info["kind"] != kind:
            continue
        done = [r for r in list_runs(other) if r.get("status") == "completed"]
        if done:
            best = max(done, key=lambda r: r["selection_score"])
            recipe = {k: v for k, v in best["recipe"].items() if k not in IGNORED | {"warm_start"}}
            recipes.append({"task": other, "run_id": best["run_id"], "metric": best["selection_metric"],
                            "score": best["selection_score"], "recipe": recipe})
    return recipes


def recall(task, limit=12):
    kind = task_info(task)["kind"]
    rows = load()
    other = [r for r in rows if r["task"] != task]
    # Same kind of problem first, then larger measured effects first.
    other.sort(key=lambda r: (r["kind"] != kind, r["type"] != "insight", -abs(r.get("delta") or 0)))
    return {"lessons_from_other_tasks": [describe(r) for r in other[:limit]],
            "lessons_from_this_task": [describe(r) for r in rows if r["task"] == task][-limit:],
            "lever_scoreboard": scoreboard(rows),
            "best_recipes_from_similar_tasks": best_recipes(kind, task),
            "note": "Lessons are measured on other datasets. Treat them as hypotheses and re-test them here."}


def record_insight(task, text, evidence, session_id=None):
    text = " ".join(str(text).split())
    if not 10 <= len(text) <= 400:
        raise ValueError("An insight must be 10-400 characters")
    known = {r["run_id"] for r in list_runs(task)}
    if not evidence or any(e not in known for e in evidence):
        raise ValueError(f"Evidence must be existing {task} run IDs: {sorted(known)}")
    lesson = {"type": "insight", "task": task, "kind": task_info(task)["kind"], "text": text,
              "evidence": list(evidence), "session_id": session_id}
    add(lesson)
    return lesson
