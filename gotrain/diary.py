"""Chat-style experiment diary built only from logged evidence.

Every message comes from a session's events.jsonl and the runs' own records.
Nothing is shown that a tool did not measure or the agent did not say. The same
page renders a saved diary (state embedded) and the live `gotrain ui` app.
"""
import base64
import json
import time
from datetime import datetime
from pathlib import Path

from .config import runs_root
from .page import PAGE
from .store import read_json, task_dir

METRIC_NAMES = {"accuracy": "accuracy", "mean_iou": "mean IoU"}


def events(session_dir):
    rows = []
    path = Path(session_dir) / "events.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A line can be mid-write while the agent is running.
    return rows


def image_uri(path):
    path = Path(path)
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode() if path.exists() else None


def live_progress(task, hypothesis):
    """Epoch progress of the run that is training right now for this hypothesis."""
    for directory in sorted(task_dir(task).glob("*/config.json"), reverse=True):
        config = read_json(directory)
        if config.get("hypothesis") != hypothesis:
            continue
        history_path = directory.parent / "history.json"
        history = read_json(history_path) if history_path.exists() else []
        if not history:
            return "Preparing data and model…"
        metric = config["selection_metric"]
        last = history[-1]
        best = max(r["validation"][metric] for r in history)
        return (f"Epoch {last['epoch']} of {config['recipe']['epochs']} · validation {METRIC_NAMES[metric]} "
                f"{last['validation'][metric]:.4f} (best so far {best:.4f})")
    return "Starting…"


def recipe_text(recipe_json):
    try:
        recipe = json.loads(recipe_json)
    except (TypeError, json.JSONDecodeError):
        return ""
    keys = ["model", "image_size", "epochs", "learning_rate", "balanced_loss", "dice", "focus_crops", "warm_start"]
    return " · ".join(f"{k}={recipe[k]:g}" if isinstance(recipe[k], float) else f"{k}={recipe[k]}" for k in keys if k in recipe and recipe[k] not in (None, False))


def message(icon, title, detail="", tone="info", images=None):
    return {"icon": icon, "title": title, "detail": detail, "tone": tone, "images": images or []}


def diagnosis_images(task, result, limit=2):
    images = []
    for example in result.get("worst_examples", [])[:limit]:
        uri = image_uri(task_dir(task) / result["run_id"] / example["image"])
        if not uri:
            continue
        if "foreground_iou" in example:
            caption = f"Validation #{example['index']} · input · true mask · prediction · IoU {example['foreground_iou']:.2f}"
        else:
            caption = (f"Validation #{example['index']} · truth {example['truth']} · predicted "
                       f"{example['predicted']} ({example['confidence']:.0%} sure)")
        images.append({"src": uri, "caption": caption})
    return images


def entries(rows, task=None):
    """Translate logged events into chat messages."""
    out, pending = [], None
    for row in rows:
        kind = row["event"]
        if kind == "session_started":
            metric = "accuracy" if row.get("kind") == "classification" else "mean IoU"
            out.append(message("🚀", f"Goal: best validation {metric} on {row['task']}",
                               f"Budget: {row['experiments']} experiments, {row['minutes']:g} minutes.", "goal"))
        elif kind == "experiment_requested":
            pending = row
            out.append(message("🧪", f"Experiment {row['attempt']}: {row['hypothesis']}",
                               recipe_text(row["recipe_json"]), "plan"))
        elif kind == "tool_result":
            tool, result = row["tool"], row.get("result") or {}
            if tool == "inspect_dataset":
                counts = result.get("train_sample_class_counts") or []
                share = ""
                if len(counts) == 2 and sum(counts):
                    share = f" Target covers {counts[1] / sum(counts):.1%} of sampled training labels."
                out.append(message("🔍", "Inspected the dataset",
                                   f"{result.get('train_samples')} training and {result.get('valid_samples')} "
                                   f"validation samples, {result.get('bands')} bands.{share}"))
            elif tool == "recall_lessons":
                recalled = result.get("lessons_from_other_tasks", []) + result.get("lessons_from_this_task", [])
                recipes = [f"Winning recipe on {r['task']}: {r['metric']} {r['score']:.4f}"
                           for r in result.get("best_recipes_from_similar_tasks", [])]
                if recalled or recipes:
                    out.append(message("📚", f"Recalled {len(recalled)} lessons from memory", (recipes + recalled)[:5]))
                else:
                    out.append(message("📚", "Memory is empty", "This is the first task GoTrain learns from."))
            elif tool == "record_lesson" and result.get("type") == "insight":
                out.append(message("💡", "Lesson learned for future tasks",
                                   f"{result['text']} (evidence: {', '.join(result['evidence'])})", "ok"))
            elif tool == "diagnose_errors" and "observations" in result:
                out.append(message("🔬", f"Diagnosed run {result['run_id']}", result["observations"], "info",
                                   diagnosis_images(task or result["task"], result)))
            elif tool == "run_experiment":
                pending = None
                status = result.get("status")
                if status == "completed":
                    metric = METRIC_NAMES.get(result.get("selection_metric"), "score")
                    verdict = result.get("verdict")
                    kept = verdict == "kept"
                    verdict_text = ("kept as new best" if kept else "rejected") if verdict in {"kept", "rejected"} else "completed (verdict not recorded)"
                    previous = result.get("previous_best_score")
                    change = f" (previous best {previous:.4f})" if previous is not None else ""
                    remaining = (f" · {result['minutes_remaining']:.0f} min of budget left"
                                 if result.get("minutes_remaining") is not None else "")
                    out.append(message("📈" if kept else "📉" if verdict == "rejected" else "🧪",
                                       f"Validation {metric} {result['selection_score']:.4f}{change}: "
                                       + verdict_text,
                                       f"Best epoch {result.get('best_epoch')} · "
                                       f"{result.get('elapsed_seconds', 0) / 60:.1f} min" + remaining
                                       + (f"\nMemory: {result['lesson_recorded']}" if result.get("lesson_recorded") else ""),
                                       verdict if verdict in {"kept", "rejected"} else "info"))
                elif status == "budget_exhausted":
                    out.append(message("⏱️", "Budget used up", "No more training runs in this session.", "plan"))
                else:
                    out.append(message("⚠️", "Experiment failed", result.get("error", ""), "rejected"))
            elif tool == "publish_best_model":
                out.append(message("📦", "Published the best model", result.get("path", ""), "plan"))
            elif tool == "verify_model_download":
                out.append(message("✅", "Re-downloaded it and verified every checksum", result.get("path", ""), "ok"))
        elif kind == "session_completed":
            out.append(message("🏁", "Final summary", row.get("response", ""), "goal"))
        elif kind == "session_failed":
            out.append(message("⚠️", "Session stopped with an error", row.get("error", ""), "rejected"))
    return out, pending


def showcase_message(session_dir):
    path = Path(session_dir) / "showcase" / "showcase.json"
    if not path.exists():
        return None
    data = read_json(path)
    metric = METRIC_NAMES.get(data["selection_metric"], "score")
    same = data["before_run"] == data["after_run"]
    if data["kind"] == "segmentation":
        legend = "input · first model · best model · true mask"
    else:
        legend = "input image"
    images = []
    for example in data["examples"]:
        if data["kind"] == "segmentation":
            def fmt(v):
                return "n/a" if v is None else f"{v:.2f}"
            caption = (f"Validation #{example['index']} · {legend} · foreground IoU "
                       f"{fmt(example['before_iou'])} → {fmt(example['after_iou'])}")
        else:
            caption = (f"Validation #{example['index']} · truth {example['truth']} · first model "
                       f"{example['before']} · best model {example['after']}")
        uri = image_uri(path.parent / example["image"])
        if uri:
            images.append({"src": uri, "caption": caption})
    detail = ("No experiment beat the first model, so it was kept. These are its hardest validation images."
              if same else "Same validation images, before and after. " + data["selection_note"])
    return message("🖼️", f"Before → after: validation {metric} {data['before_score']:.4f} → {data['after_score']:.4f}",
                   detail, "goal", images)


def memory():
    from .lessons import load, scoreboard
    rows = load()
    return {"lessons": len(rows), "tasks": len({r["task"] for r in rows}), "scoreboard": scoreboard(rows)[:8]}


def state(session_dir):
    """Everything the chat page shows for one session, as plain data."""
    session_dir = Path(session_dir)
    rows = events(session_dir)
    if not rows:
        raise ValueError(f"No events in {session_dir}")
    start = rows[0]
    task = start["task"]
    done = any(r["event"] in {"session_completed", "session_failed"} for r in rows)
    messages, pending = entries(rows, task)
    showcase = showcase_message(session_dir)
    if showcase:
        # Place the before/after picture just before the final summary.
        at = next((i for i, m in enumerate(messages) if m["icon"] == "🏁"), len(messages))
        messages.insert(at, showcase)
    began = datetime.fromisoformat(start["time"])
    ended = datetime.fromisoformat(rows[-1]["time"]) if done else datetime.now(began.tzinfo)
    live = None
    if pending and not done:
        live = {"title": f"Training experiment {pending['attempt']}…",
                "detail": live_progress(task, pending["hypothesis"])}
    elif not done:
        live = {"title": "Thinking…", "detail": "Reading results and choosing the next step."}
    return {"session_id": session_dir.name, "task": task, "done": done, "started_at": start["time"],
            "request": start.get("request") or f"Train the best {task} model you can in {start['minutes']:g} "
                                                f"minutes with {start['experiments']} experiments.",
            "minutes": start["minutes"], "used_minutes": (ended - began).total_seconds() / 60,
            "experiments": start["experiments"],
            "attempts": sum(r["event"] == "experiment_requested" for r in rows),
            "messages": messages, "live": live, "memory": memory()}


def sessions():
    """Session list for the sidebar, newest first."""
    out = []
    for path in sorted((runs_root() / "sessions").glob("session_*/events.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            first = json.loads(path.read_text().splitlines()[0])
        except (IndexError, json.JSONDecodeError):
            continue
        done = '"session_completed"' in path.read_text() or '"session_failed"' in path.read_text()
        recent = time.time() - path.stat().st_mtime < 2 * 3600  # a crashed process stops writing
        out.append({"session_id": path.parent.name, "task": first.get("task"), "started_at": first.get("time"),
                    "live": not done and recent})
    return out


def page(embedded=None):
    """The chat page. With embedded state it is a saved diary; without it, it polls the live app."""
    data = "null" if embedded is None else json.dumps(embedded).replace("<", "\\u003c")
    refresh = '<meta http-equiv="refresh" content="10">' if embedded and not embedded["state"]["done"] else ""
    return PAGE.replace("__REFRESH__", refresh).replace("__STATE__", data)


def build(session_dir):
    session_dir = Path(session_dir)
    document = page({"state": state(session_dir), "sessions": sessions()})
    path = session_dir / "diary.html"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)
    latest = runs_root() / "live-diary.html"
    latest.write_text(document, encoding="utf-8")
    return {"diary": str(path), "latest": str(latest)}


def latest_session():
    found = sessions()
    if not found:
        raise ValueError("No agent sessions yet")
    return runs_root() / "sessions" / found[0]["session_id"]

