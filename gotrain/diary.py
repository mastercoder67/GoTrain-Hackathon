"""Live experiment diary: one self-refreshing HTML page built only from logged evidence.

Every entry comes from the session's events.jsonl and the runs' own records.
Nothing is written here that a tool did not measure or the agent did not say.
"""
import base64
import html
import json
from datetime import datetime
from pathlib import Path

from .config import runs_root
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
        if not history_path.exists():
            return "Preparing data and model…"
        history = read_json(history_path)
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
    return " · ".join(f"{k}={recipe[k]}" for k in keys if k in recipe and recipe[k] not in (None, False))


def entries(rows):
    """Translate logged events into diary entries: (icon, title, detail, css class)."""
    out, pending = [], None
    for row in rows:
        kind = row["event"]
        if kind == "session_started":
            metric = "accuracy" if row.get("kind") == "classification" else "mean IoU"
            out.append(("🚀", f"Goal: best validation {metric} on {row['task']}",
                        f"Budget: {row['experiments']} experiments, {row['minutes']:g} minutes.", "goal"))
        elif kind == "experiment_requested":
            pending = row
            out.append(("🧪", f"Experiment {row['attempt']}: {row['hypothesis']}", recipe_text(row["recipe_json"]), "plan"))
        elif kind == "tool_result":
            tool, result = row["tool"], row.get("result") or {}
            if tool == "inspect_dataset":
                counts = result.get("train_sample_class_counts") or []
                share = ""
                if len(counts) == 2 and sum(counts):
                    share = f" Target covers {counts[1] / sum(counts):.1%} of sampled training labels."
                out.append(("🔍", "Inspected the dataset",
                            f"{result.get('train_samples')} training and {result.get('valid_samples')} validation "
                            f"samples, {result.get('bands')} bands.{share}", "look"))
            elif tool == "recall_lessons":
                recalled = result.get("lessons_from_other_tasks", []) + result.get("lessons_from_this_task", [])
                recipes = [f"Winning recipe on {r['task']}: {r['metric']} {r['score']:.4f}"
                           for r in result.get("best_recipes_from_similar_tasks", [])]
                if recalled or recipes:
                    out.append(("📚", f"Recalled {len(recalled)} lessons from memory", (recipes + recalled)[:5], "look"))
                else:
                    out.append(("📚", "Memory is empty", "This is the first task GoTrain learns from.", "look"))
            elif tool == "record_lesson":
                if result.get("type") == "insight":
                    out.append(("💡", "Lesson learned for future tasks",
                                f"{result['text']} (evidence: {', '.join(result['evidence'])})", "kept"))
            elif tool == "diagnose_errors" and "observations" in result:
                out.append(("🔬", f"Diagnosed run {result['run_id']}", result["observations"], "look"))
            elif tool == "run_experiment":
                pending = None
                status = result.get("status")
                if status == "completed":
                    metric = METRIC_NAMES.get(result.get("selection_metric"), "score")
                    kept = result.get("verdict") == "kept"
                    previous = result.get("previous_best_score")
                    change = f" (previous best {previous:.4f})" if previous is not None else ""
                    out.append(("📈" if kept else "📉",
                                f"Validation {metric} {result['selection_score']:.4f}{change}: "
                                f"{'kept as new best' if kept else 'rejected'}",
                                f"Best epoch {result.get('best_epoch')} · {result.get('elapsed_seconds', 0) / 60:.1f} min · "
                                f"{result.get('minutes_remaining', 0):.0f} min of budget left"
                                + (f"\nMemory: {result['lesson_recorded']}" if result.get("lesson_recorded") else ""),
                                "kept" if kept else "rejected"))
                elif status == "budget_exhausted":
                    out.append(("⏱️", "Budget used up", "No more training runs in this session.", "plan"))
                else:
                    out.append(("⚠️", "Experiment failed", result.get("error", ""), "rejected"))
            elif tool == "publish_best_model":
                out.append(("📦", "Published the best model", result.get("path", ""), "plan"))
            elif tool == "verify_model_download":
                out.append(("✅", "Re-downloaded it and verified every checksum", result.get("path", ""), "kept"))
        elif kind == "session_completed":
            out.append(("🏁", "Agent's final summary", row.get("response", ""), "goal"))
        elif kind == "session_failed":
            out.append(("⚠️", "Session stopped with an error", row.get("error", ""), "rejected"))
    return out, pending


def showcase_html(session_dir):
    path = Path(session_dir) / "showcase" / "showcase.json"
    if not path.exists():
        return ""
    data = read_json(path)
    metric = METRIC_NAMES.get(data["selection_metric"], "score")
    same = data["before_run"] == data["after_run"]
    head = (f"<h2>Before → after</h2><p class='big'>Validation {metric}: "
            f"<b>{data['before_score']:.4f}</b> → <b>{data['after_score']:.4f}</b></p>")
    if same:
        head += "<p>No experiment beat the first model, so it was kept. Its hardest validation images:</p>"
    if data["kind"] == "segmentation":
        legend = "Input · first model · best model · true mask" if not same else "Input · model · model · true mask"
    else:
        legend = "Input image"
    cards = []
    for example in data["examples"]:
        uri = image_uri(path.parent / example["image"])
        if data["kind"] == "segmentation":
            def fmt(v):
                return "n/a" if v is None else f"{v:.2f}"
            caption = (f"Validation #{example['index']} · foreground IoU {fmt(example['before_iou'])} → "
                       f"{fmt(example['after_iou'])}")
        else:
            caption = (f"Validation #{example['index']} · truth {example['truth']} · first model "
                       f"{example['before']} · best model {example['after']}")
        cards.append(f"<figure><img src='{uri}' alt='{html.escape(caption)}'><figcaption>{html.escape(caption)}"
                     f"</figcaption></figure>")
    return (head + f"<p class='muted'>{html.escape(legend)}. {html.escape(data['selection_note'])}</p>" + "".join(cards))


def latest_diagnosis_html(task, rows):
    for row in reversed(rows):
        if row["event"] == "tool_result" and row.get("tool") == "diagnose_errors" and "worst_examples" in (row.get("result") or {}):
            result = row["result"]
            directory = task_dir(task) / result["run_id"]
            figures = []
            for example in result["worst_examples"][:3]:
                uri = image_uri(directory / example["image"])
                if not uri:
                    continue
                if "foreground_iou" in example:
                    caption = f"Validation #{example['index']} · input · truth · prediction · IoU {example['foreground_iou']:.2f}"
                else:
                    caption = (f"Validation #{example['index']} · truth {example['truth']} · predicted "
                               f"{example['predicted']} ({example['confidence']:.0%} sure)")
                figures.append(f"<figure><img src='{uri}' alt='{html.escape(caption)}'><figcaption>"
                               f"{html.escape(caption)}</figcaption></figure>")
            if figures:
                return "<h2>Where the model fails now</h2>" + "".join(figures)
    return ""


def memory_html():
    from .lessons import load, scoreboard
    rows = load()
    board = scoreboard(rows)
    if not board:
        return ""
    lines = []
    for b in board[:8]:
        mean = "—" if b["mean_delta"] is None else f"{b['mean_delta']:+.4f}"
        lines.append(f"<tr><td>{html.escape(b['change'])}</td><td>{b['kept']}/{b['tried']}</td><td>{mean}</td>"
                     f"<td>{html.escape(', '.join(b['tasks']))}</td></tr>")
    tasks = len({r["task"] for r in rows})
    return (f"<h2>What GoTrain has learned</h2><p class='muted'>{len(rows)} lessons from {tasks} "
            f"task{'s' if tasks != 1 else ''}. Kept = beat the best validation score at the time.</p>"
            "<table><tr><th>Change</th><th>Kept</th><th>Mean Δ</th><th>Tasks</th></tr>" + "".join(lines) + "</table>")


STYLE = """
:root{--bg:#f6f7f4;--card:#fff;--ink:#17201c;--muted:#5d6b64;--line:#dfe4df;--accent:#0b7a5c;--bad:#b4473a;--plan:#3b5bab}
@media (prefers-color-scheme:dark){:root{--bg:#101614;--card:#18211d;--ink:#e8efe9;--muted:#9aaba2;--line:#2a3631;--accent:#4cc79c;--bad:#ec8a7c;--plan:#94adf0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:28px 16px;display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:28px}
header{grid-column:1/-1}h1{margin:0;font-size:30px}h2{font-size:18px;margin:22px 0 10px}.muted,figcaption{color:var(--muted);font-size:13px}
.tag{color:var(--accent);font-weight:600;letter-spacing:.12em;text-transform:uppercase;font-size:12px}
.budget{display:flex;gap:18px;flex-wrap:wrap;margin-top:12px}.meter{flex:1;min-width:200px}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}
ol{list-style:none;margin:0;padding:0}li{display:flex;gap:12px;background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:10px;padding:12px 14px;margin:0 0 10px}
li.kept{border-left-color:var(--accent)}li.rejected{border-left-color:var(--bad)}li.plan{border-left-color:var(--plan)}li.live{border-left-color:var(--plan);animation:pulse 1.6s infinite}
@keyframes pulse{50%{opacity:.65}}li b{display:block}li .icon{font-size:20px;line-height:1.3}li ul{margin:4px 0 0;padding-left:18px;color:var(--muted)}
li p{margin:2px 0 0;color:var(--muted);white-space:pre-wrap;overflow-wrap:anywhere}
figure{margin:0 0 16px}img{max-width:100%;border-radius:8px;border:1px solid var(--line);image-rendering:pixelated}
.big{font-size:22px}.big b{color:var(--accent)}
table{width:100%;border-collapse:collapse;font-size:13px;background:var(--card);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);overflow-wrap:anywhere}th{color:var(--muted);font-weight:600}
@media (max-width:820px){main{grid-template-columns:1fr}}
"""


def build(session_dir):
    session_dir = Path(session_dir)
    rows = events(session_dir)
    if not rows:
        raise ValueError(f"No events in {session_dir}")
    start = rows[0]
    task = start["task"]
    done = any(r["event"] in {"session_completed", "session_failed"} for r in rows)
    items, pending = entries(rows)
    began = datetime.fromisoformat(start["time"])
    ended = datetime.fromisoformat(rows[-1]["time"]) if done else datetime.now(began.tzinfo)
    used_minutes = (ended - began).total_seconds() / 60
    attempts = sum(r["event"] == "experiment_requested" for r in rows)
    html_items = []
    for icon, title, detail, css in items:
        if isinstance(detail, list):
            body = "<ul>" + "".join(f"<li style='all:revert'>{html.escape(d)}</li>" for d in detail) + "</ul>"
        else:
            body = f"<p>{html.escape(str(detail))}</p>" if detail else ""
        html_items.append(f"<li class='{css}'><span class='icon'>{icon}</span><div><b>{html.escape(title)}</b>{body}</div></li>")
    if pending and not done:
        html_items.append(f"<li class='live'><span class='icon'>⏳</span><div><b>Training experiment "
                          f"{pending['attempt']}…</b><p>{html.escape(live_progress(task, pending['hypothesis']))}</p></div></li>")
    minutes, experiments = start["minutes"], start["experiments"]
    status = "Finished" if done else "Live · refreshes every 10 s"
    visuals = showcase_html(session_dir) or latest_diagnosis_html(task, rows) or \
        "<h2>Pictures</h2><p class='muted'>Prediction examples appear after the first diagnosis.</p>"
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">{'' if done else '<meta http-equiv="refresh" content="10">'}
<title>GoTrain diary</title><style>{STYLE}</style></head><body><main>
<header><span class="tag">GoTrain · experiment diary · {html.escape(status)}</span><h1>{html.escape(task)}</h1>
<div class="budget"><div class="meter"><span class="muted">Time {used_minutes:.0f} of {minutes:g} min</span>
<div class="bar"><i style="width:{min(100, 100 * used_minutes / minutes):.0f}%"></i></div></div>
<div class="meter"><span class="muted">Experiments {attempts} of {experiments}</span>
<div class="bar"><i style="width:{min(100, 100 * attempts / experiments):.0f}%"></i></div></div></div></header>
<section><h2>Diary</h2><ol>{''.join(html_items)}</ol>
<p class="muted">Validation split only. The test split is never opened during a session.</p></section>
<section>{visuals}{memory_html()}</section></main></body></html>"""
    path = session_dir / "diary.html"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)
    latest = runs_root() / "live-diary.html"
    latest.write_text(document, encoding="utf-8")
    return {"diary": str(path), "latest": str(latest)}


def latest_session():
    sessions = sorted((runs_root() / "sessions").glob("session_*/events.jsonl"), key=lambda p: p.stat().st_mtime)
    if not sessions:
        raise ValueError("No agent sessions yet")
    return sessions[-1].parent
