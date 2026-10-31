"""A single Strands agent with bounded training tools. No test tool is exposed."""
import json
import os
import threading
import time
import uuid

from .config import Recipe, task_info, runs_root
from .store import append_event, best_run, list_runs, now, write_json


def run_agent(task, experiments=3, minutes=60, request=None):
    from strands import Agent, tool
    from strands.models import BedrockModel
    from . import data, diagnose, diary, engine, lessons, registry

    if not 1 <= experiments <= 12 or not 1 <= minutes <= 480:
        raise ValueError("Use 1–12 experiments and 1–480 minutes")
    info = task_info(task)
    session_id = "session_" + uuid.uuid4().hex[:12]
    directory = runs_root() / "sessions" / session_id
    directory.mkdir(parents=True)
    log = directory / "events.jsonl"
    started = time.monotonic()
    attempts, tool_calls = 0, 0
    session_runs = []

    diary_lock = threading.Lock()

    def refresh_diary():
        try:
            with diary_lock:
                diary.build(directory)
        except Exception as error:  # The diary is a view; it must never stop training.
            print(f"diary not updated: {error!r}", flush=True)

    def record(name, inputs, result):
        nonlocal tool_calls
        tool_calls += 1
        append_event(log, "tool_result", tool=name, inputs=inputs, result=result)
        refresh_diary()
        return result

    @tool
    def inspect_dataset() -> dict:
        """Inspect the selected dataset, training class balance and train/valid shapes. Does not read test data."""
        return record("inspect_dataset", {}, data.inspect(task))

    @tool
    def run_experiment(recipe_json: str, hypothesis: str) -> dict:
        """Train one bounded experiment and return measured validation metrics.

        Args:
            recipe_json: JSON object with Recipe fields. Models: small/resnet18/resnet34.
                Fields: pretrained, epochs (1-100), batch_size (2-128), image_size (64-512),
                learning_rate, weight_decay, augmentation, balanced_loss, dice (segmentation only),
                seed, workers (0-8), patience, stats_samples, amp, max_minutes.
                warm_start: optional completed run ID to fine-tune its best checkpoint.
                A warm start must retain the same model and image_size.
            hypothesis: A short testable reason for the chosen change, tied to observations.
        """
        nonlocal attempts
        remaining = minutes * 60 - (time.monotonic() - started)
        if attempts >= experiments or remaining <= 0:
            return record("run_experiment", {}, {"status": "budget_exhausted", "message": "Stop training; summarize results."})
        inputs = {"recipe_json": recipe_json, "hypothesis": hypothesis}
        champion, existing = None, {r["run_id"] for r in list_runs(task)}
        try:
            recipe = Recipe(**json.loads(recipe_json))
            recipe.max_minutes = min(recipe.max_minutes, remaining / 60)
            recipe.validate(info["kind"])
            attempts += 1  # Failures also consume an experiment slot.
            completed = [r for r in list_runs(task) if r.get("status") == "completed"]
            champion = best_run(task) if completed else None
            previous = champion["selection_score"] if champion else None
            append_event(log, "experiment_requested", attempt=attempts, **inputs)
            refresh_diary()
            result = engine.train(task, recipe, hypothesis)
            session_runs.append(result["run_id"])
            # Strictly better on validation, matching best_run's tie rule (earlier run wins).
            result = {**result, "previous_best_score": previous,
                      "verdict": "kept" if previous is None or result["selection_score"] > previous else "rejected",
                      "attempts_remaining": experiments - attempts,
                      "minutes_remaining": max(0.0, minutes - (time.monotonic() - started) / 60)}
            if champion:
                lesson = lessons.add(lessons.experiment_lesson(task, result, champion, session_id))
                result["lesson_recorded"] = lessons.describe(lesson) if lesson else None
            return record("run_experiment", inputs, result)
        except Exception as error:
            # Only the run this attempt created; recipe errors before training create no run.
            failed = [r for r in list_runs(task) if r["run_id"] not in existing and r.get("status") == "failed"]
            if champion and failed and "recipe" in failed[-1]:
                lessons.add(lessons.experiment_lesson(task, failed[-1], champion, session_id))
            return record("run_experiment", inputs, {"status": "failed", "error": str(error),
                          "hint": "For CUDA OOM reduce batch_size/image_size; failed runs do not change the best model."})

    @tool
    def diagnose_errors(run_id: str = "") -> dict:
        """Measure where a completed run fails on VALIDATION data: object detection by size,
        missed/extra foreground, most confused classes, worst examples, and candidate recipe levers.

        Args:
            run_id: completed run to analyse; empty means the current best run.
        """
        try:
            result = diagnose.diagnose(task, run_id or None)
        except Exception as error:
            return record("diagnose_errors", {"run_id": run_id}, {"status": "failed", "error": str(error)})
        return record("diagnose_errors", {"run_id": run_id}, result)

    @tool
    def recall_lessons() -> dict:
        """Read GoTrain's memory: measured lessons from earlier experiments on other tasks and this one,
        a scoreboard of which changes helped, and the winning recipes of similar tasks."""
        return record("recall_lessons", {}, lessons.recall(task))

    @tool
    def record_lesson(insight: str, evidence_run_ids: list[str]) -> dict:
        """Save one transferable conclusion for future tasks. It must be supported by this task's runs.

        Args:
            insight: one or two sentences, e.g. "On sparse small targets, focus_crops raised mean IoU by 0.03".
            evidence_run_ids: run IDs from this task whose measured results support the insight.
        """
        inputs = {"insight": insight, "evidence_run_ids": evidence_run_ids}
        try:
            return record("record_lesson", inputs, lessons.record_insight(task, insight, evidence_run_ids, session_id))
        except ValueError as error:
            return record("record_lesson", inputs, {"status": "rejected", "error": str(error)})

    @tool
    def compare_experiments() -> dict:
        """Read recorded validation experiments and the best model. Never returns test results."""
        rows = list_runs(task)
        result = {"runs": rows, "attempts_used": attempts, "attempts_limit": experiments,
                  "minutes_remaining": max(0, minutes - (time.monotonic() - started) / 60)}
        if any(r["status"] == "completed" for r in rows):
            result["best"] = best_run(task)
        return record("compare_experiments", {}, result)

    @tool
    def publish_best_model() -> dict:
        """Publish best validation checkpoint and versioned logs to filesystem and optional GOTRAIN_S3_BUCKET."""
        chosen = best_run(task)
        return record("publish_best_model", {}, registry.act("publish-model", task, chosen["run_id"]))

    @tool
    def publish_dataset() -> dict:
        """Upload the selected dataset archive to versioned filesystem storage and optional GOTRAIN_S3_BUCKET."""
        return record("publish_dataset", {}, registry.act("publish-data", task))

    @tool
    def verify_model_download() -> dict:
        """Download a previously published best model and verify every artifact checksum."""
        chosen = best_run(task)
        return record("verify_model_download", {}, registry.act("restore-model", task, chosen["run_id"]))

    system = f"""You are GoTrain, an autonomous GEO-Bench ML scientist.
Work ONLY on {task}, a {info['kind']} task. Inspect the data before training.
Your objective is to improve validation {'accuracy' if info['kind'] == 'classification' else 'mean_iou'}.
Work as an error-driven scientist with a memory:
0. After inspect_dataset, call recall_lessons. A winning recipe from a similar task is a good first
   experiment; say which lesson you use in the hypothesis. Lessons from other datasets are hints, not
   proof: re-test them here and never cite another task's score as this task's result.
1. Train a baseline (or the recalled recipe).
2. diagnose_errors on the latest completed run before every new experiment.
3. Pick ONE measured observation from the diagnosis and test ONE change aimed at it. The hypothesis
   must quote the number it targets, e.g. "Small objects are 31% detected vs 90% for large; test focus_crops".
4. run_experiment reports verdict kept/rejected against the previous best. A rejected change is a valid
   finding: say so, do not retry the same recipe, and choose a different lever.
For segmentation, focus_crops=true zooms half of the training samples onto foreground objects
(they appear 2x larger); it is a candidate when small objects are detected less often.
Read comparison history; prefer measured evidence over speculation. Never invent scores or claim an
unmeasured cause of failure. First try pretrained resnet18, then change a small number of settings
based on results. resnet34 is a heavier option. Use train-only normalization supplied by the engine.
For segmentation start image_size=256, batch_size=8, dice=true; for classification start
image_size=128, batch_size=32. Set workers=2 to overlap dataset reading with GPU computation.
Start learning_rate=0.0003, epochs=15, augmentation=true.
For severe imbalance test balanced_loss. Keep the seed fixed for comparable experiments.
When a strong run is still improving, consider warm_start with its run ID and a lower learning_rate
instead of restarting from pretrained weights. Keep model and image_size unchanged for warm starts.
You have at most {experiments} training attempts and {minutes} minutes (soft limit, validation finishes).
Call tools sequentially. On budget_exhausted, do not call run_experiment again.
After training, compare experiments, publish the best model, and verify its download.
Dataset publishing is optional because archive copies can take time. Do not finalize or request test data.
Before finishing, call record_lesson 1-3 times with transferable conclusions, each citing run IDs.
Finish with observed scores, each hypothesis and its verdict, limitations, and the selected run ID. Tools provide evidence;
the training history is authoritative. Do not claim the 75% historical baseline is beaten unless measured.
"""
    append_event(log, "session_started", task=task, kind=info["kind"], experiments=experiments, minutes=minutes,
                 request=request,
                 system_prompt=system, model=os.environ.get("GOTRAIN_BEDROCK_MODEL", "eu.anthropic.claude-sonnet-5-5"))
    model = BedrockModel(
        model_id=os.environ.get("GOTRAIN_BEDROCK_MODEL", "eu.anthropic.claude-sonnet-5-5"),
        region_name=os.environ.get("AWS_REGION", "eu-central-1"),
    )
    agent = Agent(model=model, system_prompt=system,
                  tools=[inspect_dataset, recall_lessons, run_experiment, diagnose_errors, record_lesson,
                         compare_experiments,
                         publish_best_model, publish_dataset, verify_model_download])
    refresh_diary()
    stop = threading.Event()

    def keep_diary_live():
        while not stop.wait(10):
            refresh_diary()  # Shows epoch progress while a run trains.

    threading.Thread(target=keep_diary_live, daemon=True).start()
    try:
        result = agent("Inspect the dataset and execute the budgeted experiment workflow now.")
        stop.set()
        if session_runs:
            try:
                before, after = diagnose.first_and_best(task, session_runs)
                diagnose.showcase(task, before, after, directory / "showcase")
            except Exception as error:
                append_event(log, "showcase_failed", error=repr(error))
        transcript = agent.messages
        write_json(directory / "messages.json", transcript)
        write_json(directory / "session.json", {"task": task, "session_id": session_id,
                   "attempts": attempts, "completed_runs": session_runs, "tool_calls": tool_calls,
                   "elapsed_seconds": time.monotonic() - started, "finished_at": now(), "response": str(result)})
        append_event(log, "session_completed", response=str(result))
        refresh_diary()
        return {"session_id": session_id, "path": str(directory), "response": str(result),
                "diary": str(directory / "diary.html")}
    except Exception as error:
        stop.set()
        write_json(directory / "messages.json", agent.messages)
        append_event(log, "session_failed", error=repr(error))
        refresh_diary()
        raise

