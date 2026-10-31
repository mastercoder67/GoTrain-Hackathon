"""A single Strands agent with bounded training tools. No test tool is exposed."""
import json
import os
import time
import uuid

from .config import Recipe, task_info, runs_root
from .store import append_event, best_run, list_runs, now, write_json


def run_agent(task, experiments=3, minutes=60):
    from strands import Agent, tool
    from strands.models import BedrockModel
    from . import artifacts, data, engine

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

    def record(name, inputs, result):
        nonlocal tool_calls
        tool_calls += 1
        append_event(log, "tool_result", tool=name, inputs=inputs, result=result)
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
        try:
            recipe = Recipe(**json.loads(recipe_json))
            recipe.max_minutes = min(recipe.max_minutes, remaining / 60)
            recipe.validate(info["kind"])
            attempts += 1  # Failures also consume an experiment slot.
            append_event(log, "experiment_requested", attempt=attempts, **inputs)
            result = engine.train(task, recipe, hypothesis)
            session_runs.append(result["run_id"])
            return record("run_experiment", inputs, result)
        except Exception as error:
            return record("run_experiment", inputs, {"status": "failed", "error": str(error),
                          "hint": "For CUDA OOM reduce batch_size/image_size; failed runs do not change the best model."})

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
        return record("publish_best_model", {}, artifacts.publish_model(task, chosen["run_id"], os.environ.get("GOTRAIN_S3_BUCKET")))

    @tool
    def publish_dataset() -> dict:
        """Upload the selected dataset archive to versioned filesystem storage and optional GOTRAIN_S3_BUCKET."""
        return record("publish_dataset", {}, artifacts.publish_dataset(task, os.environ.get("GOTRAIN_S3_BUCKET")))

    @tool
    def verify_model_download() -> dict:
        """Download a previously published best model and verify every artifact checksum."""
        chosen = best_run(task)
        return record("verify_model_download", {}, artifacts.restore_model(task, chosen["run_id"], os.environ.get("GOTRAIN_S3_BUCKET")))

    system = f"""You are GoTrain, an autonomous GEO-Bench ML experimenter.
Work ONLY on {task}, a {info['kind']} task. Inspect the data before training.
Your objective is to improve validation {'accuracy' if info['kind'] == 'classification' else 'mean_iou'}.
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
Finish with observed scores, decisions, limitations, and the selected run ID. Tools provide evidence;
the training history is authoritative. Do not claim the 75% historical baseline is beaten unless measured.
"""
    append_event(log, "session_started", task=task, experiments=experiments, minutes=minutes,
                 system_prompt=system, model=os.environ.get("GOTRAIN_BEDROCK_MODEL", "eu.anthropic.claude-sonnet-5-5"))
    model = BedrockModel(
        model_id=os.environ.get("GOTRAIN_BEDROCK_MODEL", "eu.anthropic.claude-sonnet-5-5"),
        region_name=os.environ.get("AWS_REGION", "eu-central-1"),
    )
    agent = Agent(model=model, system_prompt=system,
                  tools=[inspect_dataset, run_experiment, compare_experiments,
                         publish_best_model, publish_dataset, verify_model_download])
    try:
        result = agent("Inspect the dataset and execute the budgeted experiment workflow now.")
        transcript = agent.messages
        write_json(directory / "messages.json", transcript)
        write_json(directory / "session.json", {"task": task, "session_id": session_id,
                   "attempts": attempts, "completed_runs": session_runs, "tool_calls": tool_calls,
                   "elapsed_seconds": time.monotonic() - started, "finished_at": now(), "response": str(result)})
        append_event(log, "session_completed", response=str(result))
        return {"session_id": session_id, "path": str(directory), "response": str(result)}
    except Exception as error:
        write_json(directory / "messages.json", agent.messages)
        append_event(log, "session_failed", error=repr(error))
        raise

