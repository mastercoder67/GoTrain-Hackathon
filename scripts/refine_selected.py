"""A reproducible fixed fine-tuning pass after autonomous recipe search.

This is a deterministic refinement policy, not a claim that Claude discovered
these settings. Models are still selected only using validation results.
"""
import argparse
import os

from gotrain.artifacts import publish_model, restore_model
from gotrain.config import Recipe, TASKS
from gotrain.engine import train
from gotrain.report import create_report
from gotrain.store import best_run


def refine(task, epochs, minutes):
    before = best_run(task)
    settings = dict(before["recipe"])
    settings.update(warm_start=before["run_id"], epochs=epochs, patience=5,
                    learning_rate=min(settings["learning_rate"] / 3, 0.0001),
                    workers=2, cache=True, max_minutes=minutes)
    result = train(task, Recipe(**settings),
                   "Fixed refinement policy: fine-tune validation-selected weights with a lower learning rate")
    after = best_run(task)
    print(f"{task}: validation {before['selection_score']:.4f} -> {after['selection_score']:.4f}", flush=True)
    publish_model(task, after["run_id"], os.environ.get("GOTRAIN_S3_BUCKET"))
    restore_model(task, after["run_id"], os.environ.get("GOTRAIN_S3_BUCKET"))
    create_report(task)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minutes-per-task", type=float, default=15)
    args = parser.parse_args()
    for task in ([args.task] if args.task else TASKS):
        refine(task, args.epochs, args.minutes_per_task)
