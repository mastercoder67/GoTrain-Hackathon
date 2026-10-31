"""Validation-only inference variants, recorded as new immutable model versions."""
import time
import torch
from torch.utils.data import DataLoader

from .config import Recipe, task_info
from .data import GeoDataset, cached_split, fingerprint, open_dev_split
from .engine import compute_lock, evaluate, provenance
from .models import build_model
from .store import (append_event, assert_unfrozen, best_run, new_run, now,
                    read_json, run_dir, sha256, write_json)


def compare_tta(task):
    with compute_lock():
        assert_unfrozen(task)
        parent = best_run(task)
        directory = run_dir(task, parent["run_id"])
        checkpoint = directory / "best.pt"
        if sha256(checkpoint) != parent["checkpoint_sha256"]:
            raise RuntimeError("Parent checkpoint checksum mismatch")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["config"]["dataset"] != fingerprint(task):
            raise RuntimeError("Dataset changed since training")
        recipe = Recipe(**saved["config"]["recipe"])
        if recipe.tta:
            return {"status": "already_enabled", "run_id": parent["run_id"]}
        recipe.tta = True
        info = task_info(task)
        run_id, output = new_run(task)
        started = time.monotonic()
        hypothesis = "Validation-only inference comparison: average probabilities over four flips"
        config = {"task": task, "run_id": run_id, "recipe": recipe.to_dict(), "hypothesis": hypothesis,
                  "experiment_type": "inference_variant", "parent_run_id": parent["run_id"],
                  "parent_checkpoint_sha256": parent["checkpoint_sha256"], "created_at": now(),
                  "dataset": fingerprint(task), "provenance": provenance(),
                  "selection_metric": parent["selection_metric"]}
        write_json(output / "config.json", config)
        write_json(output / "normalization.json", saved["normalization"])
        append_event(output / "events.jsonl", "inference_variant_started", **config)
        try:
            raw = open_dev_split(task, "valid")
            estimated = len(raw) * len(saved["normalization"]["mean"]) * recipe.image_size**2 * 2
            cache = cached_split(task, "valid", raw, recipe.image_size, info["kind"], True) if recipe.cache and estimated <= 2 * 1024**3 else None
            config["input_precision"] = "float16" if cache is not None else "float32"
            write_json(output / "config.json", config)
            dataset = GeoDataset(raw, info["kind"], recipe.image_size, saved["normalization"], native_mask=True, cache=cache)
            loader = DataLoader(dataset, batch_size=1 if info["kind"] == "segmentation" else recipe.batch_size,
                                num_workers=recipe.workers)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.set_num_threads(4)
            model = build_model(recipe, len(saved["normalization"]["mean"]), info["classes"], info["kind"], False).to(device)
            model.load_state_dict(saved["state_dict"])
            metrics = evaluate(model, loader, device, info["classes"], info["kind"], True)
            saved.update(config=config, validation=metrics)
            torch.save(saved, output / "best.pt")
            elapsed = time.monotonic() - started
            summary = {"task": task, "run_id": run_id, "status": "completed", "hypothesis": hypothesis,
                       "experiment_type": "inference_variant", "parent_run_id": parent["run_id"],
                       "selection_metric": parent["selection_metric"],
                       "selection_score": metrics[parent["selection_metric"]],
                       "best_epoch": saved["epoch"], "validation": metrics, "elapsed_seconds": elapsed,
                       "checkpoint_sha256": sha256(output / "best.pt"), "recipe": recipe.to_dict()}
            write_json(output / "history.json", [{"epoch": saved["epoch"], "train_loss": None,
                       "validation": metrics, "elapsed_seconds": elapsed, "inference_only": True}])
            write_json(output / "summary.json", summary)
            append_event(output / "events.jsonl", "inference_variant_completed", **summary)
            return summary
        except Exception as error:
            write_json(output / "summary.json", {"task": task, "run_id": run_id, "status": "failed", "error": str(error)})
            raise
