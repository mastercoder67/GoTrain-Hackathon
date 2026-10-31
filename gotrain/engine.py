"""GPU training, validation-only selection, and a separately locked final evaluation."""
import importlib.metadata
import fcntl
from contextlib import contextmanager
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import Recipe, task_info
from .config import runs_root
from .data import GeoDataset, fingerprint, get_task, open_dev_split, training_statistics
from .metrics import confusion_matrix, scores
from .models import build_model, loss_value
from .store import (append_event, assert_unfrozen, best_run, claim_once, new_run,
                    now, read_json, run_dir, sha256, task_dir, write_json)


def provenance():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    versions = {}
    for package in ["torch", "torchvision", "geobench", "strands-agents", "numpy"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {"git_commit": commit, "working_tree_dirty": dirty, "versions": versions,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    # Record a seed, but do not claim bitwise GPU reproducibility.


def evaluate(model, loader, device, classes, kind):
    model.eval()
    cm = np.zeros((classes, classes), dtype=np.int64)
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            if kind == "segmentation" and logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, labels.shape[-2:], mode="bilinear", align_corners=False)
            predictions = logits.argmax(1).cpu().numpy()
            cm += confusion_matrix(labels.numpy(), predictions, classes)
    return scores(cm)


def loaders(task, recipe, stats):
    kind = task_info(task)["kind"]
    train = GeoDataset(open_dev_split(task, "train"), kind, recipe.image_size, stats, recipe.augmentation)
    valid = GeoDataset(open_dev_split(task, "valid"), kind, recipe.image_size, stats, native_mask=True)
    # A batch of one permits native resolution masks of varying shape at evaluation.
    valid_batch = 1 if kind == "segmentation" else recipe.batch_size
    kwargs = {"num_workers": recipe.workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train, batch_size=recipe.batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid, batch_size=valid_batch, **kwargs)
    return train_loader, valid_loader


@contextmanager
def compute_lock():
    root = runs_root()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "compute.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another GoTrain training/evaluation process is active") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def train(task, recipe, hypothesis="Manual baseline experiment"):
    with compute_lock():
        return _train(task, recipe, hypothesis)


def _train(task, recipe, hypothesis):
    info = task_info(task)
    recipe.validate(info["kind"])
    assert_unfrozen(task)
    run_id, directory = new_run(task)
    config = {"task": task, "run_id": run_id, "recipe": recipe.to_dict(), "hypothesis": hypothesis,
              "created_at": now(), "provenance": provenance(), "dataset": fingerprint(task),
              "selection_metric": "accuracy" if info["kind"] == "classification" else "mean_iou"}
    write_json(directory / "config.json", config)
    append_event(directory / "events.jsonl", "run_started", **config)
    started = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = []
    model = optimizer = None
    try:
        seed_all(recipe.seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        warm = None
        if recipe.warm_start:
            previous = run_dir(task, recipe.warm_start)
            previous_summary = read_json(previous / "summary.json")
            if previous_summary["status"] != "completed":
                raise ValueError("Warm start requires a completed validation run")
            checksum = sha256(previous / "best.pt")
            if checksum != previous_summary["checkpoint_sha256"]:
                raise RuntimeError("Warm-start checkpoint checksum mismatch")
            warm = torch.load(previous / "best.pt", map_location="cpu", weights_only=True)
            prior_recipe = warm["config"]["recipe"]
            if any(prior_recipe[k] != getattr(recipe, k) for k in ("model", "image_size")):
                raise ValueError("Warm start must retain model and image_size")
            if warm["config"]["dataset"] != config["dataset"]:
                raise ValueError("Warm-start dataset fingerprint mismatch")
            stats = warm["normalization"]
            config["warm_start_sha256"] = checksum
            write_json(directory / "config.json", config)
        else:
            stats = training_statistics(task, recipe.image_size, recipe.stats_samples, recipe.seed)
        write_json(directory / "normalization.json", stats)
        train_loader, valid_loader = loaders(task, recipe, stats)
        model = build_model(recipe, len(stats["mean"]), info["classes"], info["kind"], warm is None).to(device)
        if warm is not None:
            model.load_state_dict(warm["state_dict"])
            del warm
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe.epochs)
        amp = recipe.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        weights = None
        if recipe.balanced_loss:
            counts = torch.tensor(stats["class_counts"], dtype=torch.float32, device=device)
            weights = (counts.sum() / counts.clamp_min(1)).sqrt()
            weights = (weights / weights.mean()).clamp(0.1, 10)
        best, best_epoch, stale = -1.0, 0, 0
        reason = "epochs_completed"
        for epoch in range(1, recipe.epochs + 1):
            if time.monotonic() - started >= recipe.max_minutes * 60:
                reason = "time_budget"
                break
            model.train()
            # Pretrained BN statistics are unstable on tiny final batches; freeze BN running stats.
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            total_loss, total_samples = 0.0, 0
            complete = True
            for images, labels in train_loader:
                if time.monotonic() - started >= recipe.max_minutes * 60:
                    complete = False
                    reason = "time_budget"
                    break
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(images)
                    loss = loss_value(logits, labels, weights, recipe.dice)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach()) * len(labels)
                total_samples += len(labels)
            if total_samples == 0:
                break
            # Finish validation even if the soft training budget expired mid-epoch.
            metrics = evaluate(model, valid_loader, device, info["classes"], info["kind"])
            score = metrics[config["selection_metric"]]
            row = {"epoch": epoch, "complete_epoch": complete,
                   "train_loss": total_loss / total_samples, "validation": metrics,
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "elapsed_seconds": time.monotonic() - started}
            history.append(row)
            write_json(directory / "history.json", history)
            append_event(directory / "events.jsonl", "epoch_completed", **row)
            print(f"[{task}/{run_id}] epoch={epoch} loss={row['train_loss']:.4f} "
                  f"valid_{config['selection_metric']}={score:.4f}", flush=True, file=sys.stderr)
            if score > best:
                best, best_epoch, stale = score, epoch, 0
                temporary = directory / "best.pt.tmp"
                torch.save({"state_dict": model.state_dict(), "config": config, "normalization": stats,
                            "validation": metrics, "epoch": epoch}, temporary)
                os.replace(temporary, directory / "best.pt")
            else:
                stale += 1
            scheduler.step()
            if not complete or stale >= recipe.patience:
                reason = "time_budget" if not complete else "early_stopping"
                break
        if best_epoch == 0:
            raise RuntimeError("Budget ended before any model could be validated; increase max_minutes")
        summary = {"task": task, "run_id": run_id, "status": "completed", "hypothesis": hypothesis,
                   "selection_metric": config["selection_metric"], "selection_score": best,
                   "best_epoch": best_epoch, "validation": history[best_epoch - 1]["validation"],
                   "stop_reason": reason, "elapsed_seconds": time.monotonic() - started,
                   "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
                   "checkpoint_sha256": sha256(directory / "best.pt"), "recipe": recipe.to_dict()}
        write_json(directory / "summary.json", summary)
        append_event(directory / "events.jsonl", "run_completed", **summary)
        return summary
    except Exception as error:
        write_json(directory / "summary.json", {"task": task, "run_id": run_id, "status": "failed",
                                                "error": str(error), "recipe": recipe.to_dict()})
        append_event(directory / "events.jsonl", "run_failed", error=repr(error))
        raise
    finally:
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def finalize(task):
    with compute_lock():
        return _finalize(task)


def _finalize(task):
    """Freeze by VALIDATION and claim the test access before opening any test sample."""
    directory = task_dir(task)
    frozen = directory / "final-selection.json"
    if frozen.exists():
        selection = read_json(frozen)
    else:
        selection = best_run(task)
        claim_once(frozen, selection)
    run = run_dir(task, selection["run_id"])
    checkpoint = run / "best.pt"
    if sha256(checkpoint) != selection["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash differs from validation record")
    config = read_json(run / "config.json")
    if fingerprint(task) != config["dataset"]:
        raise RuntimeError("Dataset metadata changed since training")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    recipe = Recipe(**saved["config"]["recipe"])
    info = task_info(task)
    seed_all(recipe.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(recipe, len(saved["normalization"]["mean"]), info["classes"], info["kind"], False)
    model.load_state_dict(saved["state_dict"])
    model.to(device)
    # Exclusive claim survives crashes. Never automatically retry a consumed test split.
    claim_once(directory / "test-started.json", {"started_at": now(), "run_id": selection["run_id"],
                                                "checkpoint_sha256": selection["checkpoint_sha256"]})
    raw = get_task(task).get_dataset(split="test")
    dataset = GeoDataset(raw, info["kind"], recipe.image_size, saved["normalization"], native_mask=True)
    loader = DataLoader(dataset, batch_size=1 if info["kind"] == "segmentation" else recipe.batch_size)
    metrics = evaluate(model, loader, device, info["classes"], info["kind"])
    result = {"task": task, "run_id": selection["run_id"], "evaluated_at": now(), "split": "test",
              "samples": len(raw), "metrics": metrics, "checkpoint_sha256": selection["checkpoint_sha256"],
              "selection_metric": selection["selection_metric"], "selection_score": selection["selection_score"]}
    write_json(directory / "test-result.json", result)
    return result
