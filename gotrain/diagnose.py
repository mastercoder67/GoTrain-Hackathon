"""Validation-only error analysis: where a model fails, measured, never guessed.

Results are written next to the run (diagnosis.json, diagnosis/*.png) and give
the agent factual observations plus candidate recipe levers to test next.
"""
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .config import Recipe, task_info
from .data import GeoDataset, get_task, open_dev_split, validation_uses_half_inputs
from .engine import predict
from .models import build_model
from .report import rgb_view
from .store import best_run, list_runs, read_json, run_dir, sha256, write_json


def components(mask):
    """8-connected component labels of a 2D boolean mask, via max-pool propagation."""
    m = torch.from_numpy(np.asarray(mask, dtype=bool))
    if not bool(m.any()):
        return np.zeros(m.shape, dtype=np.int64)
    labels = (torch.arange(1, m.numel() + 1, dtype=torch.float64).view(m.shape) * m)[None, None]
    while True:
        grown = F.max_pool2d(labels, 3, 1, 1) * m
        if torch.equal(grown, labels):
            return labels[0, 0].long().numpy()
        labels = grown


def object_records(truth, prediction):
    """One record per true foreground object: area and fraction of it predicted foreground."""
    labels = components(truth == 1)
    ids, areas = np.unique(labels[labels > 0], return_counts=True)
    found = np.bincount(labels[(labels > 0) & (prediction == 1)], minlength=labels.max() + 1)
    return [(int(a), float(found[i] / a)) for i, a in zip(ids, areas)]


def load_run(task, run_id):
    directory = run_dir(task, run_id)
    summary = read_json(directory / "summary.json")
    if summary.get("status") != "completed":
        raise ValueError("Only completed runs can be diagnosed")
    if sha256(directory / "best.pt") != summary["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint checksum mismatch")
    saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    recipe = Recipe(**saved["config"]["recipe"])
    info = task_info(task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(recipe, len(saved["normalization"]["mean"]), info["classes"], info["kind"], False)
    model.load_state_dict(saved["state_dict"])
    model.to(device).eval()
    dataset = GeoDataset(open_dev_split(task, "valid"), info["kind"], recipe.image_size, saved["normalization"],
                         native_mask=True, quantize_inputs=validation_uses_half_inputs(
                             task, saved["config"], len(saved["normalization"]["mean"])))
    return {"summary": summary, "saved": saved, "recipe": recipe, "model": model, "dataset": dataset,
            "device": device, "directory": directory}


def infer(run, index, kind):
    x, target = run["dataset"][index]
    with torch.inference_mode():
        logits = predict(run["model"], x[None].to(run["device"]), kind, run["recipe"].tta).float()
        if kind == "segmentation":
            logits = F.interpolate(logits, target.shape[-2:], mode="bilinear", align_corners=False)
        probabilities = logits.softmax(1)[0].cpu()
    return x, target, probabilities


def mask_view(mask):
    return np.repeat((np.asarray(mask) * 255).astype(np.uint8)[..., None], 3, axis=2)


def panel_strip(images, height=192):
    """Join RGB arrays side by side at a common display height."""
    resized = [np.asarray(Image.fromarray(a).resize((max(1, round(a.shape[1] * height / a.shape[0])), height),
                                                     Image.NEAREST)) for a in images]
    gap = np.full((height, 6, 3), 255, dtype=np.uint8)
    parts = []
    for a in resized:
        parts += [a, gap]
    return np.concatenate(parts[:-1], axis=1)


def foreground_iou(truth, prediction):
    union = np.logical_or(truth == 1, prediction == 1).sum()
    return float(np.logical_and(truth == 1, prediction == 1).sum() / union) if union else None


def diagnose(task, run_id=None, examples=6):
    """Measure a completed run's validation failures and write diagnosis.json."""
    run_id = run_id or best_run(task)["run_id"]
    run = load_run(task, run_id)
    info = task_info(task)
    stats = run["saved"]["normalization"]
    images_dir = run["directory"] / "diagnosis"
    images_dir.mkdir(exist_ok=True)
    history = read_json(run["directory"] / "history.json")
    epochs_run, best_epoch = len(history), run["summary"]["best_epoch"]
    result = {"task": task, "run_id": run_id, "split": "valid",
              "selection_metric": run["summary"]["selection_metric"],
              "selection_score": run["summary"]["selection_score"],
              "best_epoch": best_epoch, "epochs_run": epochs_run}
    observations, levers = [], []
    trained = run["summary"].get("experiment_type") != "inference_variant"
    if trained and best_epoch >= epochs_run - 1 and run["summary"].get("stop_reason") != "early_stopping":
        observations.append(f"Best validation epoch was {best_epoch} of {epochs_run}; the score may still be rising.")
        levers.append("warm_start from this run with a lower learning_rate, or more epochs")

    if info["kind"] == "segmentation":
        objects, per_image = [], []
        truth_pixels = predicted_pixels = total_pixels = 0
        empty_images = empty_with_false_positive = 0
        for index in range(len(run["dataset"])):
            _, target, probabilities = infer(run, index, "segmentation")
            truth, prediction = target.numpy(), probabilities.argmax(0).numpy()
            total_pixels += truth.size
            truth_pixels += int((truth == 1).sum())
            predicted_pixels += int((prediction == 1).sum())
            if not (truth == 1).any():
                empty_images += 1
                empty_with_false_positive += int((prediction == 1).any())
            objects += object_records(truth, prediction)
            iou = foreground_iou(truth, prediction)
            if iou is not None:
                per_image.append((iou, index))
        areas = np.array([a for a, _ in objects]) if objects else np.zeros(0)
        detected = np.array([f >= 0.5 for _, f in objects]) if objects else np.zeros(0, dtype=bool)
        buckets = []
        if len(areas) >= 3:
            low, high = (int(v) for v in np.quantile(areas, [1 / 3, 2 / 3]))
            for name, selector in (("small", areas <= low), ("medium", (areas > low) & (areas <= high)),
                                   ("large", areas > high)):
                if selector.any():
                    buckets.append({"size": name, "objects": int(selector.sum()),
                                    "max_area_px": int(areas[selector].max()),
                                    "detected_fraction": float(detected[selector].mean())})
        area_ratio = predicted_pixels / truth_pixels if truth_pixels else None
        result.update(foreground_pixel_fraction=truth_pixels / max(total_pixels, 1),
                      predicted_to_true_area_ratio=area_ratio, objects=len(objects),
                      object_detection_by_size=buckets, empty_images=empty_images,
                      empty_images_with_false_positives=empty_with_false_positive)
        observations.append(f"Foreground covers {result['foreground_pixel_fraction']:.1%} of validation pixels "
                            f"across {len(objects)} annotated objects.")
        if buckets:
            text = ", ".join(f"{b['size']} (≤{b['max_area_px']} px) {b['detected_fraction']:.0%}" for b in buckets)
            observations.append(f"Objects detected (≥50% of their pixels found): {text}.")
            if buckets[-1]["detected_fraction"] - buckets[0]["detected_fraction"] >= 0.15:
                levers.append("focus_crops=true to zoom training on foreground objects, or a larger image_size")
        if area_ratio is not None:
            observations.append(f"Predicted foreground area is {area_ratio:.2f}× the true area.")
            if area_ratio < 0.8:
                levers.append("balanced_loss=true and/or dice=true to give foreground more weight")
            elif area_ratio > 1.25:
                levers.append("balanced_loss=false or more epochs to reduce false-positive foreground")
        if empty_images:
            observations.append(f"{empty_with_false_positive} of {empty_images} validation images with no target "
                                f"contain false-positive foreground.")
        worst = sorted(per_image)[:examples]
        result["worst_examples"] = []
        for rank, (iou, index) in enumerate(worst):
            x, target, probabilities = infer(run, index, "segmentation")
            strip = panel_strip([rgb_view(x, stats, task), mask_view(target.numpy()),
                                 mask_view(probabilities.argmax(0).numpy())])
            name = f"worst-{rank}.png"
            Image.fromarray(strip).save(images_dir / name)
            result["worst_examples"].append({"index": index, "foreground_iou": iou, "image": f"diagnosis/{name}",
                                             "panels": ["input", "truth", "prediction"]})
    else:
        names = getattr(get_task(task).label_type, "class_names", None) or [str(i) for i in range(info["classes"])]
        matrix = np.array(run["summary"]["validation"]["confusion_matrix"])
        off = matrix.copy()
        np.fill_diagonal(off, 0)
        pairs = []
        for flat in np.argsort(off, axis=None)[::-1][:3]:
            truth, guess = np.unravel_index(flat, off.shape)
            if off[truth, guess] == 0:
                break
            pairs.append({"truth": names[truth], "predicted": names[guess], "count": int(off[truth, guess]),
                          "of": int(matrix[truth].sum())})
        recall = np.array(run["summary"]["validation"]["per_class_recall"])
        weakest = int(recall.argmin())
        support = matrix.sum(1)
        result.update(top_confusions=pairs, weakest_class={"name": names[weakest], "recall": float(recall[weakest])})
        for p in pairs:
            observations.append(f"{p['truth']} predicted as {p['predicted']}: {p['count']} of {p['of']} validation images.")
        observations.append(f"Weakest class recall: {names[weakest]} at {recall[weakest]:.0%}.")
        if support.min() > 0 and support.max() / support.min() >= 2:
            observations.append(f"Validation class sizes range from {support.min()} to {support.max()} images.")
            levers.append("balanced_loss=true for class imbalance")
        mistakes = []
        for index in range(len(run["dataset"])):
            _, target, probabilities = infer(run, index, "classification")
            guess = int(probabilities.argmax())
            if guess != int(target):
                mistakes.append((-float(probabilities[guess]), index, int(target), guess))
        result["worst_examples"] = []
        for rank, (confidence, index, truth, guess) in enumerate(sorted(mistakes)[:examples]):
            x, _ = run["dataset"][index]
            name = f"worst-{rank}.png"
            Image.fromarray(panel_strip([rgb_view(x, stats, task)])).save(images_dir / name)
            result["worst_examples"].append({"index": index, "truth": names[truth], "predicted": names[guess],
                                             "confidence": -confidence, "image": f"diagnosis/{name}"})
        if mistakes:
            levers.append("augmentation, a larger image_size or resnet34 for confusions between similar classes")
    result["observations"], result["candidate_levers"] = observations, levers
    write_json(run["directory"] / "diagnosis.json", result)
    return result


def showcase(task, before_id, after_id, destination, examples=3):
    """Same validation images before and after: the baseline's worst cases, not random picks."""
    info = task_info(task)
    before_path = run_dir(task, before_id) / "diagnosis.json"
    picks = read_json(before_path) if before_path.exists() else diagnose(task, before_id)
    indices = [e["index"] for e in picks.get("worst_examples", [])][:examples]
    before, after = load_run(task, before_id), load_run(task, after_id)
    stats = before["saved"]["normalization"]
    rows = []
    for index in indices:
        x, target, p_before = infer(before, index, info["kind"])
        _, _, p_after = infer(after, index, info["kind"])
        if info["kind"] == "segmentation":
            truth = target.numpy()
            first, second = p_before.argmax(0).numpy(), p_after.argmax(0).numpy()
            strip = panel_strip([rgb_view(x, stats, task), mask_view(first), mask_view(second), mask_view(truth)])
            rows.append({"index": index, "before_iou": foreground_iou(truth, first),
                         "after_iou": foreground_iou(truth, second), "strip": strip})
        else:
            names = getattr(get_task(task).label_type, "class_names", None) or [str(i) for i in range(info["classes"])]
            rows.append({"index": index, "truth": names[int(target)], "before": names[int(p_before.argmax())],
                         "after": names[int(p_after.argmax())], "strip": panel_strip([rgb_view(x, stats, task)])})
    destination.mkdir(parents=True, exist_ok=True)
    for n, row in enumerate(rows):
        name = f"showcase-{n}.png"
        Image.fromarray(row.pop("strip")).save(destination / name)
        row["image"] = name
    result = {"task": task, "before_run": before_id, "after_run": after_id, "split": "valid",
              "before_score": before["summary"]["selection_score"], "after_score": after["summary"]["selection_score"],
              "selection_metric": before["summary"]["selection_metric"], "kind": info["kind"],
              "selection_note": "Validation images where the first model did worst; not a random sample.",
              "examples": rows}
    write_json(destination / "showcase.json", result)
    return result


def first_and_best(task, run_ids=None):
    rows = [r for r in list_runs(task) if r.get("status") == "completed"
            and (run_ids is None or r["run_id"] in run_ids)]
    if not rows:
        raise ValueError("No completed validation experiments yet")
    return rows[0]["run_id"], max(rows, key=lambda r: r["selection_score"])["run_id"]
