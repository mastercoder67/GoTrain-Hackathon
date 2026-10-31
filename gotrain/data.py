"""GEO-Bench adapters. Development APIs accept train/valid only."""
import json
import hashlib
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .config import TASKS, data_root, task_info
from .store import sha256, write_json


def prepare(task):
    from huggingface_hub import hf_hub_download
    info = task_info(task)
    parent = data_root() / info["benchmark"]
    dest = parent / task
    if (dest / "task_specs.pkl").exists():
        return {"task": task, "path": str(dest), "already_present": True}
    filename = f"{info['benchmark']}/{task}.zip"
    archive = Path(hf_hub_download(
        repo_id="recursix/geo-bench-1.0", repo_type="dataset",
        filename=filename, local_dir=str(data_root()),
    ))
    parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as z:
        for member in z.infolist():
            path = Path(member.filename)
            target = (parent / path).resolve()
            if not target.is_relative_to(parent.resolve()):
                raise ValueError("Archive path escapes dataset directory")
            if not path.parts or path.parts[0] != task:
                raise ValueError(f"Unexpected archive layout: {member.filename}")
        z.extractall(parent)
    if not (dest / "task_specs.pkl").exists():
        raise RuntimeError("Dataset archive did not contain task_specs.pkl")
    write_json(dest / "gotrain-source.json", {
        "repo": "recursix/geo-bench-1.0", "file": filename,
        "archive_sha256": sha256(archive),
    })
    return {"task": task, "path": str(dest), "already_present": False}


def get_task(task):
    import geobench
    info = task_info(task)
    parent = data_root() / info["benchmark"]
    if not (parent / task / "task_specs.pkl").exists():
        raise FileNotFoundError(f"Run: python -m gotrain prepare {task}")
    return next(t for t in geobench.task_iterator(benchmark_dir=str(parent)) if t.dataset_name == task)


def fingerprint(task):
    info = task_info(task)
    root = data_root() / info["benchmark"] / task
    paths = [root / "task_specs.pkl", *sorted(root.glob("*partition*.json"))]
    source = root / "gotrain-source.json"
    if source.exists():
        paths.append(source)
    return {p.name: sha256(p) for p in paths}


def open_dev_split(task, split):
    if split not in {"train", "valid"}:
        raise ValueError("Development tools can only access train and valid")
    return get_task(task).get_dataset(split=split)


def image_tensor(sample, size):
    # Use physical stored channel order, never the mislabeled spectral band names.
    channels = []
    for band in sample.bands:
        array = np.asarray(band.data, dtype=np.float32)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("Expected finite 2D band data")
        tensor = torch.from_numpy(array.copy())[None, None]
        channels.append(F.interpolate(tensor, (size, size), mode="bilinear", align_corners=False)[0])
    return torch.cat(channels, dim=0)


def label_tensor(sample, kind):
    if kind == "classification":
        return torch.tensor(int(sample.label), dtype=torch.long)
    array = np.asarray(sample.label.data)
    if array.ndim != 2:
        raise ValueError("Expected 2D segmentation mask")
    return torch.from_numpy(array.astype(np.int64, copy=True))


def training_statistics(task, size, count, seed):
    """Deterministic TRAIN-ONLY moments and target distribution; never benchmark-wide stats."""
    info = task_info(task)
    dataset = open_dev_split(task, "train")
    indices = np.random.default_rng(seed).choice(len(dataset), min(count, len(dataset)), replace=False)
    sums = squares = None
    pixels = 0
    class_counts = torch.zeros(info["classes"], dtype=torch.float64)
    for idx in indices:
        sample = dataset[int(idx)]
        x = image_tensor(sample, size).double()
        if sums is None:
            sums = torch.zeros(x.shape[0], dtype=torch.float64)
            squares = torch.zeros_like(sums)
        sums += x.sum((1, 2))
        squares += (x * x).sum((1, 2))
        pixels += x.shape[1] * x.shape[2]
        y = label_tensor(sample, info["kind"]).flatten()
        if bool(((y < 0) | (y >= info["classes"])).any()):
            raise ValueError("Unknown label values; configure an explicit ignore policy")
        class_counts += torch.bincount(y, minlength=info["classes"])
    mean = sums / pixels
    std = (squares / pixels - mean.square()).clamp_min(1e-8).sqrt()
    return {"mean": mean.float().tolist(), "std": std.float().tolist(),
            "class_counts": class_counts.tolist(), "sample_count": len(indices),
            "indices": [int(i) for i in indices], "split": "train"}


def cached_split(task, split, raw, size, kind, native_mask):
    """Cache resized inputs in half precision, preserving native evaluation masks.

    Only development splits are cacheable. Cache identity includes source metadata,
    preprocessing version and resolution. Normalization and random augmentation are
    applied after cache reads. Cached tensors are mmap loaded to bound resident RAM.
    """
    if split not in {"train", "valid"}:
        raise ValueError("Only development splits can be cached")
    identity = {"task": task, "split": split, "size": size, "kind": kind,
                "native_mask": native_mask, "source": fingerprint(task), "version": 1}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    root = data_root() / "_gotrain_cache"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{task}-{split}-{key}.pt"
    if not path.exists():
        images, labels = [], []
        for index in range(len(raw)):
            sample = raw[index]
            images.append(image_tensor(sample, size).half())
            y = label_tensor(sample, kind)
            if kind == "segmentation":
                if not native_mask:
                    y = F.interpolate(y[None, None].float(), (size, size), mode="nearest")[0, 0].long()
                if bool(((y < 0) | (y > 255)).any()):
                    raise ValueError("Cache supports masks with class IDs in [0,255]")
                y = y.to(torch.uint8)
            labels.append(y)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save({"images": torch.stack(images), "labels": labels, "identity": identity}, temporary)
        os.replace(temporary, path)
    result = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if result["identity"] != identity:
        raise RuntimeError("Dataset cache identity mismatch")
    return result


def validation_uses_half_inputs(task, config, channels):
    if "input_precision" in config:
        return config["input_precision"] == "float16"
    # Backward-compatible inference of preprocessing for runs made before the
    # explicit precision field existed. Legacy uncached recipes omit cache.
    recipe = config["recipe"]
    if not recipe.get("cache", False):
        return False
    count = len(open_dev_split(task, "train")) + len(open_dev_split(task, "valid"))
    return count * channels * recipe["image_size"]**2 * 2 <= 2 * 1024**3


class GeoDataset(Dataset):
    def __init__(self, raw, kind, size, stats, augment=False, native_mask=False, cache=None, quantize_inputs=False):
        self.raw, self.kind, self.size = raw, kind, size
        self.mean = torch.tensor(stats["mean"])[:, None, None]
        self.std = torch.tensor(stats["std"])[:, None, None]
        self.augment, self.native_mask = augment, native_mask
        self.cache = cache
        self.quantize_inputs = quantize_inputs

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        if self.cache is not None:
            x = self.cache["images"][index].float()
            y = self.cache["labels"][index].long()
        else:
            sample = self.raw[index]
            x = image_tensor(sample, self.size)
            if self.quantize_inputs:
                x = x.half().float()
            y = label_tensor(sample, self.kind)
            if self.kind == "segmentation" and not self.native_mask:
                y = F.interpolate(y[None, None].float(), (self.size, self.size), mode="nearest")[0, 0].long()
        if self.augment:
            for dim in [-1, -2]:
                if torch.rand(()) < 0.5:
                    x = x.flip(dim)
                    if self.kind == "segmentation":
                        y = y.flip(dim)
            k = int(torch.randint(0, 4, ()).item())
            x = x.rot90(k, (-2, -1))
            if self.kind == "segmentation":
                y = y.rot90(k, (-2, -1))
        return (x - self.mean) / self.std, y


def inspect(task):
    spec = get_task(task)
    train = open_dev_split(task, "train")
    valid = open_dev_split(task, "valid")
    sample = train[0]
    info = task_info(task)
    stats = training_statistics(task, 64, 32, 42)
    return {"task": task, **info, "train_samples": len(train), "valid_samples": len(valid),
            "bands": len(sample.bands), "image_shapes": [list(b.data.shape) for b in sample.bands],
            "class_names": getattr(spec.label_type, "class_names", None),
            "train_sample_class_counts": stats["class_counts"],
            "distribution_sample_count": stats["sample_count"],
            "band_policy": "physical channel order; train-only normalization",
            "known_warning": "EuroSAT band metadata is incorrect; names are not used" if task == "m-eurosat" else None,
            "fingerprint": fingerprint(task)}
