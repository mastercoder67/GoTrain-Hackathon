import os
from dataclasses import asdict, dataclass
from pathlib import Path


TASKS = {
    "m-eurosat": {"benchmark": "classification_v1.0", "kind": "classification", "classes": 10},
    "m-pv4ger": {"benchmark": "classification_v1.0", "kind": "classification", "classes": 2},
    "m-pv4ger-seg": {"benchmark": "segmentation_v1.0", "kind": "segmentation", "classes": 2},
    "m-nz-cattle": {"benchmark": "segmentation_v1.0", "kind": "segmentation", "classes": 2},
}


def data_root():
    return Path(os.environ.get("GOTRAIN_DATA", "~/gotrain-data")).expanduser()


def runs_root():
    return Path(os.environ.get("GOTRAIN_RUNS", "~/gotrain-runs/v1")).expanduser()


def artifacts_root():
    return Path(os.environ.get("GOTRAIN_ARTIFACTS", "~/gotrain-artifacts")).expanduser()


def task_info(task):
    if task not in TASKS:
        raise ValueError(f"Unsupported task {task!r}. Choose from {list(TASKS)}")
    return TASKS[task]


@dataclass
class Recipe:
    model: str = "resnet18"
    pretrained: bool = True
    epochs: int = 15
    batch_size: int = 32
    image_size: int = 128
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    augmentation: bool = True
    balanced_loss: bool = False
    dice: bool = False
    seed: int = 42
    workers: int = 0
    patience: int = 6
    stats_samples: int = 128
    amp: bool = True
    max_minutes: float = 30.0
    warm_start: str | None = None
    cache: bool = True

    def validate(self, kind):
        if self.model not in {"small", "resnet18", "resnet34"}:
            raise ValueError("model must be small, resnet18, or resnet34")
        for name, lo, hi in [("epochs", 1, 100), ("batch_size", 2, 128),
                             ("image_size", 64, 512), ("workers", 0, 8),
                             ("stats_samples", 1, 512), ("patience", 1, 100)]:
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
        if not 1e-6 <= self.learning_rate <= 0.1:
            raise ValueError("learning_rate outside [1e-6, 0.1]")
        if not 0 <= self.weight_decay <= 0.1:
            raise ValueError("weight_decay outside [0, 0.1]")
        if not 0 < self.max_minutes <= 240:
            raise ValueError("max_minutes must be in (0, 240]")
        if kind == "classification" and self.dice:
            raise ValueError("Dice loss is for segmentation only")
        return self

    def to_dict(self):
        return asdict(self)
