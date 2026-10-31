import json
from pathlib import Path

import geobench
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = Path.home() / "gotrain-data" / "classification_v1.0"
task = next(
    t for t in geobench.task_iterator(benchmark_dir=str(root))
    if t.dataset_name == "m-eurosat"
)

def load_split(split):
    dataset = task.get_dataset(split=split)
    images, labels = [], []
    for sample in dataset:
        images.append(np.stack([b.data for b in sample.bands]))
        labels.append(int(sample.label))
    return (
        torch.from_numpy(np.stack(images).astype(np.float32)),
        torch.tensor(labels, dtype=torch.long),
    )

print("Loading training and validation data...", flush=True)
x_train, y_train = load_split("train")
x_val, y_val = load_split("valid")

# Compute statistics using training data only.
mean = x_train.mean(dim=(0, 2, 3), keepdim=True)
std = x_train.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std

train_loader = DataLoader(
    TensorDataset(x_train, y_train), batch_size=64, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(x_val, y_val), batch_size=128
)

model = nn.Sequential(
    nn.Conv2d(13, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(128, 10),
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
output = Path.home() / "gotrain-runs" / "eurosat-baseline"
output.mkdir(parents=True, exist_ok=True)
history, best = [], -1.0

print(f"Training on {device}", flush=True)
for epoch in range(1, 6):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            predictions = model(images.to(device)).argmax(dim=1)
            correct += (predictions.cpu() == labels).sum().item()

    accuracy = correct / len(y_val)
    row = {
        "epoch": epoch,
        "train_loss": total_loss / len(y_train),
        "val_accuracy": accuracy,
    }
    history.append(row)
    print(
        f"Epoch {epoch}/5 | loss={row['train_loss']:.4f} "
        f"| validation accuracy={accuracy:.2%}",
        flush=True,
    )
    if accuracy > best:
        best = accuracy
        torch.save({
            "model": model.state_dict(),
            "mean": mean,
            "std": std,
            "epoch": epoch,
            "val_accuracy": accuracy,
        }, output / "best.pt")

    (output / "history.json").write_text(json.dumps(history, indent=2))

print(f"Best validation accuracy: {best:.2%}")
print(f"Model and training history saved to: {output}")
