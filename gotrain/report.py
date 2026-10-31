"""Self-contained validation report with real prediction examples; no web server needed."""
import base64
import html
import io
import json

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .config import Recipe, task_info
from .data import GeoDataset, get_task, open_dev_split, validation_uses_half_inputs
from .models import build_model
from .engine import predict
from .store import best_run, list_runs, read_json, run_dir, sha256, task_dir


def png(array):
    out = io.BytesIO()
    Image.fromarray(array).save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def rgb_view(x, stats, task):
    raw = x.numpy() * np.array(stats["std"])[:, None, None] + np.array(stats["mean"])[:, None, None]
    indices = [3, 2, 1] if task == "m-eurosat" else [0, 1, 2]
    if raw.shape[0] < 3:
        indices = [0, 0, 0]
    rgb = raw[indices].transpose(1, 2, 0)
    lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1))
    return (np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def create_report(task, examples=6):
    selected = best_run(task)
    directory = run_dir(task, selected["run_id"])
    if sha256(directory / "best.pt") != selected["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint checksum mismatch")
    saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    recipe = Recipe(**saved["config"]["recipe"])
    info = task_info(task)
    model = build_model(recipe, len(saved["normalization"]["mean"]), info["classes"], info["kind"], False)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    raw = open_dev_split(task, "valid")
    dataset = GeoDataset(raw, info["kind"], recipe.image_size, saved["normalization"], native_mask=True,
                         quantize_inputs=validation_uses_half_inputs(task, saved["config"], len(saved["normalization"]["mean"])))
    names = getattr(get_task(task).label_type, "class_names", None)
    if names is None:
        names = [str(i) for i in range(info["classes"])]
    cards = []
    # Fixed evenly spaced validation examples: no cherry-picked claim of typical performance.
    indices = np.linspace(0, len(dataset) - 1, min(examples, len(dataset)), dtype=int)
    with torch.inference_mode():
        for index in indices:
            x, target = dataset[int(index)]
            logits = predict(model, x[None], info["kind"], recipe.tta)
            rgb = rgb_view(x, saved["normalization"], task)
            if info["kind"] == "classification":
                pred = int(logits.argmax(1).item())
                truth = int(target)
                text = f"Truth: {names[truth]} · Prediction: {names[pred]}"
                cards.append(f'<article><img src="{png(rgb)}"><p>Validation #{index}<br>{html.escape(text)}</p></article>')
            else:
                logits = F.interpolate(logits, target.shape[-2:], mode="bilinear", align_corners=False)
                pred = logits.argmax(1)[0].numpy()
                # Masks are visualized at their native annotation resolution.
                panels = [f'<figure><img src="{png(rgb)}"><figcaption>Input (display stretch)</figcaption></figure>',
                          f'<figure><img src="{png((target.numpy() * 255).astype(np.uint8))}"><figcaption>Ground truth</figcaption></figure>',
                          f'<figure><img src="{png((pred * 255).astype(np.uint8))}"><figcaption>Prediction</figcaption></figure>']
                cards.append(f'<article><p>Validation #{index}</p><div class="panels">' + "".join(panels) + '</div></article>')
    rows, curves = [], []
    colors = ["#65d6ce", "#f5ba65", "#be99ff", "#86c779"]
    for n, run in enumerate(list_runs(task)):
        if run["status"] != "completed":
            continue
        history = read_json(run_dir(task, run["run_id"]) / "history.json")
        points = " ".join(f"{40 + i * 660 / max(len(history)-1, 1):.1f},{230 - r['validation'][run['selection_metric']] * 200:.1f}" for i, r in enumerate(history))
        color = colors[n % len(colors)]
        curves.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
        rows.append(f'<tr><td style="color:{color}">{html.escape(run["run_id"])}</td>'
                    f'<td>{run["selection_score"]:.4f}</td><td>{run["best_epoch"]}</td>'
                    f'<td>{run["elapsed_seconds"]/60:.1f} min</td><td>{html.escape(run["hypothesis"])}</td></tr>')
    test_path = task_dir(task) / "test-result.json"
    test_section = "Final test evaluation has not been run."
    if test_path.exists():
        test_section = html.escape(json.dumps(read_json(test_path), indent=2))
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GoTrain experiment report</title><style>
body{margin:0;background:#101722;color:#e9eff9;font:16px system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:36px}
h1{font-size:38px;margin-bottom:8px}h2{margin-top:40px}p{line-height:1.6;color:#bac8dc}code{color:#65d6ce}
.tag{color:#65d6ce;text-transform:uppercase;letter-spacing:2px}table{width:100%;border-collapse:collapse;font-size:14px}
td,th{padding:12px;text-align:left;border-bottom:1px solid #354359;overflow-wrap:anywhere}article{background:#1b2738;border-radius:14px;padding:20px;margin:14px 0}
img{max-width:100%;width:192px;image-rendering:auto;border-radius:8px}.panels{display:flex;gap:20px;flex-wrap:wrap}figure{margin:0}figcaption{font-size:13px;color:#bac8dc}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#1b2738;padding:20px;border-radius:14px}svg{width:100%;height:auto}
</style><main><span class="tag">GoTrain · Experiment evidence</span>'''
    document += f'<h1>{html.escape(task)}</h1><p>Best validation {selected["selection_metric"]}: <strong>{selected["selection_score"]:.4f}</strong><br>Selected run: <code>{selected["run_id"]}</code></p>'
    document += '<h2>Validation experiments</h2><svg viewBox="0 0 740 270" role="img" aria-label="Validation score by epoch for each experiment"><path d="M40 20V230H710" stroke="#7b8da8" fill="none"/><text x="8" y="35" fill="#bac8dc">1</text><text x="8" y="235" fill="#bac8dc">0</text><text x="300" y="265" fill="#bac8dc">Epoch (within each run)</text>' + "".join(curves) + '</svg>'
    document += '<table><tr><th>Run</th><th>Best score</th><th>Epoch</th><th>Runtime</th><th>Hypothesis</th></tr>' + "".join(rows) + '</table>'
    document += '<h2>Validation predictions</h2><p>Fixed evenly spaced validation examples. White denotes foreground. Input contrast is stretched for display only.</p>' + "".join(cards)
    diagnosis_path = directory / "diagnosis.json"
    if diagnosis_path.exists():
        observations = read_json(diagnosis_path)["observations"]
        document += '<h2>Error diagnosis (validation)</h2><ul>' + "".join(
            f'<li>{html.escape(o)}</li>' for o in observations) + '</ul>'
    document += '<h2>Measured validation metrics</h2><pre>' + html.escape(json.dumps(selected["validation"], indent=2)) + '</pre>'
    document += '<h2>Final test status</h2><pre>' + test_section + '</pre></main></html>'
    path = task_dir(task) / "report.html"
    path.write_text(document)
    return {"report": str(path)}
