from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from gotrain import data, diagnose, diary, engine, report
from gotrain.artifacts import publish_model, restore_model
from gotrain.config import Recipe
from gotrain.metrics import confusion_matrix, scores
from gotrain.models import build_model, loss_value
from gotrain.store import best_run, claim_once, read_json, run_dir


class FakeSplit:
    def __init__(self, segmentation=False):
        self.segmentation = segmentation

    def __len__(self):
        return 4

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(index)
        image = np.random.default_rng(index).normal(index % 2, 1, (3, 32, 32)).astype(np.float32)
        label = index % 2
        if self.segmentation:
            mask = np.zeros((32, 32), dtype=np.int64)
            mask[8:24, 8:24] = 1
            label = SimpleNamespace(data=mask)
        return SimpleNamespace(bands=[SimpleNamespace(data=b) for b in image], label=label)


@pytest.fixture
def fake_benchmark(tmp_path, monkeypatch):
    monkeypatch.setenv("GOTRAIN_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("GOTRAIN_ARTIFACTS", str(tmp_path / "artifacts"))
    accesses = []

    def raw(task, split):
        accesses.append(split)
        return FakeSplit(task.endswith("seg"))

    def dev(task, split):
        if split not in {"train", "valid"}:
            raise ValueError("Development cannot read test")
        return raw(task, split)

    def spec(task):
        return SimpleNamespace(get_dataset=lambda split: raw(task, split),
                               label_type=SimpleNamespace(class_names=["background", "foreground"]))

    for module in (data, engine, report, diagnose):
        monkeypatch.setattr(module, "open_dev_split", dev)
        monkeypatch.setattr(module, "get_task", spec)
    monkeypatch.setattr(engine, "fingerprint", lambda task: {"fixture": "v1"})
    monkeypatch.setattr(data, "fingerprint", lambda task: {"fixture": "v1"})
    monkeypatch.setenv("GOTRAIN_DATA", str(tmp_path / "data"))
    return accesses


def test_metrics_global_confusion():
    cm = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], 2)
    result = scores(cm)
    assert result["accuracy"] == 0.75
    assert result["foreground_iou"] == pytest.approx(2 / 3)
    assert result["mean_iou"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert scores(np.zeros((2, 2), dtype=int))["mean_iou"] == 0
    with pytest.raises(ValueError):
        confusion_matrix([255], [0], 2)


def test_development_split_guard():
    with pytest.raises(ValueError, match="only access"):
        data.open_dev_split("m-eurosat", "test")


@pytest.mark.parametrize("kind", ["classification", "segmentation"])
def test_pretrained_architecture_without_download(kind):
    torch.set_num_threads(2)
    recipe = Recipe(model="resnet18", pretrained=False)
    model = build_model(recipe, 13, 2, kind).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 13, 64, 64))
    assert out.shape == ((2, 2) if kind == "classification" else (2, 2, 64, 64))


def test_normalization_train_only(fake_benchmark):
    stats = data.training_statistics("m-pv4ger", 64, 4, 42)
    assert stats["split"] == "train"
    assert len(stats["mean"]) == 3
    assert set(fake_benchmark) == {"train"}


def test_segmentation_augmentation_aligned():
    mask = np.zeros((64, 64), dtype=np.int64)
    mask[5:20, 12:30] = 1
    sample = SimpleNamespace(bands=[SimpleNamespace(data=mask.astype(np.float32))],
                             label=SimpleNamespace(data=mask))
    dataset = data.GeoDataset([sample], "segmentation", 64,
                              {"mean": [0], "std": [1]}, augment=True)
    for _ in range(8):
        x, y = dataset[0]
        assert torch.equal(x[0].long(), y)


@pytest.mark.parametrize("task", ["m-pv4ger", "m-pv4ger-seg"])
def test_train_publish_restore_report_and_test_once(fake_benchmark, task):
    recipe = Recipe(model="small", pretrained=False, epochs=1, batch_size=2,
                    image_size=64, stats_samples=4, amp=False, max_minutes=2,
                    dice=task.endswith("seg"))
    summary = engine.train(task, recipe, "Synthetic integration test")
    assert summary["status"] == "completed"
    assert "test" not in fake_benchmark
    assert best_run(task)["run_id"] == summary["run_id"]
    published = publish_model(task, summary["run_id"])
    assert Path(published["path"]).exists()
    assert restore_model(task, summary["run_id"])["verified"]
    generated = report.create_report(task, examples=2)
    assert "data:image/png" in Path(generated["report"]).read_text()
    assert "test" not in fake_benchmark
    result = engine.finalize(task)
    assert result["split"] == "test"
    assert fake_benchmark.count("test") == 1
    with pytest.raises(FileExistsError):
        engine.finalize(task)
    assert fake_benchmark.count("test") == 1
    with pytest.raises(RuntimeError, match="frozen"):
        engine.train(task, recipe)


def test_lock_claim_is_exclusive(tmp_path):
    path = tmp_path / "claim.json"
    claim_once(path, {"run": "first"})
    with pytest.raises(FileExistsError):
        claim_once(path, {"run": "second"})
    assert read_json(path)["run"] == "first"


def test_invalid_recipe():
    with pytest.raises(ValueError):
        Recipe(batch_size=0).validate("classification")
    with pytest.raises(ValueError):
        Recipe(dice=True).validate("classification")


def test_artifact_corruption_rejected(fake_benchmark):
    summary = engine.train("m-pv4ger", Recipe(model="small", pretrained=False,
                           epochs=1, image_size=64, batch_size=2, stats_samples=2))
    published = publish_model("m-pv4ger", summary["run_id"])
    (Path(published["path"]) / "best.pt").write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        restore_model("m-pv4ger", summary["run_id"])


def test_agent_tools_budget_and_logs(fake_benchmark, monkeypatch):
    import strands
    import strands.models
    from gotrain.agent import run_agent
    from gotrain.config import runs_root
    monkeypatch.setattr(data, "inspect", lambda task: {"task": task, "train_samples": 4})
    monkeypatch.delenv("GOTRAIN_S3_BUCKET", raising=False)

    class FakeAgent:
        def __init__(self, **kwargs):
            self.tools = {t.tool_name: t for t in kwargs["tools"]}
            self.messages = [{"role": "user", "content": [{"text": "Synthetic tool orchestration test"}]}]

        def __call__(self, prompt):
            import json
            assert "test" not in self.tools
            self.tools["inspect_dataset"]()
            assert self.tools["recall_lessons"]()["lessons_from_other_tasks"] == []
            recipe = json.dumps({"model": "small", "pretrained": False, "epochs": 1,
                                 "image_size": 64, "batch_size": 2, "stats_samples": 2})
            result = self.tools["run_experiment"](recipe, "Test tool execution")
            assert result["status"] == "completed"
            assert result["verdict"] == "kept" and result["previous_best_score"] is None
            assert self.tools["diagnose_errors"](result["run_id"])["split"] == "valid"
            assert self.tools["record_lesson"]("Uncited claims are refused", ["no_such_run"])["status"] == "rejected"
            assert self.tools["record_lesson"]("A tiny synthetic baseline trains end to end",
                                               [result["run_id"]])["type"] == "insight"
            assert self.tools["run_experiment"](recipe, "Must not train again")["status"] == "budget_exhausted"
            assert self.tools["compare_experiments"]()["attempts_used"] == 1
            self.tools["publish_best_model"]()
            assert self.tools["verify_model_download"]()["verified"]
            return "Synthetic orchestration completed"

    monkeypatch.setattr(strands, "Agent", FakeAgent)
    monkeypatch.setattr(strands.models, "BedrockModel", lambda **kwargs: object())
    result = run_agent("m-pv4ger", experiments=1, minutes=2)
    session = read_json(Path(result["path"]) / "session.json")
    assert session["attempts"] == 1
    assert session["tool_calls"] == 10
    page = Path(result["diary"]).read_text(encoding="utf-8")
    assert "kept as new best" in page and "Diagnosed run" in page and 'http-equiv="refresh"' not in page
    assert "Memory is empty" in page and "Lesson learned for future tasks" in page
    assert (Path(result["path"]) / "showcase" / "showcase.json").exists()
    assert "test" not in fake_benchmark


def test_mcp_adapter_import():
    from gotrain.mcp_server import mcp
    assert mcp is not None


def test_s3_dataset_roundtrip_with_fake_service(tmp_path, monkeypatch):
    from zipfile import ZipFile
    from gotrain import artifacts
    monkeypatch.setenv("GOTRAIN_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("GOTRAIN_ARTIFACTS", str(tmp_path / "artifacts"))
    root = tmp_path / "data" / "classification_v1.0"
    root.mkdir(parents=True)
    with ZipFile(root / "m-eurosat.zip", "w") as z:
        z.writestr("m-eurosat/task_specs.pkl", b"synthetic fixture, not unpickled")
    objects = {}

    class FakeS3:
        def upload_file(self, filename, bucket, key):
            objects[(bucket, key)] = Path(filename).read_bytes()

        def download_file(self, bucket, key, filename):
            Path(filename).write_bytes(objects[(bucket, key)])

    monkeypatch.setattr(artifacts, "client", FakeS3)
    published = artifacts.publish_dataset("m-eurosat", bucket="test-bucket")
    assert published["s3_uri"].startswith("s3://test-bucket/")
    restored = artifacts.restore_dataset("m-eurosat", bucket="test-bucket")
    assert restored["verified"]
    assert (Path(restored["path"]) / "task_specs.pkl").exists()


def test_warm_start_preserves_preprocessing_and_lineage(fake_benchmark):
    first = engine.train("m-pv4ger", Recipe(model="small", pretrained=False, epochs=1,
                         image_size=64, batch_size=2, stats_samples=4))
    second = engine.train("m-pv4ger", Recipe(model="small", pretrained=False, epochs=1,
                          image_size=64, batch_size=2, warm_start=first["run_id"], learning_rate=0.0001))
    parent = run_dir("m-pv4ger", first["run_id"])
    child = run_dir("m-pv4ger", second["run_id"])
    assert read_json(parent / "normalization.json") == read_json(child / "normalization.json")
    assert read_json(child / "config.json")["warm_start_sha256"] == first["checkpoint_sha256"]
    with pytest.raises(ValueError, match="retain model"):
        engine.train("m-pv4ger", Recipe(model="resnet18", pretrained=False, epochs=1,
                     image_size=64, warm_start=first["run_id"]))
    assert "test" not in fake_benchmark


def test_cache_alignment_and_native_validation_mask(fake_benchmark):
    stats = data.training_statistics("m-pv4ger-seg", 64, 4, 42)
    raw = data.open_dev_split("m-pv4ger-seg", "valid")
    cache = data.cached_split("m-pv4ger-seg", "valid", raw, 64, "segmentation", True)
    dataset = data.GeoDataset(raw, "segmentation", 64, stats, native_mask=True, cache=cache)
    plain = data.GeoDataset(raw, "segmentation", 64, stats, native_mask=True)
    x, y = dataset[0]
    original_x, original_y = plain[0]
    assert y.shape == (32, 32)
    assert torch.equal(y, original_y)
    assert torch.allclose(x, original_x, atol=0.003)
    equivalent = data.GeoDataset(raw, "segmentation", 64, stats, native_mask=True, quantize_inputs=True)
    assert torch.equal(x, equivalent[0][0])
    with pytest.raises(ValueError):
        data.cached_split("m-pv4ger-seg", "test", raw, 64, "segmentation", True)


def test_flip_averaging_restores_mask_orientation():
    class PixelModel(torch.nn.Module):
        def forward(self, x):
            return torch.cat([-x[:, :1], x[:, :1]], dim=1)
    model = PixelModel()
    x = torch.randn(2, 3, 16, 24)
    expected = model(x).softmax(1)
    assert torch.allclose(engine.predict(model, x, "segmentation", True), expected, atol=1e-6)


def test_inference_variant_uses_validation_only(fake_benchmark, monkeypatch):
    from gotrain import variants
    monkeypatch.setattr(variants, "fingerprint", lambda task: {"fixture": "v1"})
    monkeypatch.setattr(variants, "open_dev_split", data.open_dev_split)
    parent = engine.train("m-pv4ger", Recipe(model="small", pretrained=False, epochs=1,
                          image_size=64, batch_size=2, stats_samples=4))
    variant = variants.compare_tta("m-pv4ger")
    assert variant["recipe"]["tta"] is True
    assert variant["parent_run_id"] == parent["run_id"]
    assert "test" not in fake_benchmark


def test_object_components_measure_size_dependent_misses():
    truth = np.zeros((20, 20), dtype=np.int64)
    truth[1:3, 1:3] = 1          # small object, missed
    truth[10:16, 10:16] = 1      # large object, found
    truth[1:3, 17:19] = 1        # second small object, separate component
    prediction = np.zeros_like(truth)
    prediction[10:16, 10:16] = 1
    records = sorted(diagnose.object_records(truth, prediction))
    assert records == [(4, 0.0), (4, 0.0), (36, 1.0)]
    assert diagnose.components(np.zeros((4, 4), dtype=bool)).max() == 0


def test_focus_crops_zoom_keeps_image_and_mask_aligned():
    torch.manual_seed(0)
    mask = torch.zeros(64, 64, dtype=torch.long)
    mask[30:34, 30:34] = 1
    image = mask[None].float().repeat(3, 1, 1)
    for _ in range(10):
        x, y = data.GeoDataset.zoom_on_foreground(image, mask)
        assert y.shape == mask.shape and int(y.sum()) == 64  # 4x4 object shown at 2x scale
        assert ((x[0] > 0.5).long() == y).float().mean() > 0.97
    with pytest.raises(ValueError, match="segmentation only"):
        Recipe(focus_crops=True).validate("classification")


def test_diagnose_and_showcase_use_validation_only(fake_benchmark):
    recipe = Recipe(model="small", pretrained=False, epochs=1, batch_size=2, image_size=64,
                    stats_samples=4, amp=False, dice=True, focus_crops=True)
    first = engine.train("m-pv4ger-seg", recipe, "Baseline")
    result = diagnose.diagnose("m-pv4ger-seg", first["run_id"], examples=2)
    assert result["split"] == "valid" and result["observations"]
    assert result["objects"] == 4 and result["object_detection_by_size"]
    directory = run_dir("m-pv4ger-seg", first["run_id"])
    assert all((directory / e["image"]).exists() for e in result["worst_examples"])
    shown = diagnose.showcase("m-pv4ger-seg", first["run_id"], first["run_id"], directory.parent / "showcase")
    assert shown["examples"] and (directory.parent / "showcase" / shown["examples"][0]["image"]).exists()
    assert "test" not in fake_benchmark


def test_diary_reads_only_logged_evidence(tmp_path, monkeypatch):
    from gotrain.store import append_event
    monkeypatch.setenv("GOTRAIN_RUNS", str(tmp_path / "runs"))
    session = tmp_path / "runs" / "sessions" / "session_x"
    log = session / "events.jsonl"
    append_event(log, "session_started", task="m-pv4ger-seg", kind="segmentation", experiments=2, minutes=30)
    append_event(log, "experiment_requested", attempt=1, hypothesis="Baseline <b>", recipe_json='{"model": "resnet18"}')
    page = Path(diary.build(session)["diary"]).read_text(encoding="utf-8")
    assert 'http-equiv="refresh"' in page and "Training experiment 1" in page and "Baseline &lt;b&gt;" in page
    append_event(log, "tool_result", tool="run_experiment", inputs={}, result={
        "status": "completed", "selection_metric": "mean_iou", "selection_score": 0.71, "verdict": "rejected",
        "previous_best_score": 0.75, "best_epoch": 3, "elapsed_seconds": 60, "minutes_remaining": 12})
    append_event(log, "session_completed", response="done")
    page = Path(diary.build(session)["diary"]).read_text(encoding="utf-8")
    assert "0.7100 (previous best 0.7500): rejected" in page and "12 min of budget left" in page
    assert 'http-equiv="refresh"' not in page
    assert (tmp_path / "runs" / "live-diary.html").exists()


def test_lessons_transfer_measured_changes_between_tasks(fake_benchmark):
    from gotrain import lessons
    base = dict(model="small", pretrained=False, epochs=1, image_size=64, batch_size=2, stats_samples=2, amp=False)
    first = engine.train("m-pv4ger", Recipe(**base), "Baseline")
    second = engine.train("m-pv4ger", Recipe(**base, learning_rate=0.001, workers=1), "Higher learning rate")
    learned = lessons.learn_from_history("m-pv4ger")
    assert learned["lessons_added"] == 1
    assert lessons.learn_from_history("m-pv4ger")["lessons_added"] == 0  # idempotent
    lesson = lessons.load()[0]
    assert lesson["change"] == {"learning_rate": [0.0003, 0.001]}  # speed-only settings ignored
    assert lesson["compared_to"] == first["run_id"] and lesson["run_id"] == second["run_id"]
    assert lesson["delta"] == pytest.approx(second["selection_score"] - first["selection_score"])
    memory = lessons.recall("m-eurosat")
    assert memory["lessons_from_other_tasks"] and not memory["lessons_from_this_task"]
    assert memory["lever_scoreboard"][0]["change"] == "learning_rate 0.0003→0.001"
    recipe = memory["best_recipes_from_similar_tasks"][0]
    assert recipe["task"] == "m-pv4ger" and "warm_start" not in recipe["recipe"]
    assert lessons.recall("m-pv4ger-seg")["best_recipes_from_similar_tasks"] == []  # different kind
    with pytest.raises(ValueError, match="Evidence"):
        lessons.record_insight("m-pv4ger", "A claim citing another task's run", ["unknown"])
    lessons.record_insight("m-pv4ger", "Learning rate changes move this score", [first["run_id"], second["run_id"]])
    assert "Learning rate changes" in lessons.recall("m-eurosat")["lessons_from_other_tasks"][0]
    assert "test" not in fake_benchmark
