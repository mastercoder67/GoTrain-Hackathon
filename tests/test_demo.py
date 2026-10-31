import json

import pytest

from gotrain.demo import build_demo, image_uri


def test_demo_uses_saved_evidence_and_escapes_script_content(tmp_path):
    submission = tmp_path / "submission"
    task = submission / "m-nz-cattle"
    run = task / "recorded-run"
    run.mkdir(parents=True)
    result = {"task": "m-nz-cattle", "run_id": "recorded-run", "split": "test", "samples": 65,
              "metrics": {"mean_iou": 0.84}, "checkpoint_sha256": "original-hash"}
    (submission / "scores.json").write_text(json.dumps({"results": [result], "complete": False}))
    (submission / "verification.json").write_text(json.dumps({"checks": [
        {"command": ["python", "-m", "pytest"], "returncode": 0, "stdout": "22 passed"}]}))
    (tmp_path / "demo-verification.json").write_text(json.dumps({"checks": [
        {"command": ["python", "-m", "pytest"], "returncode": 0, "stdout": "24 passed"}]}))
    (task / "test-started.json").write_text('{"do_not_modify": true}')
    (task / "test-result.json").write_text(json.dumps(result))
    (run / "summary.json").write_text(json.dumps({"run_id": "recorded-run", "status": "completed",
        "selection_metric": "mean_iou", "selection_score": 0.85,
        "hypothesis": "</script><script>alert('untrusted evidence')</script>"}))
    before = {p: p.read_bytes() for p in submission.rglob("*") if p.is_file()}
    output = tmp_path / "demo.html"
    build_demo(submission, output)
    document = output.read_text()
    assert "__GOTRAIN_EVIDENCE__" not in document
    assert "</script><script>alert" not in document
    assert "\\u003c/script>" in document
    assert '"samples": 65' in document
    assert '"output": "24 passed"' in document
    assert '"output": "22 passed"' not in document
    assert "Recorded session replay" in document
    assert {p: p.read_bytes() for p in before} == before


def test_demo_rejects_example_paths_outside_export(tmp_path):
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"not for the exported showcase")
    showcase = tmp_path / "showcase"
    showcase.mkdir()
    with pytest.raises(ValueError, match="inside the showcase"):
        image_uri(showcase, "../outside.png")
