"""Versioned filesystem artifacts, with optional S3 upload/download using the instance role."""
import shutil
from pathlib import Path
from zipfile import ZipFile

from .config import artifacts_root, data_root, task_info
from .store import read_json, run_dir, sha256, write_json

MODEL_FILES = ("best.pt", "config.json", "normalization.json", "summary.json", "history.json", "events.jsonl")


def client():
    import boto3
    return boto3.client("s3")


def model_key(task, run_id):
    run_dir(task, run_id)  # validates task and identifier
    return f"gotrain/models/{task}/{run_id}"


def publish_model(task, run_id, bucket=None):
    source = run_dir(task, run_id)
    summary = read_json(source / "summary.json")
    if summary["status"] != "completed":
        raise ValueError("Only completed runs can be published")
    if sha256(source / "best.pt") != summary["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint was modified after validation")
    manifest = {"task": task, "run_id": run_id, "files": {name: sha256(source / name) for name in MODEL_FILES}}
    prefix = model_key(task, run_id)
    dest = artifacts_root() / "models" / task / run_id
    dest.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        shutil.copy2(source / name, dest / name)
    write_json(dest / "manifest.json", manifest)
    if bucket:
        s3 = client()
        for name in (*MODEL_FILES, "manifest.json"):
            s3.upload_file(str(dest / name), bucket, f"{prefix}/{name}")
    return {"path": str(dest), "s3_uri": f"s3://{bucket}/{prefix}/" if bucket else None, "manifest": manifest}


def restore_model(task, run_id, bucket=None):
    # Restore to a separate directory: never overwrite a run or fabricate validation history.
    prefix = model_key(task, run_id)
    dest = artifacts_root() / "restored-models" / task / run_id
    dest.mkdir(parents=True, exist_ok=True)
    source = artifacts_root() / "models" / task / run_id
    names = (*MODEL_FILES, "manifest.json")
    if bucket:
        s3 = client()
        for name in names:
            s3.download_file(bucket, f"{prefix}/{name}", str(dest / name))
    else:
        for name in names:
            shutil.copy2(source / name, dest / name)
    manifest = read_json(dest / "manifest.json")
    if manifest.get("task") != task or manifest.get("run_id") != run_id:
        raise RuntimeError("Artifact identity mismatch")
    if set(manifest["files"]) != set(MODEL_FILES):
        raise RuntimeError("Unexpected manifest files")
    for name in MODEL_FILES:
        if sha256(dest / name) != manifest["files"][name]:
            raise RuntimeError(f"Checksum mismatch: {name}")
    return {"path": str(dest), "verified": True}


def publish_dataset(task, bucket=None):
    info = task_info(task)
    source = data_root() / info["benchmark"] / f"{task}.zip"
    if not source.exists():
        raise FileNotFoundError("Dataset archive missing; run prepare first")
    dest = artifacts_root() / "datasets" / task
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest / "dataset.zip")
    manifest = {"task": task, "sha256": sha256(source), "benchmark": info["benchmark"]}
    write_json(dest / "manifest.json", manifest)
    if bucket:
        s3 = client()
        for name in ("dataset.zip", "manifest.json"):
            s3.upload_file(str(dest / name), bucket, f"gotrain/datasets/{task}/{name}")
    return {"path": str(dest), "s3_uri": f"s3://{bucket}/gotrain/datasets/{task}/" if bucket else None,
            "manifest": manifest}


def restore_dataset(task, bucket=None):
    info = task_info(task)
    dest = artifacts_root() / "restored-datasets" / task
    dest.mkdir(parents=True, exist_ok=True)
    if bucket:
        s3 = client()
        for name in ("dataset.zip", "manifest.json"):
            s3.download_file(bucket, f"gotrain/datasets/{task}/{name}", str(dest / name))
    else:
        source = artifacts_root() / "datasets" / task
        for name in ("dataset.zip", "manifest.json"):
            shutil.copy2(source / name, dest / name)
    manifest = read_json(dest / "manifest.json")
    if manifest.get("task") != task or manifest.get("benchmark") != info["benchmark"]:
        raise RuntimeError("Dataset identity mismatch")
    if sha256(dest / "dataset.zip") != manifest["sha256"]:
        raise RuntimeError("Dataset archive checksum mismatch")
    # Extract in the restored directory, leaving the active training dataset unchanged.
    with ZipFile(dest / "dataset.zip") as z:
        for member in z.infolist():
            path = Path(member.filename)
            if not path.parts or path.parts[0] != task or not (dest / path).resolve().is_relative_to(dest.resolve()):
                raise ValueError("Unsafe dataset archive layout")
        z.extractall(dest)
    return {"path": str(dest / task), "verified": True}
