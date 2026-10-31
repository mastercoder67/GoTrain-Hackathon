import argparse
import json
import os
import sys
from pathlib import Path

from .config import Recipe, TASKS, task_info


def emit(result):
    print(json.dumps(result, indent=2, default=str))


def smoke():
    import torch
    from .models import build_model, loss_value
    torch.set_num_threads(2)
    result = {}
    for kind in ("classification", "segmentation"):
        recipe = Recipe(model="small", pretrained=False, dice=kind == "segmentation")
        model = build_model(recipe, 3, 2, kind)
        x = torch.randn(2, 3, 64, 64)
        y = torch.tensor([0, 1]) if kind == "classification" else torch.randint(0, 2, (2, 64, 64))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        loss = loss_value(model(x), y, dice=recipe.dice)
        loss.backward()
        optimizer.step()
        result[kind] = {"loss": float(loss.detach()), "output_shape": list(model(x).shape)}
    return {"synthetic_smoke_test": "passed", "results": result, "benchmark_test_split_accessed": False}


def main():
    parser = argparse.ArgumentParser(description="GoTrain: autonomous GEO-Bench experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("smoke")
    for command in ("prepare", "inspect", "train", "agent", "compare", "report", "finalize", "publish-data", "restore-data", "tta"):
        p = sub.add_parser(command)
        p.add_argument("task", choices=TASKS)
        if command == "train":
            p.add_argument("--recipe", help="JSON file containing Recipe fields")
            p.add_argument("--hypothesis", default="Pretrained baseline with train-only normalization")
        if command == "agent":
            p.add_argument("--experiments", type=int, default=3)
            p.add_argument("--minutes", type=float, default=60)
        if command == "finalize":
            p.add_argument("--confirm-test-once", action="store_true", required=True,
                           help="Freeze model selection and consume final test evaluation")
    for command in ("publish-model", "restore-model"):
        p = sub.add_parser(command)
        p.add_argument("task", choices=TASKS)
        p.add_argument("run_id")
    p = sub.add_parser("export")
    p.add_argument("--out", default="submission")
    args = parser.parse_args()
    if args.command == "doctor":
        import torch
        from .config import data_root, runs_root, artifacts_root
        from .engine import provenance
        emit({**provenance(), "python": sys.version, "cuda_available": torch.cuda.is_available(),
              "data_root": str(data_root()), "runs_root": str(runs_root()),
              "artifacts_root": str(artifacts_root()), "s3_configured": bool(os.environ.get("GOTRAIN_S3_BUCKET"))})
    elif args.command == "smoke":
        emit(smoke())
    elif args.command in {"prepare", "inspect"}:
        from . import data
        emit(getattr(data, args.command)(args.task))
    elif args.command == "train":
        from .engine import train
        values = json.loads(Path(args.recipe).read_text()) if args.recipe else {}
        if not args.recipe and task_info(args.task)["kind"] == "segmentation":
            values = {"image_size": 256, "batch_size": 8, "dice": True}
        emit(train(args.task, Recipe(**values), args.hypothesis))
    elif args.command == "agent":
        from .agent import run_agent
        emit(run_agent(args.task, args.experiments, args.minutes))
    elif args.command == "compare":
        from .store import list_runs
        emit(list_runs(args.task))
    elif args.command == "tta":
        from .variants import compare_tta
        emit(compare_tta(args.task))
    elif args.command == "report":
        from .report import create_report
        emit(create_report(args.task))
    elif args.command == "finalize":
        from .engine import finalize
        emit(finalize(args.task))
    elif args.command in {"publish-model", "restore-model", "publish-data", "restore-data"}:
        from . import artifacts
        funcs = {"publish-model": artifacts.publish_model, "restore-model": artifacts.restore_model,
                 "publish-data": artifacts.publish_dataset, "restore-data": artifacts.restore_dataset}
        kwargs = {"bucket": os.environ.get("GOTRAIN_S3_BUCKET")}
        if hasattr(args, "run_id"):
            kwargs["run_id"] = args.run_id
        emit(funcs[args.command](args.task, **kwargs))
    elif args.command == "export":
        from .submission import export
        emit(export(Path(args.out)))


if __name__ == "__main__":
    main()
