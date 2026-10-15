import argparse
import json

from .common import ROOT, resolve_path, write_json, within
from .data import audit_dataset, images_in, prepare_subset
from .pipeline import evaluate_stages, load_config, postprocess, predict, run_smoke, train


def main():
    parser = argparse.ArgumentParser(description="YOLOv8s-only temporal research MVP")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--data-root", required=True, help="Read-only dataset root; relative to MVP directory")
    audit.add_argument("--report", default="reports/dataset_audit.local.json")
    prepare = sub.add_parser("prepare-smoke")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--config", default=str(ROOT / "configs/smoke.json"))
    prepare.add_argument("--train", type=int, default=8)
    prepare.add_argument("--val", type=int, default=4)
    prepare.add_argument("--test", type=int, default=24)
    full = sub.add_parser("prepare-data", help="Copy/map full data locally; does not train")
    full.add_argument("--source", required=True)
    full.add_argument("--config", default=str(ROOT / "configs/experiment.example.json"))
    for name in ("train", "predict", "postprocess", "evaluate", "run-smoke"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=str(ROOT / "configs/smoke.json"))
        if name == "train":
            p.add_argument("--allow-full-train", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "audit":
            report = resolve_path(args.report)
            if not within(report, ROOT):
                raise ValueError("Audit report must be written inside the new MVP")
            result = audit_dataset(resolve_path(args.data_root))
            write_json(report, result)
        else:
            config = load_config(args.config)
            if args.command in ("prepare-smoke", "prepare-data"):
                source = resolve_path(args.source)
                full = args.command == "prepare-data"
                counts = ({k: len(images_in(source, k)) for k in ("train", "val", "test")} if full else
                          {k: getattr(args, k) for k in ("train", "val", "test")})
                if not full and any(counts[k] > n for k, n in (("train", 16), ("val", 8), ("test", 64))):
                    raise ValueError("prepare-smoke counts exceed the smoke limits")
                result = prepare_subset(source, resolve_path(config["paths"]["data_root"]), counts,
                                        config["sequence_pattern"], config.get("source_class_map"), all_sequences=full)
            elif args.command == "train":
                result = str(train(config, allow_full=args.allow_full_train))
            elif args.command == "predict":
                result = {"frames": len(predict(config))}
            elif args.command == "postprocess":
                postprocess(config)
                result = {"status": "completed"}
            elif args.command == "evaluate":
                result = evaluate_stages(config)
            else:
                result = run_smoke(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
