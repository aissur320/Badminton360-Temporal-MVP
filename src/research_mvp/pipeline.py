import copy
import datetime
import importlib.metadata
import os
from pathlib import Path

from .common import ROOT, frame_key, portable_config, read_json, relative_path, require_new_directory, resolve_path, safe_child, sha256, validate_frames, within, write_json
from .data import audit_dataset, build_manifest
from .postprocess import clean, interpolate


def load_config(path):
    config = read_json(path)
    if config["experiment_kind"] != "new_experiment_not_paper_reproduction":
        raise ValueError("This MVP implements the user-selected NEW experiment protocol")
    if config["evaluation"]["protocol"] != "confidence_greedy_best_unmatched_ap101_v1":
        raise ValueError("Unsupported evaluation protocol")
    if config["clean"]["width_policy"] != "actual_image_width":
        raise ValueError("Choose the approved actual_image_width policy")
    name = config["run_name"]
    if not name or name in (".", "..") or any(ch in name for ch in "/\\:"):
        raise ValueError("run_name must be one directory name")
    config = copy.deepcopy(config)
    for key in ("data_root", "pretrained_weights", "output_root"):
        config["paths"][key] = str(resolve_path(config["paths"][key]))
    data = Path(config["paths"]["data_root"])
    output = Path(config["paths"]["output_root"]) / name
    # Never write training caches into the historical project dataset.
    if not within(data, ROOT / "data"):
        raise ValueError("Copy/prepare the dataset under this MVP's ignored data/ before training")
    if not within(output, ROOT) or within(output, data) or within(data, output):
        raise ValueError("Output must be inside the new MVP and separate from data")
    if Path(config["paths"]["pretrained_weights"]).name != "yolov8s.pt":
        raise ValueError("Only yolov8s.pt pretrained initialization is supported")
    if config["train"]["epochs"] < 1:
        raise ValueError("epochs must be positive")
    interp = config["interpolate"]
    if not 0 < interp["max_linear_gap"] <= interp["max_spline_gap"]:
        raise ValueError("Invalid interpolation gap limits")
    return config


def run_dir(config):
    return Path(config["paths"]["output_root"]) / config["run_name"]


def verify_run_config(config):
    out = run_dir(config)
    if portable_config(config) != portable_config(read_json(out / "config.resolved.json")):
        raise ValueError("Config changed since training; use the same resolved configuration")
    return out


def dataset_config(data_root, yaml_path):
    """Ultralytics uses the YAML's directory when path is omitted (pinned 8.3.156)."""
    return {"train": relative_path(Path(data_root) / "images/train", base=Path(yaml_path).parent),
            "val": relative_path(Path(data_root) / "images/val", base=Path(yaml_path).parent),
            "names": {0: "Shuttlecock"}}


def portable_training_args(path):
    """Normalize the library's on-disk metadata without changing its runtime arguments."""
    import yaml
    path = Path(path)
    if not path.is_file():
        return
    args = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("model", "data", "project", "save_dir", "resume", "pretrained", "source", "cfg"):
        value = args.get(key)
        if isinstance(value, str) and Path(value).is_absolute():
            args[key] = relative_path(value)
    path.write_text(yaml.safe_dump(args, sort_keys=False, allow_unicode=True), encoding="utf-8")


def configure_runtime():
    # Redirect library settings, font and weight caches away from the historical archive.
    for key, value in {"YOLO_CONFIG_DIR": ROOT / ".cache" / "ultralytics",
                       "MPLCONFIGDIR": ROOT / ".cache" / "matplotlib",
                       "TORCH_HOME": ROOT / ".cache" / "torch"}.items():
        Path(value).mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["WANDB_DISABLED"] = "true"


def yolo_class():
    configure_runtime()
    if importlib.metadata.version("ultralytics") != "8.3.156":
        raise ValueError("Install pinned Ultralytics 8.3.156 before running detector stages")
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    return YOLO


def assert_yolov8s(model):
    info = model.model.yaml
    # Official YOLOv8 checkpoints can embed the older expanded YAML, without scale/yaml_file.
    legacy = info.get("depth_multiple") == 0.33 and info.get("width_multiple") == 0.5
    modern = info.get("scale") == "s" and "yolov8" in str(info.get("yaml_file", ""))
    has_c2f = any(module.__class__.__name__ == "C2f" for module in model.model.modules())
    if getattr(model, "task", None) != "detect" or not (legacy or modern) or not has_c2f:
        raise ValueError("Checkpoint architecture must be YOLOv8s")


def train(config, *, allow_full=False):
    import yaml
    params = config["train"]
    if params["epochs"] != 1 and not allow_full:
        raise ValueError("More than one epoch requires --allow-full-train; no formal training is automatic")
    data_root = Path(config["paths"]["data_root"])
    weights = Path(config["paths"]["pretrained_weights"])
    if not weights.is_file():
        raise ValueError("Missing local yolov8s.pt; see README download command")
    audit = audit_dataset(data_root, require_single_class=True)
    manifest = build_manifest(data_root, config["sequence_pattern"])
    out = run_dir(config)
    require_new_directory(out)
    write_json(out / "config.resolved.json", portable_config(config))
    write_json(out / "dataset_audit.json", audit)
    write_json(out / "manifest.json", manifest)
    dataset_yaml = out / "dataset.yaml"
    dataset_yaml.write_text(yaml.safe_dump(dataset_config(data_root, dataset_yaml), sort_keys=False), encoding="utf-8")
    model = yolo_class()(str(weights))
    assert_yolov8s(model)
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        model.train(data=str(dataset_yaml), project=str(out), name="train", exist_ok=False,
                    epochs=params["epochs"], imgsz=params["imgsz"], batch=params["batch"],
                    device=params["device"], workers=params["workers"], seed=params["seed"],
                    rect=params["rect"], deterministic=True, amp=False, plots=False,
                    save=True, cache=False, pretrained=True, resume=False)
    finally:
        portable_training_args(out / "train" / "args.yaml")
    best = out / "train" / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("Training did not produce best.pt; inference cannot proceed")
    packages = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "ultralytics", "numpy", "scipy", "Pillow", "PyYAML")}
    write_json(out / "training.json", {"status": "completed", "started_utc": started,
        "architecture": "YOLOv8s", "initialization_sha256": sha256(weights),
        "best_weights": "train/weights/best.pt", "best_sha256": sha256(best),
        "epochs_requested": params["epochs"], "packages": packages,
        "experiment_kind": config["experiment_kind"]})
    return best


def predict(config):
    out = verify_run_config(config)
    if (out / "raw.json").exists():
        raise ValueError("raw.json already exists; select a new run name rather than overwriting predictions")
    training = read_json(out / "training.json")
    best = out / "train" / "weights" / "best.pt"
    if sha256(best) != training["best_sha256"]:
        raise ValueError("best.pt changed since this run's training")
    model = yolo_class()(str(best))
    assert_yolov8s(model)
    manifest = read_json(out / "manifest.json")
    frames = copy.deepcopy(manifest["frames"])
    params = config["predict"]
    for frame in frames:
        # One image per call in explicit manifest order; never pass a directory/glob.
        path = safe_child(config["paths"]["data_root"], frame["image"])
        result = model.predict(source=str(path), imgsz=params["imgsz"], conf=params["conf"],
                               iou=params["iou"], device=params["device"], batch=1,
                               save=False, save_txt=False, verbose=False, max_det=300)[0]
        if tuple(result.orig_shape) != (frame["height"], frame["width"]):
            raise ValueError("Image dimensions changed since manifest creation")
        frame["detections"] = []
        for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            if cls != 0:
                raise ValueError("Only the trained shuttlecock class is supported")
            frame["detections"].append({"bbox": box, "conf": score, "source": "detector"})
    validate_frames(frames)
    write_json(out / "raw.json", {"schema_version": 1, "stage": "raw", "experiment_kind": config["experiment_kind"],
                                  "best_sha256": training["best_sha256"], "frames": frames})
    return frames


def postprocess(config):
    out = verify_run_config(config)
    if any((out / name).exists() for name in ("clean.json", "interpolated.json")):
        raise ValueError("Postprocessing outputs already exist; refusing to mix runs")
    raw = read_json(out / "raw.json")
    cleaned = clean(raw["frames"], config["clean"])
    filled = interpolate(cleaned, config["interpolate"])
    for stage, frames, name in (("clean", cleaned, "clean.json"), ("interpolated", filled, "interpolated.json")):
        write_json(out / name, {"schema_version": 1, "stage": stage,
            "experiment_kind": config["experiment_kind"], "best_sha256": raw["best_sha256"], "frames": frames})


def evaluate_stages(config):
    from .evaluate import evaluate, load_ground_truth
    out = verify_run_config(config)
    if (out / "metrics.json").exists():
        raise ValueError("metrics.json already exists; refusing to overwrite evaluation")
    stages = {stage: read_json(out / filename)["frames"] for stage, filename in
              (("raw", "raw.json"), ("clean", "clean.json"), ("interpolated", "interpolated.json"))}
    reference = [{k: v for k, v in f.items() if k != "detections"} for f in stages["raw"]]
    for frames in stages.values():
        if [{k: v for k, v in f.items() if k != "detections"} for f in frames] != reference:
            raise ValueError("Stage manifests differ; cannot make a fair comparison")
    gt = load_ground_truth(Path(config["paths"]["data_root"]), stages["raw"])
    metrics = {stage: evaluate(frames, gt) for stage, frames in stages.items()}
    write_json(out / "metrics.json", {"experiment_kind": config["experiment_kind"], "stages": metrics})
    return metrics


def run_smoke(config):
    audit = audit_dataset(Path(config["paths"]["data_root"]), require_single_class=True)
    if config["train"]["epochs"] != 1 or any(audit[k]["images"] > n for k, n in (("train", 16), ("val", 8), ("test", 64))):
        raise ValueError("Smoke is limited to 1 epoch and <=16 train / 8 val / 64 test images")
    train(config)
    predict(config)
    postprocess(config)
    return evaluate_stages(config)
