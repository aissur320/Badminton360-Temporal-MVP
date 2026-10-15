import hashlib
import json
import math
import os
import copy
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_path(value):
    p = Path(value).expanduser()
    return (ROOT / p).resolve() if not p.is_absolute() else p.resolve()


def relative_path(value, *, base=None, root=None):
    """Persist portable POSIX separators; resolve relative inputs against the MVP root."""
    root = Path(root if root is not None else ROOT).resolve()
    base = Path(base if base is not None else root).resolve()
    value = os.fspath(value)
    path = Path(value)
    if PureWindowsPath(value).drive and not path.is_absolute():
        raise ValueError("Foreign or drive-relative Windows path; use a path relative to this MVP")
    if not path.is_absolute():
        path = root / path
    try:
        return Path(os.path.relpath(path.resolve(), base)).as_posix()
    except ValueError as error:
        raise ValueError("Paths on different drives cannot be portable; place the resource with this MVP") from error


def portable_config(config, *, root=None):
    """Runtime configs may be absolute, but saved configs and comparisons are root-relative."""
    result = copy.deepcopy(config)
    for key in ("data_root", "pretrained_weights", "output_root"):
        result["paths"][key] = relative_path(result["paths"][key], root=root)
    return result


def within(path, parent):
    return Path(path).resolve().is_relative_to(Path(parent).resolve())


def require_new_directory(path):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Refusing to overwrite existing directory: {path}")
    path.mkdir(parents=True)


def safe_child(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes data root: {relative}")
    return path


def frame_key(frame):
    return frame["sequence_id"], frame["frame_index"]


def validate_frames(frames, *, predictions=True):
    """Require an explicit, ordered, dense sequence manifest; never infer missing frames."""
    if not isinstance(frames, list) or not frames:
        raise ValueError("A nonempty frame list is required (empty detections are allowed)")
    previous = None
    image_keys = set()
    sequence_shapes = {}
    for frame in frames:
        allowed = {"sequence_id", "frame_index", "image", "width", "height", "detections"}
        if set(frame) - allowed:
            raise ValueError("Unexpected frame fields; GT/annotations must not enter prediction stages")
        sid, index = frame_key(frame)
        if not isinstance(sid, str) or not sid or type(index) is not int or index < 0:
            raise ValueError("sequence_id must be nonempty and frame_index a nonnegative integer")
        key = (sid, index)
        if previous is not None:
            if key <= previous:
                raise ValueError("Frames must have unique keys and be sorted by sequence_id, numeric frame_index")
            if sid == previous[0] and index != previous[1] + 1:
                raise ValueError("Missing frame record: explicitly include the image and empty detections")
        previous = key
        if frame["image"] in image_keys:
            raise ValueError("Duplicate image in manifest")
        image_keys.add(frame["image"])
        if any(type(frame[k]) is not int or frame[k] <= 0 for k in ("width", "height")):
            raise ValueError("Positive integer image dimensions required")
        shape = (frame["width"], frame["height"])
        if sid in sequence_shapes and sequence_shapes[sid] != shape:
            raise ValueError("Image dimensions change inside a sequence")
        sequence_shapes[sid] = shape
        if predictions:
            if not isinstance(frame.get("detections"), list):
                raise ValueError("Missing detections list")
            for d in frame["detections"]:
                if set(d) - {"bbox", "conf", "traj_id", "source", "interpolation"}:
                    raise ValueError("Unexpected detection fields")
                box = d["bbox"]
                if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
                    raise ValueError("bbox must be four finite pixel xyxy coordinates")
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError("Degenerate or reversed bbox")
                if not isinstance(d["conf"], (int, float)) or not math.isfinite(d["conf"]) or not 0 <= d["conf"] <= 1:
                    raise ValueError("Confidence must be finite and in [0,1]")
    return frames
