import collections
import math
import re
import shutil
from pathlib import Path

from .common import safe_child, validate_frames, write_json, require_new_directory, within

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def images_in(root, split):
    folder = Path(root) / "images" / split
    if not folder.is_dir():
        raise ValueError(f"Missing images/{split}")
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No images in {folder}")
    if len({p.stem for p in images}) != len(images):
        raise ValueError(f"Duplicate image stems in {split}")
    return images


def read_yolo_rows(path):
    """Audit label syntax without guessing or merging class IDs."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing label file: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Expected class cx cy w h: {path.name}:{number}")
        cls, cx, cy, w, h = map(float, parts)
        if not all(math.isfinite(v) for v in (cls, cx, cy, w, h)) or cls < 0 or cls != int(cls) or not all(0 <= v <= 1 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"Invalid YOLO label: {path.name}:{number}")
        rows.append((int(cls), cx, cy, w, h))
    return rows


def read_yolo_labels(path, width, height):
    """Only audit/evaluation read GT; missing labels are errors, never backgrounds."""
    rows = read_yolo_rows(path)
    if any(row[0] != 0 for row in rows):
        raise ValueError(f"Expected explicitly mapped single-class labels: {Path(path).name}")
    return [[(cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height]
            for _, cx, cy, w, h in rows]


def audit_dataset(root, *, require_single_class=False):
    from PIL import Image
    root = Path(root)
    report = {}
    for split in ("train", "val", "test"):
        images = images_in(root, split)
        labels = {p.stem for p in (root / "labels" / split).glob("*.txt")}
        if labels != {p.stem for p in images}:
            raise ValueError(f"Image/label stems do not match in {split}")
        dims = collections.Counter()
        box_count = 0
        empty_count = 0
        classes = collections.Counter()
        for p in images:
            with Image.open(p) as img:
                size = img.size
            dims[str(size)] += 1
            rows = read_yolo_rows(root / "labels" / split / (p.stem + ".txt"))
            classes.update(row[0] for row in rows)
            box_count += len(rows)
            empty_count += not rows
        if require_single_class and set(classes) - {0}:
            raise ValueError(f"Nonzero class IDs in {split}: explicit mapping required before training")
        report[split] = {"images": len(images), "labels": len(labels), "boxes": box_count,
                         "background_images": empty_count, "class_counts": dict(classes), "dimensions": dict(dims)}
    return report


def build_manifest(root, sequence_pattern, split="test"):
    """Read image names/headers only; labels are deliberately never accessed."""
    from PIL import Image
    root = Path(root)
    pattern = re.compile(sequence_pattern)
    frames = []
    for p in images_in(root, split):
        match = pattern.fullmatch(p.name)
        if not match:
            raise ValueError(f"Image name does not match explicit sequence pattern: {p.name}")
        with Image.open(p) as img:
            width, height = img.size
        frames.append({"sequence_id": match["sequence_id"], "frame_index": int(match["frame_index"]),
                       "image": p.relative_to(root).as_posix(), "width": width, "height": height})
    frames.sort(key=lambda f: (f["sequence_id"], f["frame_index"]))
    validate_frames(frames, predictions=False)
    return {"schema_version": 1, "frame_base": 0, "order": ["sequence_id", "frame_index"],
            "sequence_pattern": sequence_pattern, "frames": frames}


def prepare_subset(source, destination, counts, pattern, class_map=None, *, all_sequences=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if within(destination, source) or within(source, destination):
        raise ValueError("Subset source and destination must not overlap")
    # Parse names before opening images so a small smoke does not read all test images.
    regex = re.compile(pattern)
    keyed = []
    for p in images_in(source, "test"):
        match = regex.fullmatch(p.name)
        if match is None:
            raise ValueError(f"No explicit sequence match for {p.name}")
        keyed.append((match["sequence_id"], int(match["frame_index"]), p))
    keyed.sort()
    first_id = keyed[0][0]
    test = [p for sid, _, p in keyed if all_sequences or sid == first_id]
    selected = {"train": images_in(source, "train")[:counts["train"]],
                "val": images_in(source, "val")[:counts["val"]], "test": test[:counts["test"]]}
    if any(len(selected[k]) != counts[k] or counts[k] <= 0 for k in selected):
        raise ValueError("Insufficient images for requested subset")
    class_map = {int(k): int(v) for k, v in (class_map or {"0": 0}).items()}
    if any(v != 0 for v in class_map.values()):
        raise ValueError("This single-class MVP only supports explicit mappings to class 0")
    for split, paths in selected.items():
        for p in paths:
            rows = read_yolo_rows(source / "labels" / split / (p.stem+".txt"))
            if any(row[0] not in class_map for row in rows):
                raise ValueError("Unmapped source class; ask for dataset semantics before copying/training")
    require_new_directory(destination)
    copied = []
    for split, paths in selected.items():
        for p in paths:
            for src, rel in ((p, Path("images") / split / p.name),
                             (source / "labels" / split / (p.stem+".txt"), Path("labels") / split / (p.stem+".txt"))):
                if not src.is_file():
                    raise ValueError(f"Missing source file: {src}")
                dest = destination / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if rel.parts[0] == "labels":
                    # Preserve numeric coordinate strings; only apply the explicit approved class mapping.
                    lines = []
                    for line in src.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            values = line.split()
                            values[0] = str(class_map[int(float(values[0]))])
                            lines.append(" ".join(values))
                    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                else:
                    shutil.copyfile(src, dest)
            copied.append({"split": split, "image": p.name})
    write_json(destination / "subset.json", {"selection": "all sequences" if all_sequences else "lexical train/val prefix; chronological first-sequence test prefix",
                                             "class_map": class_map, "files": copied})
    write_json(destination / "manifest.json", build_manifest(destination, pattern))
    return audit_dataset(destination, require_single_class=True)
