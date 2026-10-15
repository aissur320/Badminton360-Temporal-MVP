"""One evaluator for every saved stage. No detector/postprocessor imports this module."""
from .common import frame_key, validate_frames, safe_child
from .data import read_yolo_labels
from .postprocess import iou


def match_at_threshold(frames, ground_truth, threshold):
    """Confidence-first greedy matching to the BEST UNMATCHED GT, stable tie order."""
    predictions = [(d["conf"], frame_key(f), i, d["bbox"]) for f in frames for i, d in enumerate(f["detections"])]
    predictions.sort(key=lambda p: (-p[0], p[1], p[2]))
    matched = {key: set() for key in ground_truth}
    tp, fp = [], []
    for _, key, _, box in predictions:
        candidates = [(iou(box, gt), j) for j, gt in enumerate(ground_truth[key]) if j not in matched[key]]
        # Smaller GT index breaks equal-IoU ties deterministically.
        best = max(candidates, key=lambda c: (c[0], -c[1])) if candidates else (0.0, -1)
        success = best[1] >= 0 and best[0] >= threshold
        if success:
            matched[key].add(best[1])
        tp.append(int(success))
        fp.append(int(not success))
    return tp, fp


def ap101(tp, fp, n_gt):
    if n_gt == 0:
        return None
    cumulative_tp = cumulative_fp = 0
    curve = []
    for hit, false in zip(tp, fp):
        cumulative_tp += hit
        cumulative_fp += false
        curve.append((cumulative_tp/n_gt, cumulative_tp/(cumulative_tp+cumulative_fp)))
    return sum(max((p for r, p in curve if r >= sample/100), default=0.0) for sample in range(101))/101


def evaluate(frames, ground_truth):
    validate_frames(frames)
    if set(ground_truth) != {frame_key(f) for f in frames}:
        raise ValueError("GT frame keys must exactly match evaluated image manifest; missing labels are not background")
    n_gt = sum(len(boxes) for boxes in ground_truth.values())
    aps = []
    counts = None
    for k in range(10):
        tp, fp = match_at_threshold(frames, ground_truth, (50+5*k)/100)
        aps.append(ap101(tp, fp, n_gt))
        if k == 0:
            ntp, nfp = sum(tp), sum(fp)
            counts = {"TP": ntp, "FP": nfp, "FN": n_gt-ntp,
                      "precision_50": ntp/(ntp+nfp) if ntp+nfp else 0.0,
                      "recall_50": ntp/n_gt if n_gt else None}
    return {"protocol": "confidence_greedy_best_unmatched_ap101_v1", "class_id": 0,
            "images": len(frames), "ground_truth_boxes": n_gt,
            "prediction_boxes": sum(len(f["detections"]) for f in frames), **counts,
            "AP50": aps[0], "mAP50_95": sum(aps)/len(aps) if n_gt else None,
            "AP_by_IoU": {f"{(50+5*k)/100:.2f}": value for k, value in enumerate(aps)},
            "note": "101-point precision-envelope AP; not full COCOeval; no extra score filtering or maxDet cap"}


def load_ground_truth(data_root, frames):
    ground_truth = {}
    for frame in frames:
        image_path = safe_child(data_root, frame["image"])
        relative = image_path.relative_to(data_root)
        if relative.parts[0] != "images" or len(relative.parts) != 3:
            raise ValueError("Expected images/<split>/<filename> in manifest")
        path = safe_child(data_root, str(relative).replace("images", "labels", 1))
        path = path.with_suffix(".txt")
        ground_truth[frame_key(frame)] = read_yolo_labels(path, frame["width"], frame["height"])
    return ground_truth
