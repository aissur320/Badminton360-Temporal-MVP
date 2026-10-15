"""Audited clean_v2 + interpolate_v1 behavior with explicit sequence isolation.

New-experiment change approved by the user: Clean uses the actual image width.
No label paths, label readers, detector weights, or GT are accepted here.
"""
import copy
import itertools
import math

from .common import validate_frames


def iou(a, b):
    area = max(0.0, min(a[2], b[2])-max(a[0], b[0])) * max(0.0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
    return area / union if union > 0 else 0.0


def center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def distance(a, b, width):
    # Keep the audited distance expression; only width source changes in this MVP.
    dx = abs(a[0]-b[0])
    if width is not None:
        dx = min(dx, width-dx)
    return math.hypot(dx, a[1]-b[1])


def cosine(a, b):
    return (a[0]*b[0]+a[1]*b[1]) / (math.hypot(*a)*math.hypot(*b)+1e-6)


def short_protect(index, det, history, width, cfg):
    if not history:
        return False
    last = history[-1]
    delta = index-last["index"]
    if delta > cfg["protect_max_gap"]:
        return False
    c = last["center"]
    velocity = ((c[0]-history[-2]["center"][0], c[1]-history[-2]["center"][1])
                if len(history) >= 2 else (0, 0))
    radius = cfg["protect_r0"]+cfg["protect_k"]*delta
    return any(distance(center(det["bbox"]), (c[0]+sign*velocity[0]*delta, c[1]+sign*velocity[1]*delta), width) <= radius
               for sign in (1, -1))


def clean_sequence(frames, cfg):
    width = frames[0]["width"]
    history = []
    filtered = []
    for i, frame in enumerate(frames):
        index = frame["frame_index"]
        kept, candidates = [], []
        if i < 2:
            kept = copy.deepcopy(frame["detections"])
            for det in kept:
                history.append([{"index": index, "center": center(det["bbox"])}])
        else:
            # Audited behavior: reference original inputs, not previously cleaned frames.
            prev, prev2 = frames[i-1]["detections"], frames[i-2]["detections"]
            for det in frame["detections"]:
                keep = any(iou(det["bbox"], p["bbox"]) >= cfg["iou_threshold"] for p in prev)
                if not keep and prev and prev2:
                    c0, c1, c2 = center(prev2[0]["bbox"]), center(prev[0]["bbox"]), center(det["bbox"])
                    v, current_v = (c1[0]-c0[0], c1[1]-c0[1]), (c2[0]-c1[0], c2[1]-c1[1])
                    predicted = (c1[0]+v[0], c1[1]+v[1])
                    scale, error = distance(c1, c0, width), distance(c2, predicted, width)
                    low, high = cfg["distance_ratio"]
                    keep = cosine(v, current_v) > cfg["cosine_threshold"] and low*scale <= error <= high*scale
                if keep:
                    kept.append(copy.deepcopy(det))
                    # Preserve historical new-singleton history behavior.
                    history.append([{"index": index, "center": center(det["bbox"])}])
                else:
                    candidates.append(det)
            for det in candidates:
                for trajectory in history:
                    if short_protect(index, det, trajectory, width, cfg):
                        kept.append(copy.deepcopy(det))
                        trajectory.append({"index": index, "center": center(det["bbox"])})
                        break
            history = [[point for point in trajectory if index-point["index"] <= cfg["protect_max_gap"]]
                       for trajectory in history]
            history = [trajectory for trajectory in history if trajectory]
        output_frame = copy.deepcopy(frame)
        output_frame["detections"] = kept
        filtered.append(output_frame)

    # Audited build_trajectories/save_trajectories: only first retained box is exported.
    trajectory_id, prev_center = -1, None
    result = []
    for frame in filtered:
        out = copy.deepcopy(frame)
        if not frame["detections"]:
            prev_center = None
            out["detections"] = []
        else:
            det = copy.deepcopy(frame["detections"][0])
            c = center(det["bbox"])
            if prev_center is None or distance(c, prev_center, width) > cfg["trajectory_distance"]:
                trajectory_id += 1
            det["traj_id"] = trajectory_id
            out["detections"] = [det]
            prev_center = c
        result.append(out)
    return result


def clean(frames, cfg):
    validate_frames(frames)
    if cfg["width_policy"] != "actual_image_width":
        raise ValueError("This new-experiment MVP requires the explicitly selected actual_image_width policy")
    result = []
    for _, group in itertools.groupby(frames, key=lambda f: f["sequence_id"]):
        result.extend(clean_sequence(list(group), cfg))
    return validate_frames(result)


def interpolate(frames, cfg):
    validate_frames(frames)
    from scipy.interpolate import CubicSpline
    result = []
    for _, grouped in itertools.groupby(frames, key=lambda f: f["sequence_id"]):
        seq = copy.deepcopy(list(grouped))
        anchors = [i for i, f in enumerate(seq) if f["detections"]]
        for start, end in zip(anchors, anchors[1:]):
            gap = end-start-1
            if gap <= 0 or gap > cfg["max_spline_gap"]:
                continue
            d0, d1 = seq[start]["detections"][0], seq[end]["detections"][0]
            b0, b1 = d0["bbox"], d1["bbox"]
            long_gap = gap > cfg["max_linear_gap"]
            if long_gap:
                c0, c1 = center(b0), center(b1)
                # Intentional historical two-knot spline, NOT a multi-knot curved trajectory.
                sx, sy = CubicSpline([start, end], [c0[0], c1[0]]), CubicSpline([start, end], [c0[1], c1[1]])
            for j in range(start+1, end):
                fraction = (j-start)/(end-start)
                box = [x+(y-x)*fraction for x, y in zip(b0, b1)]
                if long_gap:
                    width, height = box[2]-box[0], box[3]-box[1]
                    x, y = float(sx(j)), float(sy(j))
                    box = [x-width/2, y-height/2, x+width/2, y+height/2]
                seq[j]["detections"] = [{"bbox": box,
                    "conf": min(d0["conf"], d1["conf"]) * (cfg["spline_confidence_scale"] if long_gap else 1.0),
                    "source": "interpolated", "interpolation": "two_knot_spline" if long_gap else "linear"}]
        result.extend(seq)
    return validate_frames(result)
