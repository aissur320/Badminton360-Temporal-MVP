import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_mvp.common import frame_key, safe_child, validate_frames
from research_mvp.data import build_manifest, prepare_subset, read_yolo_labels
from research_mvp.evaluate import evaluate, load_ground_truth, match_at_threshold
from research_mvp.pipeline import assert_yolov8s, load_config
from research_mvp.postprocess import clean, distance, interpolate, short_protect

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/smoke.json").read_text())


def box(x=10, score=0.9):
    return {"bbox": [x, 10, x+10, 20], "conf": score}


def frame(i, detections=None, sequence="video1"):
    return {"sequence_id": sequence, "frame_index": i,
            "image": f"images/test/{sequence}_{i:04d}.png", "width": 3840, "height": 1920,
            "detections": [] if detections is None else detections}


class MatchingTests(unittest.TestCase):
    def test_duplicate_detection_is_false_positive(self):
        frames = [frame(0, [box(score=0.9), box(score=0.8)])]
        metrics = evaluate(frames, {("video1", 0): [[10, 10, 20, 20]]})
        self.assertEqual((metrics["TP"], metrics["FP"], metrics["FN"]), (1, 1, 0))
        self.assertEqual(metrics["precision_50"], 0.5)
        self.assertAlmostEqual(metrics["AP50"], 1.0)

    def test_best_unmatched_gt_can_be_second_best_overall(self):
        frames = [frame(0, [box(score=0.9), box(score=0.8)])]
        gt = {("video1", 0): [[10, 10, 20, 20], [11, 10, 21, 20]]}
        tp, fp = match_at_threshold(frames, gt, 0.5)
        self.assertEqual((tp, fp), ([1, 1], [0, 0]))

    def test_empty_predictions_count_all_gt_as_fn(self):
        metrics = evaluate([frame(0)], {("video1", 0): [[10, 10, 20, 20]]})
        self.assertEqual((metrics["TP"], metrics["FP"], metrics["FN"]), (0, 0, 1))
        self.assertEqual((metrics["AP50"], metrics["mAP50_95"]), (0.0, 0.0))

    def test_background_false_positive(self):
        metrics = evaluate([frame(0, [box()])], {("video1", 0): []})
        self.assertEqual((metrics["TP"], metrics["FP"], metrics["FN"]), (0, 1, 0))
        self.assertIsNone(metrics["AP50"])
        self.assertIsNone(metrics["recall_50"])

    def test_empty_background_is_defined(self):
        metrics = evaluate([frame(0)], {("video1", 0): []})
        self.assertEqual(metrics["FP"], 0)
        self.assertIsNone(metrics["AP50"])

    def test_no_matching_across_sequence_ids(self):
        frames = [frame(0, [box()], "a"), frame(0, [], "b")]
        gt = {("a", 0): [], ("b", 0): [[10, 10, 20, 20]]}
        metrics = evaluate(frames, gt)
        self.assertEqual((metrics["TP"], metrics["FP"], metrics["FN"]), (0, 1, 1))

    def test_missing_gt_key_is_not_silent_background(self):
        with self.assertRaises(ValueError):
            evaluate([frame(0)], {})

    def test_ap101_known_half_recall(self):
        frames = [frame(0, [box()])]
        metrics = evaluate(frames, {("video1", 0): [[10, 10, 20, 20], [40, 10, 50, 20]]})
        self.assertAlmostEqual(metrics["AP50"], 51/101)

    def test_iou_threshold_includes_exact_boundary(self):
        frames = [frame(0, [{"bbox": [1, 0, 4, 2], "conf": 0.9}])]
        gt = {("video1", 0): [[0, 0, 3, 2]]}
        self.assertEqual(match_at_threshold(frames, gt, 0.5), ([1], [0]))
        self.assertEqual(match_at_threshold(frames, gt, 0.55), ([0], [1]))


class InterpolationTests(unittest.TestCase):
    def sequence(self, gap):
        return [frame(0, [box(10, 0.9)])] + [frame(i) for i in range(1, gap+1)] + [frame(gap+1, [box(10+10*(gap+1), 0.7)])]

    def test_linear_exact_endpoints_and_no_input_mutation(self):
        frames = self.sequence(1)
        before = copy.deepcopy(frames)
        result = interpolate(frames, CONFIG["interpolate"])
        self.assertEqual(result[1]["detections"][0]["bbox"], [20, 10, 30, 20])
        self.assertEqual(result[1]["detections"][0]["conf"], 0.7)
        self.assertEqual((result[0], result[-1]), (before[0], before[-1]))
        self.assertEqual(frames, before)

    def test_five_frame_gap_is_linear(self):
        result = interpolate(self.sequence(5), CONFIG["interpolate"])
        self.assertTrue(all(f["detections"] for f in result))
        self.assertEqual(result[3]["detections"][0]["interpolation"], "linear")

    def test_six_and_ten_frame_gaps_two_knot_spline(self):
        for gap in (6, 10):
            result = interpolate(self.sequence(gap), CONFIG["interpolate"])
            for i in range(1, gap+1):
                self.assertAlmostEqual(result[i]["detections"][0]["bbox"][0], 10+10*i)
                self.assertAlmostEqual(result[i]["detections"][0]["conf"], 0.56)

    def test_eleven_frame_gap_not_filled(self):
        result = interpolate(self.sequence(11), CONFIG["interpolate"])
        self.assertTrue(all(not f["detections"] for f in result[1:-1]))

    def test_no_extrapolation_before_or_after_anchor(self):
        frames = [frame(0), frame(1, [box()]), frame(2)]
        result = interpolate(frames, CONFIG["interpolate"])
        self.assertFalse(result[0]["detections"])
        self.assertFalse(result[2]["detections"])

    def test_never_interpolate_across_sequences(self):
        frames = [frame(0, [box()], "a"), frame(1, [], "a"), frame(0, [], "b"), frame(1, [box(40)], "b")]
        result = interpolate(frames, CONFIG["interpolate"])
        self.assertFalse(result[1]["detections"])
        self.assertFalse(result[2]["detections"])

    def test_all_empty_sequence_stays_empty(self):
        result = interpolate([frame(i) for i in range(3)], CONFIG["interpolate"])
        self.assertTrue(all(not f["detections"] for f in result))


class CleanAndManifestTests(unittest.TestCase):
    def test_actual_panorama_width(self):
        self.assertEqual(distance((3835, 20), (5, 20), 3840), 10)
        self.assertEqual(distance((1930, 20), (10, 20), 3840), 1920)

    def test_clean_keeps_only_first_retained_box(self):
        frames = [frame(0, [box(10), box(40)]), frame(1)]
        result = clean(frames, CONFIG["clean"])
        self.assertEqual(len(result[0]["detections"]), 1)
        self.assertEqual(result[0]["detections"][0]["bbox"], box(10)["bbox"])
        self.assertEqual(len(frames[0]["detections"]), 2)

    def test_motion_gate_uses_prediction_error_range(self):
        cfg = copy.deepcopy(CONFIG["clean"])
        cfg["protect_r0"] = cfg["protect_k"] = 0
        accepted = [frame(0, [box(10)]), frame(1, [box(30)]), frame(2, [box(55)])]
        perfect_prediction = [frame(0, [box(10)]), frame(1, [box(30)]), frame(2, [box(50)])]
        self.assertTrue(clean(accepted, cfg)[2]["detections"])
        # Audited lower error bound excludes exactly zero error when previous speed is nonzero.
        self.assertFalse(clean(perfect_prediction, cfg)[2]["detections"])

    def test_short_protection_forward_backward_and_window(self):
        cfg = CONFIG["clean"]
        history = [{"index": 0, "center": (100, 15)}, {"index": 1, "center": (110, 15)}]
        self.assertTrue(short_protect(3, box(125), history, 3840, cfg))
        self.assertTrue(short_protect(3, box(85), history, 3840, cfg))
        self.assertFalse(short_protect(3, box(500), history, 3840, cfg))
        self.assertFalse(short_protect(7, box(165), history, 3840, cfg))

    def test_clean_history_resets_at_boundary(self):
        frames = [frame(i, [box(10)], "a") for i in range(3)] + [frame(i, [], "b") for i in range(3)]
        result = clean(frames, CONFIG["clean"])
        self.assertTrue(all(not f["detections"] for f in result[3:]))

    def test_duplicate_out_of_order_and_missing_frame_rejected(self):
        for frames in ([frame(0), frame(0)], [frame(1), frame(0)], [frame(0), frame(2)]):
            with self.assertRaises(ValueError):
                validate_frames(frames)

    def test_gt_field_rejected_in_prediction_stage(self):
        f = frame(0)
        f["gt"] = []
        with self.assertRaises(ValueError):
            clean([f], CONFIG["clean"])

    def test_invalid_boxes_rejected(self):
        f = frame(0, [{"bbox": [2, 2, 1, 1], "conf": 0.5}])
        with self.assertRaises(ValueError):
            validate_frames([f])

    def test_missing_label_not_treated_as_empty_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                read_yolo_labels(Path(tmp)/"missing.txt", 3840, 1920)
            empty = Path(tmp)/"empty.txt"
            empty.write_text("")
            self.assertEqual(read_yolo_labels(empty, 3840, 1920), [])

    def test_numeric_sort_not_lexical_and_no_gt_access(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/"images/test"
            folder.mkdir(parents=True)
            for index in (10, 9, 8):
                Image.new("RGB", (16, 8)).save(folder/f"video1_{index}.png")
            manifest = build_manifest(Path(tmp), CONFIG["sequence_pattern"])
            self.assertEqual([f["frame_index"] for f in manifest["frames"]], [8, 9, 10])
            self.assertFalse((Path(tmp)/"labels").exists())

    def test_dataset_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                safe_child(tmp, "../outside.png")

    def test_other_model_initialization_rejected(self):
        cfg = copy.deepcopy(CONFIG)
        cfg["paths"]["pretrained_weights"] = "weights/other.pt"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"config.json"
            path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_official_legacy_checkpoint_metadata_accepted(self):
        class C2f:
            pass
        class Network:
            yaml = {"depth_multiple": 0.33, "width_multiple": 0.5, "nc": 80}
            def modules(self):
                return [C2f()]
        class Model:
            task = "detect"
            model = Network()
        assert_yolov8s(Model())
        Model.model.yaml = {"depth_multiple": 0.33, "width_multiple": 0.25, "nc": 80}
        with self.assertRaises(ValueError):
            assert_yolov8s(Model())

    def test_class_mapping_only_changes_copied_labels(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp)/"original", Path(tmp)/"copy"
            for split in ("train", "val", "test"):
                image_dir = source / "images" / split
                label_dir = source / "labels" / split
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                Image.new("RGB", (16, 8)).save(image_dir / "video1_0000.png")
                (label_dir / "video1_0000.txt").write_text("3 0.5 0.5 0.1 0.1\n")
            with self.assertRaises(ValueError):
                prepare_subset(source, dest, dict(train=1, val=1, test=1), CONFIG["sequence_pattern"])
            self.assertFalse(dest.exists())
            prepare_subset(source, dest, dict(train=1, val=1, test=1), CONFIG["sequence_pattern"], {"3": 0})
            self.assertTrue((source/"labels/train/video1_0000.txt").read_text().startswith("3 "))
            self.assertTrue((dest/"labels/train/video1_0000.txt").read_text().startswith("0 "))
            self.assertEqual((source/"images/train/video1_0000.png").read_bytes(), (dest/"images/train/video1_0000.png").read_bytes())


if __name__ == "__main__":
    unittest.main()
