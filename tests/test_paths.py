"""Portability regressions: move the project and change CWD without retraining."""
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import research_mvp.common as common
import research_mvp.pipeline as pipeline

CONFIG = json.loads((common.ROOT / "configs/smoke.json").read_text(encoding="utf-8"))


class PortablePathsTests(unittest.TestCase):
    def test_saved_config_contains_only_project_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = copy.deepcopy(CONFIG)
            expected = copy.deepcopy(config["paths"])
            config["paths"] = {key: str(root / value) for key, value in expected.items()}
            self.assertEqual(common.portable_config(config, root=root)["paths"], expected)
            self.assertTrue(all(Path(value).is_absolute() for value in config["paths"].values()))

    def test_run_config_still_matches_after_project_relocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, moved = Path(tmp) / "original", Path(tmp) / "moved"
            original.mkdir()
            config = copy.deepcopy(CONFIG)
            config["run_name"] = "relocation"
            common.write_json(original / "config.json", config)
            with patch.object(common, "ROOT", original), patch.object(pipeline, "ROOT", original):
                runtime = pipeline.load_config(original / "config.json")
                common.write_json(pipeline.run_dir(runtime) / "config.resolved.json", common.portable_config(runtime))
            shutil.copytree(original, moved)
            with patch.object(common, "ROOT", moved), patch.object(pipeline, "ROOT", moved):
                runtime = pipeline.load_config(moved / "config.json")
                self.assertEqual(pipeline.verify_run_config(runtime), moved / "outputs/relocation")
                runtime["predict"]["conf"] = 0.5
                with self.assertRaises(ValueError):
                    pipeline.verify_run_config(runtime)

    def test_dataset_yaml_resolves_from_its_directory_not_working_directory(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            yaml_path = root / "outputs/run/dataset.yaml"
            data_root = root / "data/project"
            for split in ("train", "val"):
                (data_root / "images" / split).mkdir(parents=True)
            with patch.object(common, "ROOT", root):
                value = pipeline.dataset_config(data_root, yaml_path)
            self.assertNotIn("path", value)
            yaml_path.parent.mkdir(parents=True)
            yaml_path.write_text(yaml.safe_dump(value), encoding="utf-8")
            for split in ("train", "val"):
                self.assertFalse(Path(value[split]).is_absolute())
                self.assertEqual((yaml_path.parent / value[split]).resolve(), data_root / "images" / split)

    def test_ultralytics_dataset_loader_accepts_relative_yaml_after_relocation(self):
        import yaml
        # Import only the library's dataset checker; never construct/train a model.
        pipeline.configure_runtime()
        from ultralytics.data.utils import check_det_dataset
        with tempfile.TemporaryDirectory() as tmp:
            original, moved = Path(tmp) / "original", Path(tmp) / "moved"
            data_root = original / "data/project"
            yaml_path = original / "outputs/run/dataset.yaml"
            for split in ("train", "val"):
                (data_root / "images" / split).mkdir(parents=True)
            yaml_path.parent.mkdir(parents=True)
            with patch.object(common, "ROOT", original):
                yaml_path.write_text(yaml.safe_dump(pipeline.dataset_config(data_root, yaml_path)), encoding="utf-8")
            shutil.copytree(original, moved)
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                with patch("ultralytics.data.utils.check_font"):
                    resolved = check_det_dataset(str(moved / "outputs/run/dataset.yaml"), autodownload=False)
            finally:
                os.chdir(previous)
            self.assertEqual(Path(resolved["train"]), moved / "data/project/images/train")
            self.assertEqual(Path(resolved["val"]), moved / "data/project/images/val")

    def test_library_args_are_relative_and_nonpath_values_preserved(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "outputs/run/train/args.yaml"
            args = {"model": str(root / "weights/yolov8s.pt"), "data": str(root / "outputs/run/dataset.yaml"),
                    "project": str(root / "outputs/run"), "save_dir": str(root / "outputs/run/train"),
                    "epochs": 1, "device": "cpu", "resume": False, "pretrained": True}
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump(args), encoding="utf-8")
            with patch.object(common, "ROOT", root):
                pipeline.portable_training_args(path)
            result = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(result["model"], "weights/yolov8s.pt")
            self.assertEqual(result["data"], "outputs/run/dataset.yaml")
            self.assertEqual(result["save_dir"], "outputs/run/train")
            self.assertEqual(result["epochs"], 1)
            self.assertIs(result["resume"], False)
            self.assertIs(result["pretrained"], True)


if __name__ == "__main__":
    unittest.main()
