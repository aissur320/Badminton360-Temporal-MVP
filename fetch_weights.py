"""Explicit download of the single supported pretrained initializer, never a tracker/model fallback."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt"


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(root / "weights/yolov8s.pt"))
    output = Path(parser.parse_args().output).resolve()
    if output.name != "yolov8s.pt" or not output.is_relative_to(root):
        parser.error("Only yolov8s.pt inside this MVP can be downloaded")
    if output.exists():
        parser.error("Refusing to overwrite an existing checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Buffer before writing: a failed HTTP request cannot leave a partial checkpoint.
    with urllib.request.urlopen(URL, timeout=120) as response:
        payload = response.read()
    if len(payload) < 1_000_000:
        raise ValueError("Unexpected checkpoint response size")
    output.write_bytes(payload)
    print(json.dumps({"file": output.name, "url": URL, "bytes": len(payload),
                      "sha256": hashlib.sha256(payload).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
