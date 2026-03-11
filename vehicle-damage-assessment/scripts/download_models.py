#!/usr/bin/env python3
"""
Download pretrained models required for the project.
"""

from pathlib import Path
from ultralytics import YOLO


def download_yolo_seg():
    model_dir = Path("models/pretrained")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "yolov8n-seg.pt"

    if model_path.exists():
        print("Model already exists:", model_path)
        return

    print("Downloading YOLOv8 segmentation model...")
    model = YOLO("yolov8n-seg.pt")
    model.save(str(model_path))

    print("Saved to:", model_path)


def main():
    download_yolo_seg()

if __name__ == "__main__":
    main()