#!/usr/bin/env python3
"""Train / fine-tune YOLOv8-seg for damage type segmentation.

Usage
-----
    python scripts/train_damage_model.py --config configs/training.yaml

Prerequisites
-------------
- A damage dataset in YOLO segmentation format (see data/damage_dataset.yaml)
- Labeled data with polygon annotations for damage types
- GPU recommended (training on CPU is very slow)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments.

    Returns
    -------
    argparse.Namespace
        --config, --resume (optional checkpoint to resume from)
    """
    # TODO: argparse setup
    raise NotImplementedError


def train(config: dict) -> None:
    """Run the YOLOv8-seg fine-tuning loop.

    Parameters
    ----------
    config : dict
        The ``training`` section of the config.
    """
    # TODO:
    #   1. from ultralytics import YOLO
    #   2. model = YOLO(config["base_model"])
    #   3. model.train(
    #        data=config["data_yaml"],
    #        epochs=config["epochs"],
    #        batch=config["batch_size"],
    #        imgsz=config["image_size"],
    #        patience=config["patience"],
    #        optimizer=config["optimizer"],
    #        lr0=config["lr0"],
    #        lrf=config["lrf"],
    #        weight_decay=config["weight_decay"],
    #        project=config["save_dir"],
    #        name=config["project_name"],
    #        **config.get("augmentation", {}),
    #   )
    #   4. print final metrics
    raise NotImplementedError


def main() -> None:
    """Entry point."""
    # TODO: parse args, load config, call train
    raise NotImplementedError


if __name__ == "__main__":
    main()
