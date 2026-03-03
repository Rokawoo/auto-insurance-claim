#!/usr/bin/env python3
"""Evaluate the damage segmentation model on a test set.

Usage
-----
    python scripts/evaluate.py --config configs/training.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    # TODO: --config, --model (optional override), --split
    raise NotImplementedError


def evaluate(config: dict) -> dict:
    """Run evaluation and print metrics.

    Parameters
    ----------
    config : dict
        The ``evaluation`` section of the config.

    Returns
    -------
    dict
        Metrics dictionary (mAP50, mAP50-95, precision, recall).
    """
    # TODO:
    #   1. from ultralytics import YOLO
    #   2. model = YOLO(config["model_path"])
    #   3. metrics = model.val(
    #        data=config["data_yaml"],
    #        split=config["split"],
    #        iou=config["iou_threshold"],
    #        conf=config["conf_threshold"],
    #   )
    #   4. extract and return relevant metrics
    raise NotImplementedError


def main() -> None:
    """Entry point."""
    # TODO: parse args, load config, run evaluate, print results
    raise NotImplementedError


if __name__ == "__main__":
    main()
