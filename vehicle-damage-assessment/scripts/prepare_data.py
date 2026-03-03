#!/usr/bin/env python3
"""Prepare and augment data for training / pipeline testing.

Handles:
- Organizing raw before/after image pairs
- Converting annotation formats (Label Studio → YOLO)
- Basic data augmentation for training
- Generating the YOLO dataset YAML file
- Train/val/test splitting

Usage
-----
    python scripts/prepare_data.py \\
        --raw-dir data/raw \\
        --output-dir data/processed \\
        --split-ratio 0.7 0.15 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    """Parse data preparation CLI arguments."""
    # TODO:
    #   --raw-dir (path to raw data)
    #   --annotations-dir (path to label studio / CVAT exports)
    #   --output-dir (where to write processed dataset)
    #   --split-ratio (train val test fractions, e.g. 0.7 0.15 0.15)
    #   --augment (flag to enable augmentation)
    raise NotImplementedError


def convert_annotations(annotations_dir: Path, output_dir: Path) -> None:
    """Convert Label Studio JSON annotations to YOLO segmentation format.

    YOLO seg format: one .txt per image, each line is:
        class_id x1 y1 x2 y2 ... xN yN  (normalized polygon coords)

    Parameters
    ----------
    annotations_dir : Path
        Directory containing Label Studio JSON exports.
    output_dir : Path
        Directory to write YOLO-format .txt label files.
    """
    # TODO:
    #   1. load each JSON annotation file
    #   2. extract polygon coordinates per labeled region
    #   3. normalize coords to [0, 1] based on image dimensions
    #   4. map label names to class IDs
    #   5. write one .txt per image
    raise NotImplementedError


def split_dataset(
    image_dir: Path, label_dir: Path, output_dir: Path, ratios: tuple[float, float, float]
) -> None:
    """Split images and labels into train/val/test sets.

    Parameters
    ----------
    image_dir : Path
        Directory of images.
    label_dir : Path
        Directory of YOLO label .txt files.
    output_dir : Path
        Root output dir (will create train/, val/, test/ subdirs).
    ratios : tuple[float, float, float]
        (train_ratio, val_ratio, test_ratio) — must sum to 1.0.
    """
    # TODO:
    #   1. list all image files
    #   2. shuffle deterministically (seed)
    #   3. split into train/val/test
    #   4. copy images and labels to respective subdirs
    raise NotImplementedError


def generate_dataset_yaml(output_dir: Path, class_names: list[str]) -> Path:
    """Generate the YOLO dataset.yaml file.

    Parameters
    ----------
    output_dir : Path
        Root dataset directory.
    class_names : list[str]
        Ordered list of class names.

    Returns
    -------
    Path
        Path to the generated YAML file.
    """
    # TODO:
    #   write a YAML with:
    #     path: output_dir
    #     train: train/images
    #     val: val/images
    #     test: test/images
    #     names: {0: class_names[0], ...}
    raise NotImplementedError


def main() -> None:
    """Entry point."""
    # TODO: parse args, run conversion + splitting + yaml generation
    raise NotImplementedError


if __name__ == "__main__":
    main()
