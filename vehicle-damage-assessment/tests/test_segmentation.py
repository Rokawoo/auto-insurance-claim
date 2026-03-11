import cv2
import numpy as np
from pathlib import Path

from src.segmentation import DamageSegmentor


def build_config():

    return {
        "enabled": True,
        "model_path": "models/pretrained/yolov8n-seg.pt",
        "damage_classes": {
            0: "scratch",
            1: "dent",
            2: "crack",
        },
        "confidence_threshold": 0.25,
    }


def test_model_loads():

    config = build_config()

    seg = DamageSegmentor(config)

    assert seg.is_available() or Path(config["model_path"]).exists()


def test_segmentation_runs():

    config = build_config()

    seg = DamageSegmentor(config)

    image = np.zeros((512, 512, 3), dtype=np.uint8)

    result = seg.segment(image)

    assert result is not None
    assert isinstance(result.instances, list)


def test_candidate_filtering():

    config = build_config()

    seg = DamageSegmentor(config)

    image = np.zeros((512, 512, 3), dtype=np.uint8)

    candidate_mask = np.zeros((512, 512), dtype=np.uint8)
    candidate_mask[100:200, 100:200] = 1

    result = seg.segment_with_candidates(image, candidate_mask)

    assert result is not None
    assert isinstance(result.instances, list)