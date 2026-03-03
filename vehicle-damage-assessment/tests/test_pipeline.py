"""Integration tests for the full DamagePipeline.

These tests run the complete pipeline end-to-end on synthetic data.
They require model weights to be available (skip if not present).
"""

import numpy as np
import pytest
from pathlib import Path

from src.pipeline import DamagePipeline
from src.pipeline.damage_pipeline import PipelineResult
from src.utils.config import load_config


@pytest.fixture
def config():
    """Load default config (or return a hardcoded dict for CI)."""
    config_path = Path("configs/default.yaml")
    if config_path.exists():
        return load_config(config_path)
    # fallback for CI without config file
    return {
        "preprocessing": {"target_size": [320, 320], "grayscale": True, "blur_kernel": 5},
        "alignment": {"feature_method": "orb", "max_features": 3000},
        "detection": {"model_name": "yolov8s.pt", "vehicle_classes": [2, 5, 7]},
        "comparison": {"threshold": 30, "min_contour_area": 100},
        "segmentation": {"enabled": False},
        "output": {"save_visualizations": False},
    }


class TestDamagePipeline:

    def test_pipeline_instantiation(self, config):
        """Pipeline should instantiate all sub-modules."""
        # TODO: create DamagePipeline, assert all sub-modules exist
        pass

    @pytest.mark.skipif(
        not Path("models/pretrained/yolov8s.pt").exists(),
        reason="YOLOv8 weights not downloaded"
    )
    def test_run_returns_pipeline_result(self, config, tmp_path):
        """Full run on synthetic images should return PipelineResult."""
        # TODO:
        #   1. create synthetic before/after images with known damage
        #   2. save to tmp_path
        #   3. run pipeline
        #   4. assert isinstance(result, PipelineResult)
        pass

    def test_report_is_json_serializable(self, config):
        """The report dict should be JSON-serializable."""
        # TODO: json.dumps(result.report) should not raise
        pass
