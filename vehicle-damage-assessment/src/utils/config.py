"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file, merged with defaults.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed and merged configuration.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        user_config = yaml.safe_load(f) or {}

    return _deep_merge(_get_defaults(), user_config)


def _get_defaults() -> dict:
    """Default values for all config sections."""
    return {
        "preprocessing": {
            "target_size": [640, 640],
            "grayscale": True,
            "blur_kernel": 5,
            "blur_sigma": 0,
            "clahe": {
                "enabled": True,
                "clip_limit": 2.0,
                "tile_grid_size": [8, 8],
            },
        },
        "alignment": {
            "feature_method": "orb",
            "max_features": 10000,
            "match_method": "bf",
            "ratio_threshold": 0.75,
            "ransac_reproj_threshold": 5.0,
            "min_match_count": 10,
            "warp_method": "homography",
            "max_reprojection_error": 5.0,
            "min_inlier_ratio": 0.25,
            "fallback": True,
            "normalize_exposure": True,
        },
        "detection": {
            "model_name": "yolo11m.pt",
            "vehicle_classes": [2, 5, 7],
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
            "mask_dilation_kernel": 15,
        },
        "comparison": {
            "diff_method": "absolute",
            "threshold": 30,
            "morph_kernel": 5,
            "morph_iterations": 2,
            "min_contour_area": 500,
            "max_contour_area": 100000,
        },
        "analysis": {
            "intensity_weight": 0.40,
            "area_weight": 0.35,
            "count_weight": 0.25,
            "scratch_min_aspect": 3.0,
            "scratch_max_solidity": 0.6,
            "dent_min_circularity": 0.4,
            "shatter_min_area_frac": 0.05,
            "scuff_max_intensity": 40.0,
            "scuff_max_area": 2000,
        },
        "output": {
            "save_visualizations": True,
            "save_report": True,
            "output_dir": "outputs",
            "visualization_format": "png",
        },
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
