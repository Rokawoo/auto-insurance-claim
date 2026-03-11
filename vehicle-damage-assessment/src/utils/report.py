"""Structured report generation for damage assessment results.

Produces a JSON-serializable damage report from DiffEngine output and
the heuristic DamageAnalyzer results.  No ML model involved — everything
is derived from pixel differencing and contour geometry.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.segmentation.damage_analyzer import AnalysisResult, DamageRegion


class ReportGenerator:
    """Generates structured JSON reports from pipeline results.

    The report is designed to be consumed by downstream insurance
    systems — it includes machine-readable damage regions, heuristic
    types, and a weighted severity estimate.
    """

    def generate(
        self,
        analysis: AnalysisResult,
        vehicle_area_px: int,
        image_shape: tuple[int, int],
        before_path: str = "",
        after_path: str = "",
        alignment_reliable: bool = True,
    ) -> dict:
        """Build the full damage report.

        Parameters
        ----------
        analysis : AnalysisResult
            Output of DamageAnalyzer.analyze().
        vehicle_area_px : int
            Total vehicle mask area in pixels.
        image_shape : tuple[int, int]
            (height, width) of the processed images.
        before_path : str
            Path to the before image (for reference).
        after_path : str
            Path to the after image.
        alignment_reliable : bool
            Whether the aligner flagged the alignment as reliable.

        Returns
        -------
        dict
            JSON-serializable damage report.
        """
        region_dicts = []
        for i, region in enumerate(analysis.regions):
            region_dicts.append(self._region_to_dict(region, i, vehicle_area_px))

        # sort worst-first
        region_dicts.sort(key=lambda r: r["severity_score"], reverse=True)

        report = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "before_image": str(before_path),
                "after_image": str(after_path),
                "image_dimensions": {
                    "height": image_shape[0],
                    "width": image_shape[1],
                },
                "vehicle_area_px": vehicle_area_px,
                "alignment_reliable": alignment_reliable,
            },
            "summary": {
                "overall_severity": analysis.overall_severity,
                "overall_severity_score": analysis.overall_severity_score,
                "total_damage_area_px": analysis.total_damage_area_px,
                "total_damage_area_pct": round(
                    analysis.total_damage_area_px / max(vehicle_area_px, 1) * 100, 2
                ),
                "num_damage_regions": len(analysis.regions),
                "damage_type_counts": analysis.damage_type_summary,
            },
            "regions": region_dicts,
        }

        if not alignment_reliable:
            report["warnings"] = [
                "Image alignment quality was below threshold. "
                "Pixel-diff results may include false positives from "
                "misalignment. Consider re-shooting with similar camera angle."
            ]

        return report

    def save(self, report: dict, path: str | Path) -> Path:
        """Write the report to a JSON file.

        Parameters
        ----------
        report : dict
            The report dictionary.
        path : str | Path
            Output file path.

        Returns
        -------
        Path
            Resolved path where the report was saved.
        """
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return path

    def _region_to_dict(
        self,
        region: DamageRegion,
        index: int,
        vehicle_area_px: int,
    ) -> dict:
        """Convert a DamageRegion to a report-friendly dict."""
        x, y, w, h = region.bbox
        area_pct = region.area_px / max(vehicle_area_px, 1) * 100

        return {
            "region_id": index,
            "damage_type": region.damage_type.value,
            "severity_score": region.severity_score,
            "bounding_box": {"x": x, "y": y, "width": w, "height": h},
            "area_px": region.area_px,
            "area_pct_of_vehicle": round(area_pct, 3),
            "mean_diff_intensity": region.mean_intensity,
            "geometry": {
                "circularity": getattr(region, "circularity", 0.0),
                "aspect_ratio": getattr(region, "aspect_ratio", 0.0),
                "solidity": getattr(region, "solidity", 0.0),
            },
        }
