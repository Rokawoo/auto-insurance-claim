"""Visualization helpers for pipeline outputs."""

from __future__ import annotations

import cv2
import numpy as np
from src.segmentation.damage_analyzer import DamageRegion


def _severity_to_color(score: float) -> tuple[int, int, int]:
    """Maps severity score [0,1] to BGR color (green->red)."""
    s = float(np.clip(score, 0.0, 1.0))
    return (0, int(255 * (1 - s)), int(255 * s))


class Visualizer:
    """Creates annotated output images from pipeline results."""

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.save_viz: bool = config.get("save_visualizations", True)
        self.fmt: str = config.get("visualization_format", "png")

    def _ensure_bgr(self, img: np.ndarray) -> np.ndarray:
        """Ensure image is BGR."""
        if img is None:
            raise ValueError("Image cannot be None")
        return img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def draw_damage_overlay(
        self,
        image: np.ndarray,
        regions: list[DamageRegion],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Draw polygon overlays with severity labels."""
        canvas = self._ensure_bgr(image)
        overlay = canvas.copy()

        if not regions:
            return canvas

        for region in regions:
            if region.contour is None or len(region.contour) == 0:
                continue

            score = getattr(region, "severity_score", 0.0)
            color = _severity_to_color(score)

            cv2.drawContours(overlay, [region.contour], -1, color, -1)

        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

        for region in regions:
            if region.contour is None or len(region.contour) == 0:
                continue

            score = getattr(region, "severity_score", 0.0)
            color = _severity_to_color(score)

            cv2.drawContours(canvas, [region.contour], -1, color, 2)

            try:
                top_point = tuple(region.contour[region.contour[:, :, 1].argmin()][0])
            except Exception:
                continue

            label = f"{score:.2f}"
            text_y = max(top_point[1] - 8, 16)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(
                canvas,
                (top_point[0], text_y - th - 4),
                (top_point[0] + tw + 4, text_y + 4),
                color,
                -1,
            )

            cv2.putText(
                canvas,
                label,
                (top_point[0] + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return canvas

    def draw_vehicle_mask(
        self,
        image: np.ndarray,
        vehicle_mask: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw polygon outline of vehicle mask."""
        canvas = self._ensure_bgr(image)

        if vehicle_mask is None:
            return canvas

        mask = vehicle_mask

        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if contours:
            cv2.drawContours(canvas, contours, -1, color, thickness)

        return canvas

    def create_summary_image(
        self,
        before: np.ndarray,
        after: np.ndarray,
        diff_mask: np.ndarray,
        regions: list[DamageRegion],
        overall_severity: str,
        severity_score: float,
    ) -> np.ndarray:
        """Create a 2x2 summary grid."""
        h, w = before.shape[:2]

        before = self._ensure_bgr(before)
        after = self._ensure_bgr(after)

        diff = diff_mask
        if diff.ndim == 3:
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

        annotated = self.draw_damage_overlay(after, regions)

        panels = [before, after, heatmap, annotated]
        labels = [
            "Before",
            "After (aligned)",
            "Diff Heatmap",
            f"Damage: {overall_severity} ({severity_score:.2f})",
        ]

        for i, panel in enumerate(panels):
            if panel.shape[:2] != (h, w):
                panel = cv2.resize(panel, (w, h))

            cv2.putText(
                panel,
                labels[i],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            panels[i] = panel

        top = np.hstack([panels[0], panels[1]])
        bottom = np.hstack([panels[2], panels[3]])

        return np.vstack([top, bottom])
