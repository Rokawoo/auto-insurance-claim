"""Visualization helpers for pipeline outputs.

Draws polygon contours, damage masks, side-by-side comparisons, and
annotated overlay images for human review.
"""

from __future__ import annotations

import cv2
import numpy as np
from src.segmentation.damage_analyzer import DamageRegion


class Visualizer:
    """Creates annotated output images from pipeline results."""

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.save_viz: bool = config.get("save_visualizations", True)
        self.fmt: str = config.get("visualization_format", "png")

    def draw_damage_overlay(
        self,
        image: np.ndarray,
        regions: list[DamageRegion],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Draw semi-transparent polygon overlays with severity labels."""
        canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        overlay = canvas.copy()

        for region in regions:
            # Color based on severity: Green (0) to Red (1)
            # score 0.0 -> (0, 255, 0); score 1.0 -> (0, 0, 255)
            color = (0, int(255 * (1 - region.severity_score)), int(255 * region.severity_score))

            # 1. Draw the filled polygon (the "Wide Body" of the damage)
            cv2.drawContours(overlay, [region.contour], -1, color, thickness=-1)
            
            # 2. Draw the sharp outline
            cv2.drawContours(canvas, [region.contour], -1, color, thickness=2)

        # Blend the filled shapes
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

        # 3. Add Labels (Placed at the top-most point of the polygon, not a box corner)
        for region in regions:
            color = (0, int(255 * (1 - region.severity_score)), int(255 * region.severity_score))
            
            # Find the highest point of the contour to place the label
            most_top = tuple(region.contour[region.contour[:, :, 1].argmin()][0])
            label = f"Score: {region.severity_score:.2f}"
            
            cv2.putText(
                canvas, label, (most_top[0], max(most_top[1] - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

        return canvas

    def draw_vehicle_mask(
        self,
        image: np.ndarray,
        vehicle_mask: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Draws a polygon outline of the vehicle mask instead of a bounding box."""
        canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Extract contours from the binary vehicle mask
        contours, _ = cv2.findContours(vehicle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw the actual shape of the car
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
        """Create a 2x2 summary grid using polygon-based overlays."""
        h, w = before.shape[:2]
        
        def to_bgr(img):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()

        # Build panels
        panel_before = to_bgr(before)
        panel_after = to_bgr(after)
        
        # Heatmap of the raw differences
        heatmap = cv2.applyColorMap(
            diff_mask if diff_mask.ndim == 2 else cv2.cvtColor(diff_mask, cv2.COLOR_BGR2GRAY), 
            cv2.COLORMAP_JET
        )
        
        # Polygon annotated version
        panel_annotated = self.draw_damage_overlay(after, regions)

        panels = [panel_before, panel_after, heatmap, panel_annotated]
        labels = ["Before", "After (aligned)", "Raw Diff Map", f"Severity: {overall_severity}"]

        for i, panel in enumerate(panels):
            p_h, p_w = panel.shape[:2]
            if (p_h, p_w) != (h, w):
                panel = cv2.resize(panel, (w, h))
            
            cv2.putText(
                panel, labels[i], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )
            panels[i] = panel

        top = np.hstack([panels[0], panels[1]])
        bottom = np.hstack([panels[2], panels[3]])
        return np.vstack([top, bottom])