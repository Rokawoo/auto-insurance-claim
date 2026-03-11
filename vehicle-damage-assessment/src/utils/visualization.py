"""Visualization helpers for pipeline outputs.

Draws bounding boxes, damage masks, side-by-side comparisons, and
annotated overlay images for human review.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.segmentation.damage_analyzer import DamageRegion, DamageType


# color palette for damage types (BGR)
DAMAGE_COLORS: dict[str, tuple[int, int, int]] = {
    "scratch": (0, 255, 255),      # yellow
    "dent": (0, 165, 255),         # orange
    "crack": (0, 0, 255),          # red
    "shatter": (255, 0, 255),      # magenta
    "scuff": (200, 200, 200),      # light gray
    "unknown": (0, 255, 0),        # green
}


class Visualizer:
    """Creates annotated output images from pipeline results.

    Parameters
    ----------
    config : dict
        The ``output`` section of the pipeline config.
    """

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.save_viz: bool = config.get("save_visualizations", True)
        self.fmt: str = config.get("visualization_format", "png")

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def draw_damage_overlay(
        self,
        image: np.ndarray,
        regions: list[DamageRegion],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Draw semi-transparent damage region overlays with labels.

        Parameters
        ----------
        image : np.ndarray
            BGR image to annotate (will be copied, not modified in place).
        regions : list[DamageRegion]
            Analyzed damage regions from DamageAnalyzer.
        alpha : float
            Transparency of the overlay (0 = invisible, 1 = opaque).

        Returns
        -------
        np.ndarray
            Annotated image.
        """
        # ensure 3-channel for drawing
        if image.ndim == 2:
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = image.copy()

        overlay = canvas.copy()

        for region in regions:
            color = DAMAGE_COLORS.get(region.damage_type.value, (0, 255, 0))

            # filled contour on overlay
            cv2.drawContours(overlay, [region.contour], -1, color, thickness=-1)

        # blend
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

        # draw outlines and labels on top of the blend
        for region in regions:
            color = DAMAGE_COLORS.get(region.damage_type.value, (0, 255, 0))

            # contour outline
            cv2.drawContours(canvas, [region.contour], -1, color, thickness=2)

            # label text
            x, y, w, h = region.bbox
            label = f"{region.damage_type.value} ({region.severity_score:.2f})"
            text_y = max(y - 8, 16)

            # background rectangle for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x, text_y - th - 4), (x + tw + 4, text_y + 4), color, -1)
            cv2.putText(
                canvas, label, (x + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

        return canvas

    def draw_side_by_side(
        self,
        before: np.ndarray,
        after: np.ndarray,
        diff_mask: np.ndarray | None = None,
        labels: tuple[str, ...] = ("Before", "After", "Diff"),
    ) -> np.ndarray:
        """Create a side-by-side comparison image.

        Parameters
        ----------
        before : np.ndarray
            "Before" image.
        after : np.ndarray
            "After" image (warped/aligned).
        diff_mask : np.ndarray | None
            Binary or grayscale diff mask (will be colorized as heatmap).
        labels : tuple[str, ...]
            Panel labels.

        Returns
        -------
        np.ndarray
            Horizontally concatenated comparison image.
        """
        panels = []
        for img in [before, after]:
            if img.ndim == 2:
                panels.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            else:
                panels.append(img.copy())

        if diff_mask is not None:
            if diff_mask.ndim == 2:
                heatmap = cv2.applyColorMap(diff_mask, cv2.COLORMAP_JET)
            else:
                heatmap = diff_mask.copy()
            panels.append(heatmap)

        # ensure all panels are the same height
        target_h = panels[0].shape[0]
        for i, panel in enumerate(panels):
            if panel.shape[0] != target_h:
                scale = target_h / panel.shape[0]
                new_w = int(panel.shape[1] * scale)
                panels[i] = cv2.resize(panel, (new_w, target_h))

        # add labels
        for i, panel in enumerate(panels):
            label = labels[i] if i < len(labels) else ""
            if label:
                cv2.putText(
                    panel, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
                )

        return np.hstack(panels)

    def draw_alignment_matches(
        self,
        before: np.ndarray,
        after: np.ndarray,
        keypoints_before: list,
        keypoints_after: list,
        matches: list,
    ) -> np.ndarray:
        """Visualize feature matches between before/after images.

        Parameters
        ----------
        before, after : np.ndarray
        keypoints_before, keypoints_after : list[cv2.KeyPoint]
        matches : list[cv2.DMatch]

        Returns
        -------
        np.ndarray
            Image with match lines drawn.
        """
        return cv2.drawMatches(
            before, keypoints_before,
            after, keypoints_after,
            matches[:50],  # cap at 50 for readability
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    def draw_detection_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        confidences: np.ndarray,
    ) -> np.ndarray:
        """Draw vehicle detection bounding boxes.

        Parameters
        ----------
        image : np.ndarray
            BGR image.
        boxes : np.ndarray
            xyxy boxes (N, 4).
        confidences : np.ndarray
            Detection scores (N,).

        Returns
        -------
        np.ndarray
            Annotated image.
        """
        canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"vehicle {conf:.2f}"
            cv2.putText(
                canvas, label, (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )

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
        """Create a 2x2 summary grid: before, after, diff heatmap, annotated.

        Parameters
        ----------
        before : np.ndarray
            Original before image.
        after : np.ndarray
            Aligned after image.
        diff_mask : np.ndarray
            Raw diff image.
        regions : list[DamageRegion]
            Classified damage regions.
        overall_severity : str
            "none" / "minor" / "moderate" / "severe" / "critical"
        severity_score : float
            0.0–1.0

        Returns
        -------
        np.ndarray
            2x2 grid summary image.
        """
        def to_bgr(img):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()

        # ensure same size
        h, w = before.shape[:2]
        panels = [
            to_bgr(before),
            to_bgr(after),
            cv2.applyColorMap(diff_mask if diff_mask.ndim == 2 else cv2.cvtColor(diff_mask, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET),
            self.draw_damage_overlay(after, regions),
        ]

        for i, p in enumerate(panels):
            panels[i] = cv2.resize(p, (w, h))

        labels = ["Before", "After (aligned)", "Diff Heatmap",
                   f"Damage: {overall_severity} ({severity_score:.2f})"]
        for i, panel in enumerate(panels):
            cv2.putText(
                panel, labels[i], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )

        top = np.hstack([panels[0], panels[1]])
        bottom = np.hstack([panels[2], panels[3]])
        return np.vstack([top, bottom])
