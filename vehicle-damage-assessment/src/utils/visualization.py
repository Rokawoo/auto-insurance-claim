"""Visualization helpers for pipeline outputs.

Draws bounding boxes, damage masks, side-by-side comparisons, and
annotated overlay images for human review.
"""

from __future__ import annotations

import cv2
import numpy as np


# color palette for damage types (BGR)
DAMAGE_COLORS = {
    "scratch": (0, 255, 255),     # yellow
    "dent": (0, 165, 255),        # orange
    "crack": (0, 0, 255),         # red
    "shatter": (255, 0, 255),     # magenta
    "deformation": (255, 0, 0),   # blue
    "unknown": (0, 255, 0),       # green (diff-only, no classification)
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
        contours: list,
        labels: list[str] | None = None,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Draw semi-transparent damage region overlays on the image.

        Parameters
        ----------
        image : np.ndarray
            BGR image to annotate (will be copied, not modified in place).
        contours : list
            List of cv2 contours representing damage regions.
        labels : list[str] | None
            Damage type label per contour.  If None, all are "unknown".
        alpha : float
            Transparency of the overlay (0 = invisible, 1 = opaque).

        Returns
        -------
        np.ndarray
            Annotated image.
        """
        # TODO:
        #   1. copy image
        #   2. for each contour, draw filled polygon on overlay
        #   3. blend overlay with original using alpha
        #   4. draw contour outlines and labels
        raise NotImplementedError

    def draw_side_by_side(
        self,
        before: np.ndarray,
        after: np.ndarray,
        diff_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Create a side-by-side comparison image.

        Optionally includes the diff mask as a third panel.

        Parameters
        ----------
        before : np.ndarray
            "Before" image.
        after : np.ndarray
            "After" image (warped/aligned).
        diff_mask : np.ndarray | None
            Binary difference mask (will be colorized).

        Returns
        -------
        np.ndarray
            Horizontally concatenated comparison image.
        """
        # TODO:
        #   1. ensure all images are same height
        #   2. if diff_mask, convert to BGR heatmap (cv2.applyColorMap)
        #   3. np.hstack
        raise NotImplementedError

    def draw_alignment_matches(
        self,
        before: np.ndarray,
        after: np.ndarray,
        keypoints_before: list,
        keypoints_after: list,
        matches: list,
    ) -> np.ndarray:
        """Visualize feature matches between before/after images.

        Useful for debugging alignment quality.

        Parameters
        ----------
        before, after : np.ndarray
            Input images.
        keypoints_before, keypoints_after : list[cv2.KeyPoint]
        matches : list[cv2.DMatch]

        Returns
        -------
        np.ndarray
            Image with match lines drawn.
        """
        # TODO: cv2.drawMatches
        raise NotImplementedError

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
        # TODO: for each box, cv2.rectangle + cv2.putText with confidence
        raise NotImplementedError
