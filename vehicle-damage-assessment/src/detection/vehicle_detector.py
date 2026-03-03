"""YOLOv8-based vehicle detection and ROI mask generation.

Detects vehicles in the image so that the comparison stage only considers
pixels belonging to the car body.  Without this, background changes
(different parking lot, new sign, etc.) would be flagged as "damage."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DetectionResult:
    """Container for vehicle detection outputs.

    Attributes
    ----------
    boxes : np.ndarray
        Bounding boxes in xyxy format, shape (N, 4).
    confidences : np.ndarray
        Detection confidence scores, shape (N,).
    class_ids : np.ndarray
        COCO class IDs for each detection, shape (N,).
    vehicle_mask : np.ndarray
        Binary mask (H, W) — 255 inside detected vehicle(s), 0 outside.
        Used to constrain the comparison stage to vehicle pixels only.
    """
    boxes: np.ndarray = field(repr=False)
    confidences: np.ndarray = field(repr=False)
    class_ids: np.ndarray = field(repr=False)
    vehicle_mask: np.ndarray = field(repr=False)


class VehicleDetector:
    """Detects vehicles and produces a binary ROI mask.

    Uses a pretrained YOLOv8 model (COCO weights) to find cars, trucks,
    and buses.  The bounding boxes are converted into a dilated binary
    mask that the comparison stage uses to ignore non-vehicle pixels.

    Parameters
    ----------
    config : dict
        The ``detection`` section of the pipeline config.
    """

    def __init__(self, config: dict) -> None:
        self.model_name: str = config.get("model_name", "yolov8s.pt")
        self.vehicle_classes: list[int] = config.get("vehicle_classes", [2, 5, 7])
        self.conf_threshold: float = config.get("confidence_threshold", 0.5)
        self.iou_threshold: float = config.get("iou_threshold", 0.45)
        self.mask_dilation_kernel: int = config.get("mask_dilation_kernel", 15)

        self.model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the YOLOv8 model weights.

        Separated from __init__ so that the pipeline can control when the
        (potentially large) model is loaded into memory.
        """
        # TODO:
        #   from ultralytics import YOLO
        #   self.model = YOLO(self.model_name)
        raise NotImplementedError

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run vehicle detection on a single image.

        Parameters
        ----------
        image : np.ndarray
            BGR image (original resolution is fine — YOLO resizes internally).

        Returns
        -------
        DetectionResult
            Boxes, confidences, class IDs, and the binary vehicle mask.
        """
        # TODO:
        #   1. ensure model is loaded
        #   2. run inference with conf and iou thresholds
        #   3. filter to vehicle_classes only
        #   4. build binary mask from bounding boxes
        #   5. dilate mask to include edges/bumpers
        #   6. return DetectionResult
        raise NotImplementedError

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _filter_vehicle_detections(self, results) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter YOLO results to keep only vehicle classes.

        Parameters
        ----------
        results : ultralytics.engine.results.Results
            Raw YOLO inference results.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (filtered_boxes_xyxy, filtered_confidences, filtered_class_ids)
        """
        # TODO: iterate results[0].boxes, keep entries whose cls is in self.vehicle_classes
        raise NotImplementedError

    def _boxes_to_mask(
        self, boxes: np.ndarray, image_shape: tuple[int, int]
    ) -> np.ndarray:
        """Convert bounding boxes to a filled binary mask.

        Parameters
        ----------
        boxes : np.ndarray
            Bounding boxes in xyxy format (N, 4).
        image_shape : tuple[int, int]
            (height, width) of the target mask.

        Returns
        -------
        np.ndarray
            Binary mask, dtype uint8, values {0, 255}.
        """
        # TODO:
        #   1. create zero mask of image_shape
        #   2. for each box, draw filled rectangle
        #   3. dilate with self.mask_dilation_kernel
        raise NotImplementedError
