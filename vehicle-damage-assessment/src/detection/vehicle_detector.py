"""YOLOv8-based vehicle detection and ROI mask generation.

This is the critical gatekeeper between alignment and comparison.
Without it, the pixel diff picks up background changes, warp borders,
lighting shifts — anything that isn't the car.  This module runs a
pretrained YOLOv8 model (COCO weights) to find vehicles, then produces
a binary mask so the diff engine only looks at car pixels.

NOTE: This module uses YOLOv8 ONLY for vehicle localization (pretrained
COCO weights).  No fine-tuned damage model is needed — damage
classification is handled downstream by the heuristic DamageAnalyzer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# COCO class ID reference (the ones we care about):
#   2 = car
#   3 = motorcycle
#   5 = bus
#   7 = truck
COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ══════════════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════════════

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
    num_vehicles : int
        Number of vehicles detected.
    best_confidence : float
        Highest detection confidence among all vehicles.
    """

    boxes: np.ndarray = field(repr=False)
    confidences: np.ndarray = field(repr=False)
    class_ids: np.ndarray = field(repr=False)
    vehicle_mask: np.ndarray = field(repr=False)
    num_vehicles: int = 0
    best_confidence: float = 0.0


# ══════════════════════════════════════════════════════════════════════
# Detector
# ══════════════════════════════════════════════════════════════════════

class VehicleDetector:
    """Detects vehicles and produces a binary ROI mask.

    Uses a pretrained YOLOv8 model (COCO weights) to find cars, trucks,
    and buses.  The bounding boxes are converted into a dilated binary
    mask that downstream stages use to ignore non-vehicle pixels.

    Parameters
    ----------
    config : dict
        The ``detection`` section of the pipeline config.  Keys:

        model_name : str
            YOLOv8 weight file (default ``"yolo11m.pt"``).
            Ultralytics auto-downloads on first use (~22 MB for yolov8s).
        vehicle_classes : list[int]
            COCO class IDs to treat as vehicles (default [2, 5, 7]).
        confidence_threshold : float
            Minimum detection confidence (default 0.5).
        iou_threshold : float
            NMS IoU threshold (default 0.45).
        mask_dilation_kernel : int
            Elliptical kernel size for mask dilation; 0 = no dilation
            (default 15).  Larger values give more margin around the
            bounding box so edge damage isn't clipped.
        select_largest : bool
            If True and multiple vehicles are detected, build the mask
            from only the largest bounding box (default False).
            Useful for insurance claims where one vehicle is the subject.
    """

    DEFAULT_VEHICLE_CLASSES = [2, 5, 7]

    def __init__(self, config: dict) -> None:
        self.model_name: str = config.get("model_name", "yolo11m.pt")
        self.vehicle_classes: list[int] = config.get(
            "vehicle_classes", self.DEFAULT_VEHICLE_CLASSES
        )
        self.conf_threshold: float = config.get("confidence_threshold", 0.5)
        self.iou_threshold: float = config.get("iou_threshold", 0.45)
        self.mask_dilation_kernel: int = config.get("mask_dilation_kernel", 15)
        self.select_largest: bool = config.get("select_largest", False)

        self.model = None  # lazy-loaded

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def load_model(self) -> None:
        """Load the YOLOv8 model weights into memory.

        Separated from ``__init__`` so the pipeline controls when the
        (potentially large) model gets loaded.  On first call with the
        default ``"yolo11m.pt"``, ultralytics auto-downloads weights
        if they aren't cached.

        Raises
        ------
        ImportError
            If the ``ultralytics`` package is not installed.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for vehicle detection. "
                "Install with: pip install ultralytics"
            )

        logger.info("Loading YOLOv8 model: %s", self.model_name)
        self.model = YOLO(self.model_name)
        logger.info("Model loaded successfully.")

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run vehicle detection on a single image.

        Parameters
        ----------
        image : np.ndarray
            BGR image (any resolution — YOLO resizes internally).

        Returns
        -------
        DetectionResult
            Boxes, confidences, class IDs, and the binary vehicle mask.
            If no vehicles are found, all arrays are empty and the mask
            is all zeros — the caller decides how to handle this.
        """
        if self.model is None:
            self.load_model()

        # run inference — verbose=False suppresses ultralytics' print spam
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        # filter to vehicle classes only
        boxes, confidences, class_ids = self._filter_vehicle_detections(results)

        h, w = image.shape[:2]

        if len(boxes) == 0:
            logger.warning(
                "No vehicles detected (conf_threshold=%.2f). "
                "Downstream diff will have no mask constraint.",
                self.conf_threshold,
            )
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                confidences=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                vehicle_mask=np.zeros((h, w), dtype=np.uint8),
                num_vehicles=0,
                best_confidence=0.0,
            )

        # optionally select only the largest detection
        if self.select_largest and len(boxes) > 1:
            boxes, confidences, class_ids = self._select_largest(
                boxes, confidences, class_ids
            )

        # log what we found
        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, class_ids)):
            cls_name = COCO_VEHICLE_CLASSES.get(int(cls), f"class_{cls}")
            logger.debug(
                "Vehicle %d: %s conf=%.2f box=[%d,%d,%d,%d]",
                i, cls_name, conf, *box.astype(int),
            )

        # build binary mask from bounding boxes
        vehicle_mask = self._boxes_to_mask(boxes, (h, w))

        return DetectionResult(
            boxes=boxes,
            confidences=confidences,
            class_ids=class_ids,
            vehicle_mask=vehicle_mask,
            num_vehicles=len(boxes),
            best_confidence=float(confidences.max()),
        )

    def detect_pair(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> tuple[DetectionResult, DetectionResult, np.ndarray]:
        """Detect vehicles in both images and produce a merged mask.

        The merged mask is the union of both detections — ensures that
        if the vehicle shifts slightly between shots (common), the diff
        mask covers it in both frames.

        Parameters
        ----------
        before : np.ndarray
            "Before" image (BGR).
        after : np.ndarray
            "After" image (BGR).

        Returns
        -------
        tuple[DetectionResult, DetectionResult, np.ndarray]
            (before_detection, after_detection, merged_mask)
        """
        det_before = self.detect(before)
        det_after = self.detect(after)

        # union of both masks
        merged = cv2.bitwise_or(det_before.vehicle_mask, det_after.vehicle_mask)

        logger.debug(
            "Merged mask: before=%d vehicles, after=%d vehicles, "
            "union_area=%d px",
            det_before.num_vehicles,
            det_after.num_vehicles,
            int(cv2.countNonZero(merged)),
        )

        return det_before, det_after, merged

    # ══════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════

    def _filter_vehicle_detections(
        self, results
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter YOLO results to keep only vehicle classes.

        Parameters
        ----------
        results : list[ultralytics.engine.results.Results]
            Raw YOLO inference output (list of 1 Results object).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (boxes_xyxy, confidences, class_ids) — vehicles only.
        """
        result = results[0]  # single image → single Results object
        all_boxes = result.boxes

        if all_boxes is None or len(all_boxes) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        # pull tensors to numpy
        xyxy = all_boxes.xyxy.cpu().numpy()
        confs = all_boxes.conf.cpu().numpy()
        cls = all_boxes.cls.cpu().numpy().astype(int)

        # boolean mask: keep only rows whose class is in vehicle_classes
        keep = np.isin(cls, self.vehicle_classes)

        return (
            xyxy[keep].astype(np.float32),
            confs[keep].astype(np.float32),
            cls[keep].astype(np.int32),
        )

    def _select_largest(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keep only the detection with the largest bounding-box area.

        Useful for insurance claims where one vehicle is the subject
        and background vehicles should be ignored.

        Parameters
        ----------
        boxes : np.ndarray
            (N, 4) xyxy format.
        confidences : np.ndarray
            (N,)
        class_ids : np.ndarray
            (N,)

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Single-element arrays for the largest detection.
        """
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(areas.argmax())

        logger.debug(
            "select_largest: picked detection %d (area=%.0f px²) "
            "out of %d vehicles.",
            idx, areas[idx], len(boxes),
        )

        return (
            boxes[idx : idx + 1],
            confidences[idx : idx + 1],
            class_ids[idx : idx + 1],
        )

    def _boxes_to_mask(
        self, boxes: np.ndarray, image_shape: tuple[int, int]
    ) -> np.ndarray:
        """Convert bounding boxes to a filled, dilated binary mask.

        The dilation expands the mask slightly beyond the bounding box
        so that damage on the very edge of the car (bumpers, trim)
        isn't clipped.

        Parameters
        ----------
        boxes : np.ndarray
            Bounding boxes in xyxy format (N, 4).
        image_shape : tuple[int, int]
            (height, width) of the output mask.

        Returns
        -------
        np.ndarray
            Binary mask, dtype uint8, values {0, 255}.
        """
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            # clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

        # dilate to give margin around the detection box
        if self.mask_dilation_kernel > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.mask_dilation_kernel, self.mask_dilation_kernel),
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask