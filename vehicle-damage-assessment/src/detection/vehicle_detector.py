"""YOLOv11-based vehicle instance segmentation and ROI mask generation.

This is the critical gatekeeper between alignment and comparison.
Produces a precise polygonal mask using instance segmentation so the 
diff engine only looks at the car's actual surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO class ID reference (the ones we care about):
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
    """Container for vehicle detection outputs."""
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
    """Detects vehicles and produces a precise binary ROI mask.

    Uses a YOLOv11 segmentation model to find cars, trucks, and buses. 
    The polygon segments are converted into a dilated binary mask.
    """

    DEFAULT_VEHICLE_CLASSES = [2, 5, 7]

    def __init__(self, config: dict) -> None:
        # Changed default to a segmentation model
        self.model_name: str = config.get("model_name", "yolo11m-seg.pt")
        self.vehicle_classes: list[int] = config.get(
            "vehicle_classes", self.DEFAULT_VEHICLE_CLASSES
        )
        self.conf_threshold: float = config.get("confidence_threshold", 0.5)
        self.iou_threshold: float = config.get("iou_threshold", 0.45)
        self.mask_dilation_kernel: int = config.get("mask_dilation_kernel", 15)
        self.select_largest: bool = config.get("select_largest", False)

        self.model = None

    def load_model(self) -> None:
        """Load the YOLOv11 segmentation model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics is required. Install with: pip install ultralytics")

        logger.info("Loading YOLO model: %s", self.model_name)
        self.model = YOLO(self.model_name)

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run vehicle segmentation on a single image."""
        if self.model is None:
            self.load_model()

        h, w = image.shape[:2]

        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )[0]  # Take first result object

        # 1. Filter raw detections to vehicle classes only
        indices = self._get_vehicle_indices(results)

        if len(indices) == 0:
            return self._empty_result(h, w)

        # 2. Extract filtered boxes/confs
        boxes = results.boxes.xyxy.cpu().numpy()[indices]
        confidences = results.boxes.conf.cpu().numpy()[indices]
        class_ids = results.boxes.cls.cpu().numpy().astype(int)[indices]

        # 3. Handle selection of largest vehicle
        if self.select_largest and len(boxes) > 1:
            indices, boxes, confidences, class_ids = self._filter_to_largest(
                indices, boxes, confidences, class_ids
            )

        # 4. Generate the Mask (Polygon if available, fallback to Box)
        if hasattr(results, 'masks') and results.masks is not None:
            # Use polygonal segments
            vehicle_mask = self._polygons_to_mask(results.masks.data[indices], (h, w))
        else:
            # Fallback to bounding boxes if using a non-segmentation model
            logger.warning("No segmentation masks found. Falling back to bounding box mask.")
            vehicle_mask = self._boxes_to_mask(boxes, (h, w))

        return DetectionResult(
            boxes=boxes,
            confidences=confidences,
            class_ids=class_ids,
            vehicle_mask=vehicle_mask,
            num_vehicles=len(boxes),
            best_confidence=float(confidences.max()),
        )

    def detect_pair(self, before: np.ndarray, after: np.ndarray):
        """Standard detect_pair remains identical to maintain pipeline compatibility."""
        det_before = self.detect(before)
        det_after = self.detect(after)
        merged = cv2.bitwise_or(det_before.vehicle_mask, det_after.vehicle_mask)
        return det_before, det_after, merged

    # ══════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════

    def _get_vehicle_indices(self, results) -> list[int]:
        """Find indices of detections that match vehicle classes."""
        if results.boxes is None:
            return []
        cls = results.boxes.cls.cpu().numpy().astype(int)
        return [i for i, c in enumerate(cls) if c in self.vehicle_classes]

    def _filter_to_largest(self, indices, boxes, confs, cls):
        """Keep only the index/data for the largest bbox area."""
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best_idx = areas.argmax()
        return [indices[best_idx]], boxes[best_idx:best_idx+1], confs[best_idx:best_idx+1], cls[best_idx:best_idx+1]

    def _polygons_to_mask(self, mask_tensors, shape) -> np.ndarray:
        """Converts YOLO segmentation tensors to a single dilated binary mask."""
        h, w = shape
        # Merge multiple masks into one using 'any' (logical OR)
        if len(mask_tensors) > 0:
            # Resize internal YOLO masks to image size
            combined = np.any(mask_tensors.cpu().numpy(), axis=0).astype(np.uint8) * 255
            full_mask = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            full_mask = np.zeros((h, w), dtype=np.uint8)

        return self._dilate_mask(full_mask)

    def _boxes_to_mask(self, boxes, shape) -> np.ndarray:
        """Standard box mask as fallback."""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(mask, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)
        return self._dilate_mask(mask)

    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply dilation if configured."""
        if self.mask_dilation_kernel > 0:
            k = self.mask_dilation_kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            return cv2.dilate(mask, kernel, iterations=1)
        return mask

    def _empty_result(self, h, w) -> DetectionResult:
        return DetectionResult(
            boxes=np.empty((0, 4), dtype=np.float32),
            confidences=np.empty((0,), dtype=np.float32),
            class_ids=np.empty((0,), dtype=np.int32),
            vehicle_mask=np.zeros((h, w), dtype=np.uint8),
        )