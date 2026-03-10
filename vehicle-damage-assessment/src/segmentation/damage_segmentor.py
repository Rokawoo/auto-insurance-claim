"""Fine-tuned YOLOv8 segmentation model for damage classification.

This module is OPTIONAL and only activates when a fine-tuned model
checkpoint exists.  It takes the candidate damage regions identified
by the comparison stage and classifies each into a damage type
(scratch, dent, crack, shatter, deformation) with per-pixel masks.

Two modes of operation:
  1. **Direct inference** — run the seg model on the full "after" image
     and merge with the diff-based candidates for higher confidence.
  2. **ROI-only inference** — crop to each candidate bounding box and
     classify just that patch (faster, but may lose context).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class DamageInstance:
    """A single detected damage region with type and mask.

    Attributes
    ----------
    damage_type : str
        Predicted damage class name (e.g. "scratch", "dent").
    confidence : float
        Model confidence for this prediction.
    mask : np.ndarray
        Per-pixel binary mask for this damage instance (H, W).
    bbox : tuple[int, int, int, int]
        Bounding box (x1, y1, x2, y2) in image coordinates.
    area_px : int
        Number of positive pixels in the mask.
    """
    damage_type: str
    confidence: float
    mask: np.ndarray = field(repr=False)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    area_px: int = 0


@dataclass
class SegmentationResult:
    """Container for segmentation outputs.

    Attributes
    ----------
    instances : list[DamageInstance]
        All detected damage instances with masks and types.
    combined_mask : np.ndarray
        Single mask with all damage regions merged, dtype uint8.
    class_map : np.ndarray
        Per-pixel class ID map (0 = background, 1+ = damage classes).
    """
    instances: list[DamageInstance] = field(default_factory=list)
    combined_mask: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    class_map: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)


class DamageSegmentor:
    """Classifies and segments damage types using a fine-tuned YOLOv8-seg.

    Parameters
    ----------
    config : dict
        The ``segmentation`` section of the pipeline config.
    """

    def __init__(self, config: dict) -> None:
        self.enabled: bool = config.get("enabled", False)
        self.model_path: str = config.get("model_path", "")
        self.damage_classes: dict[int, str] = config.get("damage_classes", {})
        self.conf_threshold: float = config.get("confidence_threshold", 0.4)

        self.model = None

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check whether a fine-tuned model exists on disk.

        Returns
        -------
        bool
            True if the model file exists and segmentation is enabled.
        """
        return self.enabled and Path(self.model_path).exists()

    def load_model(self) -> None:
        """Load the fine-tuned YOLOv8-seg checkpoint.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        """
        if not Path(self.model_path).exists():
            raise FileNotFoundError(self.model_path)

        from ultralytics import YOLO

        self.model = YOLO(self.model_path)

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Run damage segmentation on a single image.

        Parameters
        ----------
        image : np.ndarray
            BGR "after" image (full resolution).

        Returns
        -------
        SegmentationResult
            Per-instance masks, types, and a combined class map.
        """
        if self.model is None:
            self.load_model()

        results = self.model(image, conf=0.1)

        instances = self._parse_yolo_seg_results(results)

        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for inst in instances:
            combined_mask = cv2.bitwise_or(combined_mask, inst.mask.astype(np.uint8))

        class_map = self._build_class_map(instances, (h, w))

        return SegmentationResult(
            instances=instances,
            combined_mask=combined_mask,
            class_map=class_map,
        )

    def segment_with_candidates(
        self,
        image: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> SegmentationResult:
        """Run segmentation but only keep predictions overlapping candidates.

        Merges the diff-based candidate mask with the model's predictions
        for higher precision — a region must be flagged by *both* the
        pixel-diff pipeline and the learned model to count as damage.

        Parameters
        ----------
        image : np.ndarray
            BGR "after" image.
        candidate_mask : np.ndarray
            Binary mask from the DiffEngine comparison stage.

        Returns
        -------
        SegmentationResult
            Filtered results where each instance overlaps the candidate mask.
        """
        base_result = self.segment(image)

        filtered = []

        for inst in base_result.instances:

            overlap = cv2.bitwise_and(inst.mask.astype(np.uint8), candidate_mask)

            if np.count_nonzero(overlap) > 0:
                filtered.append(inst)

        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for inst in filtered:
            combined_mask = cv2.bitwise_or(combined_mask, inst.mask.astype(np.uint8))

        class_map = self._build_class_map(filtered, (h, w))

        return SegmentationResult(
            instances=filtered,
            combined_mask=combined_mask,
            class_map=class_map,
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _parse_yolo_seg_results(self, results) -> list[DamageInstance]:
        """Convert raw YOLO segmentation results to DamageInstance list.

        Parameters
        ----------
        results : ultralytics.engine.results.Results
            Raw inference output.

        Returns
        -------
        list[DamageInstance]
        """
        instances = []

        if len(results) == 0:
            return instances

        r = results[0]

        if r.masks is None:
            return instances

        masks = r.masks.data.cpu().numpy()
        boxes = r.boxes

        for i in range(len(masks)):
            conf = float(boxes.conf[i])

            if conf < self.conf_threshold:
                continue

            class_id = int(boxes.cls[i])
            damage_type = self.damage_classes.get(class_id, str(class_id))

            mask = masks[i].astype(np.uint8)
            if mask.shape != r.orig_shape[:2]:
                mask = cv2.resize(mask, (r.orig_shape[1], r.orig_shape[0]))

            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

            area = int(np.count_nonzero(mask))

            inst = DamageInstance(
                damage_type=damage_type,
                confidence=conf,
                mask=mask,
                bbox=(x1, y1, x2, y2),
                area_px=area,
            )

            instances.append(inst)

        return instances

    def _build_class_map(
        self, instances: list[DamageInstance], image_shape: tuple[int, int]
    ) -> np.ndarray:
        """Build a per-pixel class ID map from instance masks.

        Parameters
        ----------
        instances : list[DamageInstance]
        image_shape : tuple[int, int]
            (height, width)

        Returns
        -------
        np.ndarray
            Class map where 0 = background, positive ints = class IDs.
        """
        h, w = image_shape

        class_map = np.zeros((h, w), dtype=np.uint8)

        for inst in instances:

            class_id = list(self.damage_classes.keys())[
                list(self.damage_classes.values()).index(inst.damage_type)
            ]

            class_map[inst.mask.astype(bool)] = class_id + 1

        return class_map
