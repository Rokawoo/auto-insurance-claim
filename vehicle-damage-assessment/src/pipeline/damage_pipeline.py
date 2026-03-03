"""End-to-end damage assessment pipeline.

Orchestrates all stages: preprocessing → alignment → detection →
comparison → (optional) segmentation → report generation.

This is the main entry point that scripts and notebooks should use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing import Preprocessor
from src.alignment import ImageAligner
from src.alignment.aligner import AlignmentResult
from src.detection import VehicleDetector
from src.detection.vehicle_detector import DetectionResult
from src.comparison import DiffEngine
from src.comparison.diff_engine import DiffResult
from src.segmentation import DamageSegmentor
from src.segmentation.damage_segmentor import SegmentationResult
from src.utils.visualization import Visualizer
from src.utils.report import ReportGenerator


@dataclass
class PipelineResult:
    """Full output of the damage assessment pipeline.

    Attributes
    ----------
    before_original : np.ndarray
        Original "before" image as loaded.
    after_original : np.ndarray
        Original "after" image as loaded.
    alignment : AlignmentResult
        Alignment stage output (warped after + homography info).
    detection : DetectionResult
        Vehicle detection output (boxes + mask).
    comparison : DiffResult
        Pixel differencing output (contours + damage candidates).
    segmentation : SegmentationResult | None
        Damage type segmentation (if enabled and model available).
    annotated_image : np.ndarray
        Final output image with damage regions drawn.
    report : dict
        Structured damage report (JSON-serializable).
    """
    before_original: np.ndarray = field(repr=False)
    after_original: np.ndarray = field(repr=False)
    alignment: AlignmentResult | None = None
    detection: DetectionResult | None = None
    comparison: DiffResult | None = None
    segmentation: SegmentationResult | None = None
    annotated_image: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    report: dict = field(default_factory=dict)


class DamagePipeline:
    """Orchestrates the full before/after damage assessment.

    Parameters
    ----------
    config : dict
        Full pipeline config (all sections).
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        # instantiate all stage modules
        self.preprocessor = Preprocessor(config.get("preprocessing", {}))
        self.aligner = ImageAligner(config.get("alignment", {}))
        self.detector = VehicleDetector(config.get("detection", {}))
        self.diff_engine = DiffEngine(config.get("comparison", {}))
        self.segmentor = DamageSegmentor(config.get("segmentation", {}))
        self.visualizer = Visualizer(config.get("output", {}))
        self.report_gen = ReportGenerator()

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def run(
        self, before_path: str | Path, after_path: str | Path
    ) -> PipelineResult:
        """Execute the full pipeline on a before/after image pair.

        Parameters
        ----------
        before_path : str | Path
            Path to the "before" (undamaged) image.
        after_path : str | Path
            Path to the "after" (damaged) image.

        Returns
        -------
        PipelineResult
            All intermediate and final results.
        """
        # TODO:
        #   1. load images with cv2.imread
        #   2. preprocess pair
        #   3. align (warp after to before)
        #   4. detect vehicle in the before image → get mask
        #   5. compare preprocessed before vs warped after within mask
        #   6. if segmentor is available, run damage segmentation
        #   7. generate annotated image
        #   8. generate structured report
        #   9. save outputs if configured
        #  10. return PipelineResult
        raise NotImplementedError

    def run_batch(
        self, pairs: list[tuple[str, str]], output_dir: str | Path | None = None
    ) -> list[PipelineResult]:
        """Run the pipeline on multiple before/after pairs.

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            List of (before_path, after_path) tuples.
        output_dir : str | Path | None
            Optional directory to save all outputs.

        Returns
        -------
        list[PipelineResult]
        """
        # TODO:
        #   1. load models once (detector, segmentor)
        #   2. iterate pairs, call self.run for each
        #   3. collect and return results
        raise NotImplementedError

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """Eagerly load all ML models into memory.

        Called once before batch processing to avoid reloading per image.
        """
        # TODO:
        #   self.detector.load_model()
        #   if self.segmentor.is_available():
        #       self.segmentor.load_model()
        raise NotImplementedError

    def _save_outputs(
        self, result: PipelineResult, output_dir: Path, pair_id: str
    ) -> None:
        """Save annotated images and JSON report to disk.

        Parameters
        ----------
        result : PipelineResult
        output_dir : Path
        pair_id : str
            Identifier for this image pair (used in filenames).
        """
        # TODO:
        #   1. save annotated image as PNG
        #   2. save report as JSON
        #   3. optionally save intermediate visualizations
        raise NotImplementedError
