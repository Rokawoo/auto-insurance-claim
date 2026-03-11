"""End-to-end damage assessment pipeline.

Orchestrates all stages:
  preprocessing → alignment → detection → comparison → analysis → report

No fine-tuned models required.  Vehicle detection uses pretrained YOLO11
(COCO weights).  Damage classification uses contour geometry heuristics
via DamageAnalyzer.

Graceful degradation:
  - If alignment fails → falls back to unaligned comparison (less accurate
    but still produces results).
  - If vehicle detection finds nothing → runs comparison on the full image
    with a warning.
  - Individual stage errors are caught and logged; the pipeline attempts to
    produce the best result it can with whatever stages succeed.
"""

from __future__ import annotations

import logging
import time
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
from src.segmentation import DamageAnalyzer
from src.segmentation.damage_analyzer import AnalysisResult
from src.utils.visualization import Visualizer
from src.utils.report import ReportGenerator

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════════════

@dataclass
class StageTimings:
    """Wall-clock time for each pipeline stage (seconds).

    Useful for profiling and for display in reports / presentations.
    """
    load: float = 0.0
    preprocess: float = 0.0
    alignment: float = 0.0
    detection: float = 0.0
    comparison: float = 0.0
    analysis: float = 0.0
    visualization: float = 0.0
    report: float = 0.0
    total: float = 0.0


@dataclass
class PipelineResult:
    """Full output of the damage assessment pipeline.

    Attributes
    ----------
    before_original : np.ndarray
        Original "before" image as loaded.
    after_original : np.ndarray
        Original "after" image as loaded.
    alignment : AlignmentResult | None
        Alignment stage output.  None if alignment failed and was skipped.
    detection : DetectionResult | None
        Vehicle detection output (boxes + mask).
    comparison : DiffResult | None
        Pixel differencing output (contours + damage candidates).
    analysis : AnalysisResult | None
        Heuristic damage classification and severity scoring.
    annotated_image : np.ndarray
        Final output image with damage regions drawn.
    summary_image : np.ndarray
        2×2 grid: before / after / diff heatmap / annotated.
    report : dict
        Structured JSON-serializable damage report.
    timings : StageTimings
        Per-stage wall-clock times.
    warnings : list[str]
        Any non-fatal issues encountered during the run.
    """
    before_original: np.ndarray = field(repr=False)
    after_original: np.ndarray = field(repr=False)
    alignment: AlignmentResult | None = None
    detection: DetectionResult | None = None
    comparison: DiffResult | None = None
    analysis: AnalysisResult | None = None
    annotated_image: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    summary_image: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    report: dict = field(default_factory=dict)
    timings: StageTimings = field(default_factory=StageTimings)
    warnings: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════

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
        self.analyzer = DamageAnalyzer(config.get("analysis", {}))
        self.visualizer = Visualizer(config.get("output", {}))
        self.report_gen = ReportGenerator()

        self._models_loaded = False

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def run(
        self,
        before_path: str | Path,
        after_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline on a before/after image pair.

        Parameters
        ----------
        before_path : str | Path
            Path to the "before" (undamaged) image.
        after_path : str | Path
            Path to the "after" (damaged) image.
        output_dir : str | Path | None
            If provided, save annotated images and JSON report here.

        Returns
        -------
        PipelineResult
            All intermediate and final results.

        Raises
        ------
        FileNotFoundError
            If either image path doesn't exist or can't be decoded.
        """
        before_path, after_path = Path(before_path), Path(after_path)
        timings = StageTimings()
        warnings: list[str] = []
        pipeline_start = time.perf_counter()

        logger.info(
            "Pipeline started: before=%s, after=%s",
            before_path.name, after_path.name,
        )

        # ── 1. load ────────────────────────────────────────────────
        t0 = time.perf_counter()
        before_orig = cv2.imread(str(before_path))
        after_orig = cv2.imread(str(after_path))

        if before_orig is None:
            raise FileNotFoundError(f"Could not load before image: {before_path}")
        if after_orig is None:
            raise FileNotFoundError(f"Could not load after image: {after_path}")

        timings.load = time.perf_counter() - t0
        logger.debug(
            "Loaded images: before=%s, after=%s (%.3fs)",
            before_orig.shape, after_orig.shape, timings.load,
        )

        # ── 2. preprocess ─────────────────────────────────────────
        t0 = time.perf_counter()
        before_proc, after_proc = self.preprocessor.process_pair(
            before_orig, after_orig
        )
        timings.preprocess = time.perf_counter() - t0
        logger.debug("Preprocessed to %s (%.3fs)", before_proc.shape, timings.preprocess)

        # ── 3. align ──────────────────────────────────────────────
        t0 = time.perf_counter()
        alignment: AlignmentResult | None = None
        aligned_after = after_proc  # fallback: use unaligned

        try:
            alignment = self.aligner.align(before_proc, after_proc)
            aligned_after = alignment.warped_after
            timings.alignment = time.perf_counter() - t0

            logger.info(
                "Alignment: method=%s, inliers=%d, reproj=%.2fpx, reliable=%s (%.3fs)",
                alignment.feature_method,
                alignment.num_inliers,
                alignment.reprojection_error,
                alignment.is_reliable,
                timings.alignment,
            )

            if not alignment.is_reliable:
                msg = (
                    f"Alignment quality below threshold "
                    f"(inliers={alignment.num_inliers}, "
                    f"reproj={alignment.reprojection_error:.2f}px). "
                    f"Diff results may include false positives."
                )
                warnings.append(msg)
                logger.warning(msg)

        except RuntimeError as exc:
            timings.alignment = time.perf_counter() - t0
            msg = f"Alignment failed: {exc}. Falling back to unaligned comparison."
            warnings.append(msg)
            logger.warning(msg)

        # ── 4. detect vehicle ─────────────────────────────────────
        t0 = time.perf_counter()
        if not self._models_loaded:
            self._load_models()

        detection = self.detector.detect(before_orig)
        timings.detection = time.perf_counter() - t0

        logger.info(
            "Detection: %d vehicle(s) found, best_conf=%.2f (%.3fs)",
            detection.num_vehicles,
            detection.best_confidence,
            timings.detection,
        )

        # resize vehicle mask to preprocessed dimensions
        proc_h, proc_w = before_proc.shape[:2]
        vehicle_mask = cv2.resize(
            detection.vehicle_mask, (proc_w, proc_h),
            interpolation=cv2.INTER_NEAREST,
        )

        # intersect with alignment valid_mask to exclude warp borders
        if alignment is not None and alignment.valid_mask is not None:
            valid_resized = cv2.resize(
                alignment.valid_mask, (proc_w, proc_h),
                interpolation=cv2.INTER_NEAREST,
            )
            vehicle_mask = cv2.bitwise_and(vehicle_mask, valid_resized)

        # handle empty detection
        if detection.num_vehicles == 0:
            msg = (
                "No vehicles detected — comparison will run on full image. "
                "Results may include significant background noise."
            )
            warnings.append(msg)
            logger.warning(msg)
            # use full-image mask so the pipeline still produces output
            vehicle_mask = np.full((proc_h, proc_w), 255, dtype=np.uint8)

        vehicle_area_px = int(np.count_nonzero(vehicle_mask))

        # ── 5. compare ────────────────────────────────────────────
        t0 = time.perf_counter()
        comparison = self.diff_engine.compare(
            before_proc, aligned_after, vehicle_mask
        )
        timings.comparison = time.perf_counter() - t0

        logger.info(
            "Comparison: %d contours, damage_area=%dpx (%.3fs)",
            len(comparison.contours),
            comparison.damage_area_px,
            timings.comparison,
        )

        # ── 6. analyze damage (heuristic) ─────────────────────────
        t0 = time.perf_counter()
        analysis = self.analyzer.analyze(
            contours=comparison.contours,
            diff_image=comparison.raw_diff,
            vehicle_mask=vehicle_mask,
        )
        timings.analysis = time.perf_counter() - t0

        logger.info(
            "Analysis: severity=%s (%.4f), %d regions, types=%s (%.3fs)",
            analysis.overall_severity,
            analysis.overall_severity_score,
            len(analysis.regions),
            analysis.damage_type_summary,
            timings.analysis,
        )

        # ── 7. visualize ──────────────────────────────────────────
        t0 = time.perf_counter()
        annotated = self.visualizer.draw_damage_overlay(
            aligned_after, analysis.regions
        )
        summary = self.visualizer.create_summary_image(
            before=before_proc,
            after=aligned_after,
            diff_mask=comparison.raw_diff,
            regions=analysis.regions,
            overall_severity=analysis.overall_severity,
            severity_score=analysis.overall_severity_score,
        )
        timings.visualization = time.perf_counter() - t0

        # ── 8. generate report ────────────────────────────────────
        t0 = time.perf_counter()
        alignment_reliable = alignment.is_reliable if alignment is not None else False
        report = self.report_gen.generate(
            analysis=analysis,
            vehicle_area_px=vehicle_area_px,
            image_shape=(proc_h, proc_w),
            before_path=str(before_path),
            after_path=str(after_path),
            alignment_reliable=alignment_reliable,
        )

        # inject timings and warnings into report
        report["timings"] = {
            k: round(v, 4) for k, v in timings.__dict__.items()
        }
        if warnings:
            report.setdefault("warnings", []).extend(warnings)

        timings.report = time.perf_counter() - t0
        timings.total = time.perf_counter() - pipeline_start

        logger.info(
            "Pipeline complete: severity=%s, total_time=%.2fs",
            analysis.overall_severity, timings.total,
        )

        result = PipelineResult(
            before_original=before_orig,
            after_original=after_orig,
            alignment=alignment,
            detection=detection,
            comparison=comparison,
            analysis=analysis,
            annotated_image=annotated,
            summary_image=summary,
            report=report,
            timings=timings,
            warnings=warnings,
        )

        # ── 9. save ───────────────────────────────────────────────
        if output_dir is not None:
            pair_id = before_path.stem
            self._save_outputs(result, Path(output_dir), pair_id)

        return result

    def run_from_arrays(
        self,
        before: np.ndarray,
        after: np.ndarray,
        before_label: str = "before",
        after_label: str = "after",
        output_dir: str | Path | None = None,
    ) -> PipelineResult:
        """Run the pipeline on in-memory numpy arrays.

        Convenience method for notebooks and testing where images are
        already loaded or generated synthetically.

        Parameters
        ----------
        before : np.ndarray
            BGR "before" image.
        after : np.ndarray
            BGR "after" image.
        before_label : str
            Label for the before image in the report (default "before").
        after_label : str
            Label for the after image in the report (default "after").
        output_dir : str | Path | None
            If provided, save outputs here.

        Returns
        -------
        PipelineResult
        """
        # write to temp files and delegate to run()
        # (simpler than duplicating the full pipeline logic)
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            before_path = Path(tmp) / f"{before_label}.png"
            after_path = Path(tmp) / f"{after_label}.png"
            cv2.imwrite(str(before_path), before)
            cv2.imwrite(str(after_path), after)
            return self.run(before_path, after_path, output_dir)

    def run_batch(
        self,
        pairs: list[tuple[str, str]],
        output_dir: str | Path | None = None,
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
        self._load_models()

        results: list[PipelineResult] = []
        total = len(pairs)

        for i, (before_path, after_path) in enumerate(pairs, 1):
            logger.info("Batch [%d/%d]: %s", i, total, Path(before_path).name)
            try:
                result = self.run(before_path, after_path, output_dir)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Batch [%d/%d] FAILED on %s / %s: %s",
                    i, total, before_path, after_path, exc,
                )
                continue

        logger.info(
            "Batch complete: %d/%d succeeded.", len(results), total
        )
        return results

    # ══════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════

    def _load_models(self) -> None:
        """Load the YOLO vehicle detection model (once)."""
        if not self._models_loaded:
            self.detector.load_model()
            self._models_loaded = True

    def _save_outputs(
        self,
        result: PipelineResult,
        output_dir: Path,
        pair_id: str,
    ) -> None:
        """Save annotated images, debug images, and JSON report to disk.

        Parameters
        ----------
        result : PipelineResult
        output_dir : Path
        pair_id : str
            Identifier for this image pair (used in filenames).
        """
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        report_dir = output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        fmt = self.visualizer.fmt

        # annotated overlay
        cv2.imwrite(
            str(viz_dir / f"{pair_id}_annotated.{fmt}"),
            result.annotated_image,
        )

        # 2×2 summary grid
        cv2.imwrite(
            str(viz_dir / f"{pair_id}_summary.{fmt}"),
            result.summary_image,
        )

        # raw diff heatmap (useful for debugging threshold tuning)
        if result.comparison is not None:
            heatmap = cv2.applyColorMap(result.comparison.raw_diff, cv2.COLORMAP_JET)
            cv2.imwrite(
                str(viz_dir / f"{pair_id}_diff_heatmap.{fmt}"),
                heatmap,
            )

        # alignment matches (useful for debugging alignment issues)
        if result.alignment is not None and result.alignment.matches:
            try:
                match_img = self.visualizer.draw_alignment_matches(
                    result.before_original,
                    result.after_original,
                    result.alignment.keypoints_before,
                    result.alignment.keypoints_after,
                    result.alignment.matches,
                )
                cv2.imwrite(
                    str(viz_dir / f"{pair_id}_alignment_matches.{fmt}"),
                    match_img,
                )
            except Exception:
                pass  # non-critical — don't fail the save

        # JSON report
        self.report_gen.save(
            result.report,
            report_dir / f"{pair_id}_report.json",
        )

        logger.info("Outputs saved to %s", output_dir)