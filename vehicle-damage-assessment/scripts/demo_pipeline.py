#!/usr/bin/env python3
"""Run the full damage detection pipeline on real car images.

Align → Detect Vehicle → Mask → Diff → Heuristic Analysis → Visualize

This demo uses paths relative to the project root and compares results
with and without vehicle masking to demonstrate noise reduction.

Usage:
    python scripts/demo_pipeline.py
    python scripts/demo_pipeline.py --method sift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Resolve project root (vda/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.alignment.aligner import ImageAligner
from src.detection.vehicle_detector import VehicleDetector
from src.comparison.diff_engine import DiffEngine
from src.segmentation.damage_analyzer import DamageAnalyzer  # Ensure this path matches your project


# ─── defaults ───────────────────────────────────────────────────

DEFAULT_BEFORE = TESTS_DIR / "images" / "car A - 1.png"
DEFAULT_AFTER  = TESTS_DIR / "images" / "car A - 2.png"
DEFAULT_outputs = TESTS_DIR / "outputs" / "pipeline_demo"


# ─── helpers ────────────────────────────────────────────────────

def load_or_die(path: Path) -> np.ndarray:
    if not path.exists():
        print(f"\n  ERROR: {path} not found")
        sys.exit(1)
    img = cv2.imread(str(path))
    if img is None:
        print(f"\n  ERROR: cv2.imread failed on {path}")
        sys.exit(1)
    return img


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def label(img, text, pos=(10, 30), scale=0.6):
    out = to_bgr(img).copy()
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def save(path: Path, img: np.ndarray):
    cv2.imwrite(str(path), img)
    print(f"    -> {path.name}")


# ─── main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full VDA Pipeline Demo")
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--after", type=Path, default=DEFAULT_AFTER)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_outputs)
    parser.add_argument("--method", default="orb", choices=["orb", "sift", "akaze"])
    args = parser.parse_args()

    out_dir = args.outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  FULL VDA PIPELINE: Align → Detect → Mask → Diff → Analyze")
    print("=" * 65)

    # 1. LOAD
    print(f"\n[1/7] Loading images...")
    before_bgr = load_or_die(args.before)
    after_bgr = load_or_die(args.after)
    
    if before_bgr.shape[:2] != after_bgr.shape[:2]:
        after_bgr = cv2.resize(after_bgr, (before_bgr.shape[1], before_bgr.shape[0]))

    save(out_dir / "01_inputs.png", np.hstack([
        label(before_bgr, "BEFORE"), label(after_bgr, "AFTER")
    ]))

    # 2. ALIGN
    print(f"\n[2/7] Aligning ({args.method.upper()})...")
    aligner = ImageAligner({"feature_method": args.method, "fallback": True})
    res_align = aligner.align(before_bgr, after_bgr)
    warped_bgr = res_align.warped_after
    save(out_dir / "02_warped.png", label(warped_bgr, "ALIGNED AFTER"))

    # 3. DETECT VEHICLE
    print(f"\n[3/7] Detecting Vehicle (YOLOv11)...")
    detector = VehicleDetector({"confidence_threshold": 0.3})
    res_det = detector.detect(before_bgr)
    
    # Visualization: Mask Overlay
    mask_viz = before_bgr.copy()
    mask_layer = np.zeros_like(mask_viz)
    mask_layer[:, :, 2] = res_det.vehicle_mask # Red mask
    mask_overlay = cv2.addWeighted(mask_viz, 0.7, mask_layer, 0.3, 0)
    save(out_dir / "03_vehicle_mask.png", label(mask_overlay, "VEHICLE MASK OVERLAY"))

    # 4. DIFF (UNMASKED)
    print(f"\n[4/7] Diff WITHOUT mask (baseline noise)...")
    diff_engine = DiffEngine({"threshold": 30, "min_contour_area": 100})
    res_unmasked = diff_engine.compare(to_gray(before_bgr), to_gray(warped_bgr), vehicle_mask=None)
    
    unmasked_viz = warped_bgr.copy()
    cv2.drawContours(unmasked_viz, res_unmasked.contours, -1, (0, 0, 255), 2)
    save(out_dir / "04_diff_unmasked.png", label(unmasked_viz, f"UNMASKED: {len(res_unmasked.contours)} regions"))

    # 5. DIFF (MASKED)
    print(f"\n[5/7] Diff WITH mask (clean detection)...")
    res_masked = diff_engine.compare(to_gray(before_bgr), to_gray(warped_bgr), vehicle_mask=res_det.vehicle_mask)
    
    masked_viz = warped_bgr.copy()
    cv2.drawContours(masked_viz, res_masked.contours, -1, (0, 255, 0), 2)
    save(out_dir / "05_diff_masked.png", label(masked_viz, f"MASKED: {len(res_masked.contours)} regions"))

    # 6. DAMAGE ANALYSIS
    print(f"\n[6/7] Heuristic Damage Classification...")
    analyzer = DamageAnalyzer()
    analysis = analyzer.analyze(res_masked.contours, res_masked.raw_diff, vehicle_mask=res_det.vehicle_mask)
    
    # Final Visual with labelsoutputs
    final_viz = warped_bgr.copy()
    for reg in analysis.regions:
        x, y, w, h = reg.bbox
        color = (0, 255, 0) if reg.damage_type != "unknown" else (0, 255, 255)
        cv2.rectangle(final_viz, (x, y), (x + w, y + h), color, 2)
        cv2.putText(final_viz, f"{reg.damage_type.value} ({reg.severity_score:.2f})", 
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    save(out_dir / "06_analysis_final.png", label(final_viz, f"SEVERITY: {analysis.overall_severity.upper()}"))

    # 7. SUMMARY
    print(f"\n[7/7] Generating Summary...")
    tw, th = 400, 300
    def t(img, txt): return cv2.resize(label(img, txt, scale=0.4), (tw, th))
    
    summary = np.vstack([
        np.hstack([t(before_bgr, "Before"), t(warped_bgr, "Aligned"), t(mask_overlay, "Mask")]),
        np.hstack([t(unmasked_viz, "Naive Diff"), t(masked_viz, "Masked Diff"), t(final_viz, "Analysis")])
    ])
    save(out_dir / "07_summary_grid.png", summary)

    print(f"\n{'=' * 65}")
    print(f"  DEMO COMPLETE")
    print(f"  outputs: {out_dir.resolve()}")
    print(f"  Overall Severity: {analysis.overall_severity_score} ({analysis.overall_severity})")
    print(f"  Damage Summary: {analysis.damage_type_summary}")
    print("=" * 65)


if __name__ == "__main__":
    main()