#!/usr/bin/env python3
"""Assess vehicle damage from before/after images using pipeline detections.

This preserves all original reporting and visualization behavior, but
replaces the internal damage detection with the pipeline's masked diff + analyzer.

Outputs:
- Annotated images
- JSON report
- Human-readable terminal report
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
import cv2
import numpy as np

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ANSI colors for terminal
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    GRAY = "\033[90m"

SEVERITY_COLORS = {
    "none": C.GREEN,
    "minor": C.CYAN,
    "moderate": C.YELLOW,
    "severe": C.RED,
    "critical": C.MAGENTA,
}

# ------------------- Imports from pipeline -------------------
from src.alignment.aligner import ImageAligner
from src.detection.vehicle_detector import VehicleDetector
from src.comparison.diff_engine import DiffEngine
from src.segmentation.damage_analyzer import DamageAnalyzer

# ------------------- Paths -------------------
TESTS_DIR = PROJECT_ROOT / "tests"
BEFORE_IMG = TESTS_DIR / "images" / "car A - 1.png"
AFTER_IMG  = TESTS_DIR / "images" / "car A - 2.png"
OUTPUT_DIR = TESTS_DIR / "outputs" / "assess_damage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------- Helpers -------------------
def load_img(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"cv2.imread failed on {path}")
    return img

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def label(img, text, pos=(10,30), scale=0.6):
    out = img.copy()
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)
    return out

def save_img(name: str, img: np.ndarray):
    path = OUTPUT_DIR / name
    cv2.imwrite(str(path), img)
    print(f"Saved: {path.name}")

# ------------------- Main -------------------
def main():
    print("\nUsing pipeline for accurate damage detection...")

    # 1. Load images
    before = load_img(BEFORE_IMG)
    after = load_img(AFTER_IMG)
    if before.shape[:2] != after.shape[:2]:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))

    # 2. Align
    aligner = ImageAligner({"feature_method": "orb", "fallback": True})
    res_align = aligner.align(before, after)
    warped_after = res_align.warped_after

    # 3. Detect vehicle
    detector = VehicleDetector({"confidence_threshold": 0.3})
    vehicle_res = detector.detect(before)

    # 4. Diff with mask
    diff_engine = DiffEngine({"threshold": 30, "min_contour_area": 100})
    diff_res = diff_engine.compare(to_gray(before), to_gray(warped_after), vehicle_mask=vehicle_res.vehicle_mask)

    # 5. Damage analysis
    analyzer = DamageAnalyzer()
    analysis = analyzer.analyze(diff_res.contours, diff_res.raw_diff, vehicle_mask=vehicle_res.vehicle_mask)

    # ------------------- Visualization: colored damage regions (true shapes) -------------------
    final_viz = warped_after.copy()
    overlay = final_viz.copy()

    for reg in analysis.regions:
        color = (0, 255, 0) if reg.damage_type.value != "unknown" else (0, 255, 255)
        # Draw filled contour
        cv2.drawContours(overlay, [reg.contour], -1, color, thickness=cv2.FILLED)
        
        # Compute centroid from contour
        M = cv2.moments(reg.contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # fallback to bounding box center
            x, y, w, h = reg.bbox
            cx, cy = x + w // 2, y + h // 2

        cv2.putText(final_viz, f"{reg.damage_type.value} ({reg.severity_score:.2f})",
                    (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Blend overlay with original image
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, final_viz, 1 - alpha, 0, final_viz)

    # Save final analysis
    save_img("final_analysis.png", label(final_viz, f"OVERALL: {analysis.overall_severity.upper()}"))

    # ------------------- JSON report -------------------
    report = {
        "before_image": str(BEFORE_IMG),
        "after_image": str(AFTER_IMG),
        "overall_severity": analysis.overall_severity,
        "overall_severity_score": analysis.overall_severity_score,
        "damage_type_summary": analysis.damage_type_summary,
        "regions": [
            {
                "region_id": i+1,
                "bbox": reg.bbox,
                "damage_type": reg.damage_type.value,
                "severity_score": reg.severity_score
            } for i, reg in enumerate(analysis.regions)
        ]
    }
    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"JSON report saved: {report_path.name}")

    # ------------------- Terminal report -------------------
    print(f"\n{'═'*60}")
    print(f"  VEHICLE DAMAGE ASSESSMENT REPORT")
    print(f"{'═'*60}")
    print(f"Before: {BEFORE_IMG}")
    print(f"After:  {AFTER_IMG}")
    print(f"Overall Severity: {analysis.overall_severity.upper()} ({analysis.overall_severity_score:.2f})")
    print(f"Damage Summary: {analysis.damage_type_summary}")
    print(f"Number of Damage Regions: {len(analysis.regions)}\n")

    for i, reg in enumerate(analysis.regions, 1):
        print(f"Region #{i}: Type={reg.damage_type.value}, Severity={reg.severity_score:.3f}, BBox={reg.bbox}")

    print(f"\nOutputs saved under: {OUTPUT_DIR.resolve()}")
    print(f"{'═'*60}\n")

if __name__ == "__main__":
    main()
