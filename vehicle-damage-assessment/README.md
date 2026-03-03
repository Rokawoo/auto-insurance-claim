# Vehicle Damage Assessment Pipeline

**Automated before/after vehicle damage detection using computer vision.**

## Problem

Insurance damage assessment is manual, subjective, and slow. This pipeline automates
detection and localization of visible vehicle damage by comparing "before" and "after"
images of the same vehicle.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE STAGES                              │
│                                                                     │
│  ┌──────────┐   ┌───────────┐   ┌───────────┐   ┌──────────────┐  │
│  │  PREPROC  │──▶│ ALIGNMENT │──▶│ DETECTION │──▶│  COMPARISON  │  │
│  │           │   │           │   │  (YOLOv8) │   │  (diff map)  │  │
│  └──────────┘   └───────────┘   └───────────┘   └──────┬───────┘  │
│                                                         │          │
│                                                         ▼          │
│                                                  ┌──────────────┐  │
│                                                  │ SEGMENTATION │  │
│                                                  │ (damage mask)│  │
│                                                  └──────┬───────┘  │
│                                                         │          │
│                                                         ▼          │
│                                                  ┌──────────────┐  │
│                                                  │   REPORT /   │  │
│                                                  │   OUTPUT     │  │
│                                                  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Stage Breakdown

1. **Preprocessing** — Normalize lighting, resize, grayscale, denoise (Gaussian blur)
2. **Alignment** — Feature matching (ORB/SIFT) + homography warp to spatially align before/after
3. **Detection** — YOLOv8 vehicle detection to create a vehicle-only ROI mask
4. **Comparison** — Pixel-wise diff within the vehicle mask, thresholded to find damage candidates
5. **Segmentation** — Optional YOLOv8-seg fine-tuned model to classify damage types (scratch, dent, crack)
6. **Report** — Annotated output images + structured JSON damage report

## Project Structure

```
vda/
├── src/
│   ├── preprocessing/    # image normalization, denoising, resizing
│   ├── alignment/        # feature matching, homography, image warping
│   ├── detection/        # YOLOv8 vehicle detection + ROI masking
│   ├── segmentation/     # damage type segmentation (fine-tuned YOLO-seg)
│   ├── comparison/       # pixel differencing, threshold, contour extraction
│   ├── pipeline/         # orchestration — ties all stages together
│   └── utils/            # I/O, visualization, logging, config helpers
├── configs/              # YAML config files for pipeline params
├── data/                 # raw images, processed pairs, annotations
├── models/               # pretrained weights + fine-tuned checkpoints
├── tests/                # unit + integration tests
├── notebooks/            # exploratory analysis, prototyping
├── scripts/              # CLI entry points (run pipeline, train, evaluate)
└── outputs/              # results: annotated images, JSON reports
```

## Quick Start

```bash
pip install -r requirements.txt

# run full pipeline on a before/after pair
python scripts/run_pipeline.py \
    --before data/raw/before/car_001.jpg \
    --after  data/raw/after/car_001.jpg \
    --config configs/default.yaml

# train damage segmentation model
python scripts/train_damage_model.py --config configs/training.yaml

# evaluate on test set
python scripts/evaluate.py --config configs/eval.yaml
```

## Team

- Hitanshi, Muhammad — Problem definition & research
- Jeremy — Feasibility analysis
- Augustus, Loren — Technical approach & implementation plan
