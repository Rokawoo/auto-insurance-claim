# Damage Segmentation Module

- Uses YOLOv8-seg for scratch/dent/crack detection.
- Requires `models/pretrained/yolov8n-seg.pt`.
    - can be downloaded with download_models.py
- API:
    - segment(image)
    - segment_with_candidates(image, candidate_mask)
- Unit tests: ../../tests/test_segmentation.py