# FreiHAND → YOLO hand detector

This folder contains a small utility to turn the FreiHAND v2 dataset into a YOLO-format hand detector dataset and optionally launch training.

## Prerequisites
- `FreiHAND_pub_v2.zip` placed in `yolo_training/` (already present).
- Python 3.10+ with `ultralytics`, `numpy` (they are already in `backend/requirements.txt`; otherwise `pip install ultralytics numpy`).

## Quick start
```bash
cd yolo_training
python prepare_freihand_yolo.py \
  --zip-path FreiHAND_pub_v2.zip \
  --workdir freihand_raw \
  --out-dir freihand_yolo \
  --epochs 50 \
  --img-size 640
```

What the script does:
1. Extracts the training split and annotations from the FreiHAND zip into `freihand_raw/` (skips if already extracted).
2. Projects the 3D keypoints with the provided intrinsics to get 2D boxes, writes YOLO labels, and splits train/val (default 90/10).
3. Generates a `dataset.yaml` under `freihand_yolo/` and launches a YOLO fine-tune (default `yolo11n.pt`). Use `--skip-train` to only prepare the data.

## Useful flags
- `--val-split 0.1` ratio for validation split.
- `--max-samples 500` to create a tiny debug subset.
- `--model yolo11s.pt` to pick another base weight.
- `--device cpu` or `--device cuda:0` to choose hardware.
- `--skip-train` if you only want the prepared dataset.

## Outputs
- `freihand_yolo/images/{train,val}` and `labels/{train,val}` with YOLO boxes (one class: hand).
- `freihand_yolo/dataset.yaml` ready for Ultralytics.
- YOLO runs under `freihand_yolo/runs/<run-name>` by default.
