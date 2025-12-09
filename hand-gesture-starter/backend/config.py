"""
Configuration values for the hand gesture backend.

Values can be overridden with environment variables to point to a custom
YOLO weight file or a different device (e.g. "cpu", "cuda:0").
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "hand.pt"

# Path to the YOLO weight file used for inference.
MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", DEFAULT_MODEL_PATH))

# Device is optional; let Ultralytics pick the best by default.
YOLO_DEVICE = os.getenv("YOLO_DEVICE") or None

# Comma separated list of class names that are considered "hands".
# If YOLO_HAND_CLASSES is empty or unset, no filtering is applied (all classes returned).
_hand_classes_raw = os.getenv("YOLO_HAND_CLASSES")
if _hand_classes_raw is None:
    HAND_CLASSES = None
else:
    parsed = {cls.strip().lower() for cls in _hand_classes_raw.split(",") if cls.strip()}
    HAND_CLASSES = parsed or None
