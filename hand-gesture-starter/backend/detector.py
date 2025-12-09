"""YOLO inference wrapper dedicated to hand detection."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from ultralytics import YOLO

from .config import HAND_CLASSES, MODEL_PATH, YOLO_DEVICE


class DetectorNotReady(RuntimeError):
    """Raised when inference is attempted before the model is loaded."""


class YOLOHandDetector:
    """Light wrapper around Ultralytics YOLO to normalize outputs for the front-end."""

    def __init__(
        self,
        model_path: Path,
        allowed_classes: Optional[Iterable[str]] = None,
        device: Optional[str] = None,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"YOLO weight file not found at {path}. "
                "Drop your hand detector weights there or set YOLO_MODEL_PATH."
            )

        self.model_path = path
        self.allowed_classes = (
            {c.lower() for c in allowed_classes} if allowed_classes else None
        )
        self.model = YOLO(str(path))
        if device:
            self.model.to(device)

    @property
    def class_names(self) -> List[str]:
        names = self.model.model.names
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys())]
        return list(names)

    def _is_hand(self, label: str) -> bool:
        if not self.allowed_classes:
            return True
        return label.lower() in self.allowed_classes

    def predict(
        self, image: np.ndarray, conf: float = 0.25, iou: float = 0.45
    ) -> List[dict]:
        if self.model is None:
            raise DetectorNotReady("Model is not initialised yet.")

        results = self.model.predict(image, conf=conf, iou=iou, verbose=False)
        detections: List[dict] = []

        for result in results:
            height, width = result.orig_shape
            # If the model is a pose model, keypoints will be available.
            keypoints = (
                result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
            )

            for idx, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                label = (
                    result.names.get(cls_id, str(cls_id))
                    if isinstance(result.names, dict)
                    else str(cls_id)
                )
                if not self._is_hand(label):
                    continue

                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                det = {
                    "x": x1 / width,
                    "y": y1 / height,
                    "w": (x2 - x1) / width,
                    "h": (y2 - y1) / height,
                    "score": float(box.conf[0]),
                    "class_id": cls_id,
                    "label": label,
                    "source": "yolo",
                }

                if keypoints is not None and idx < keypoints.shape[0]:
                    pts = keypoints[idx]
                    det["landmarks"] = [
                        [float(px) / width, float(py) / height] for px, py in pts
                    ]

                detections.append(det)

        return detections


_detector: Optional[YOLOHandDetector] = None
_lock = threading.Lock()


def get_detector() -> YOLOHandDetector:
    global _detector
    if _detector is not None:
        return _detector

    with _lock:
        if _detector is None:
            _detector = YOLOHandDetector(
                model_path=MODEL_PATH,
                allowed_classes=HAND_CLASSES,
                device=YOLO_DEVICE,
            )
    return _detector
