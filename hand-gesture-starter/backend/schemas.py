"""Pydantic schemas describing the API surface."""

from typing import List, Optional

from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    x: float = Field(..., description="Normalized left position (0-1)")
    y: float = Field(..., description="Normalized top position (0-1)")
    w: float = Field(..., description="Normalized width (0-1)")
    h: float = Field(..., description="Normalized height (0-1)")
    score: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_id: int = Field(..., description="Class index from the YOLO model")
    label: str = Field(..., description="Class label from the YOLO model")
    source: str = Field("yolo", description="Origin of the detection")
    landmarks: Optional[List[List[float]]] = Field(
        None,
        description="Optional keypoints normalized to 0-1 (x, y). Present if the model is a keypoint/pose model.",
    )


class DetectionResponse(BaseModel):
    count: int
    detections: List[DetectionBox]


class HealthResponse(BaseModel):
    status: str = "ok"
    model_path: str
    classes: List[str]
