"""FastAPI application exposing YOLO hand detection endpoints."""

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

from .detector import get_detector
from .schemas import DetectionResponse, HealthResponse

app = FastAPI(title="Hand Gesture Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def load_model() -> None:
    # Warm up the model once at startup to avoid first-request latency.
    get_detector()


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    detector = get_detector()
    return HealthResponse(
        model_path=str(detector.model_path),
        classes=detector.class_names,
    )


@app.post(
    "/detect",
    response_model=DetectionResponse,
    tags=["inference"],
    summary="Detect hands in an image",
)
async def detect(
    file: UploadFile = File(..., description="Image file (jpg/png)"),
    min_conf: float = 0.25,
    iou: float = 0.45,
) -> DetectionResponse:
    payload = await file.read()
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image payload."
        )

    np_img = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to decode image. Send a valid jpg/png via multipart/form-data.",
        )

    detections = get_detector().predict(image, conf=min_conf, iou=iou)
    return DetectionResponse(count=len(detections), detections=detections)
