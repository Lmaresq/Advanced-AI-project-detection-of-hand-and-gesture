# Hand Gesture Backend

Minimal API to run YOLO hand detection behind the `hand-gesture-starter` front end. The frontend README (../README.md) explains how it consumes this API; this file covers the backend specifics.

## Prerequisites
- Python 3.10+
- YOLO weights trained to detect hands (place them at `backend/models/hand.pt` or set `YOLO_MODEL_PATH`).

## Quick install
```bash
cd hand-gesture-starter
python -m venv .venv
.venv\\Scripts\\activate  # on Windows
pip install --upgrade pip
pip install -r backend/requirements.txt
```

## Run the API
```bash
uvicorn backend.app:app --reload --port 8000
```
Endpoints:
- `GET /health` checks that the model is loaded.
- `POST /detect` detects hands in an image sent as `multipart/form-data`.

## How it works
- Tech stack: FastAPI + Uvicorn, Ultralytics YOLO, OpenCV for image decode, NumPy.
- On startup, the model is loaded once (see `backend/detector.py`).
- `/detect` decodes the uploaded image, runs YOLO, filters hand classes, and returns normalized boxes (`x,y,w,h` on [0,1]), confidence, and class info. Keypoints are included if your YOLO model predicts them.
- CORS is enabled for `http://localhost:5173` to let the Vite app call the API.

Example call:
```bash
curl -X POST "http://localhost:8000/detect?min_conf=0.3" ^
  -F "file=@tests/sample.jpg" ^
  -H "Accept: application/json"
```

## Configuration
- `YOLO_MODEL_PATH`: path to the YOLO weight file (default `backend/models/hand.pt`).
- `YOLO_DEVICE`: `cpu`, `cuda:0`, etc. (leave empty for auto).
- `YOLO_HAND_CLASSES`: comma separated class names treated as hands. Leave empty to disable filtering and return all model classes.

Returned boxes (`x, y, w, h`) are normalized on [0,1] to match the current front format.
