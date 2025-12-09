IAAV – Real-Time Hand-Gesture Detection and Interpretation

School project: a web application that compares a MediaPipe detector (client-side) with a YOLO detector (FastAPI backend) on webcam video.
A full report in French is included in the repository.

Prerequisites

Node.js 18+ and npm

Python 3.10+

YOLO hand-detection weights: hand-gesture-starter/backend/models/hand.pt (included in the repo) or provide a custom path via YOLO_MODEL_PATH.

Quick Start

Clone or open the hand-gesture-starter folder.

Start the backend (FastAPI + YOLO):

cd hand-gesture-starter
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8000


Adjust configuration through:

YOLO_MODEL_PATH (path to the .pt file)

YOLO_DEVICE (cpu, cuda:0, …)

Start the frontend (Vite/React) in another terminal:

cd hand-gesture-starter
npm install
npm run dev -- --host 0.0.0.0 --port 5173


If the backend is not running at http://localhost:8000, set VITE_API_URL (example:
VITE_API_URL=http://127.0.0.1:8000 npm run dev).

Open the browser at the URL shown by Vite (default: http://localhost:5173
).

What the App Does

Two mirrored video panels:
Left = YOLO backend, Right = MediaPipe HandLandmarker.

Overlays: bounding boxes + keypoints, FPS/status HUD.

Dashboard: hand count, confidence scores, fist/fingers/direction events (from MediaPipe).

YOLO Data & Training

Folder yolo_training/:

Script prepare_freihand_yolo.py converts FreiHAND v2 into YOLO format and launches fine-tuning (Ultralytics).

Latest training run logged in:
yolo_training/freihand_yolo/runs/freihand/ (contains results.csv and weights/best.pt).

The default model used by the API is hand-gesture-starter/backend/models/hand.pt.

Report

A detailed French report accompanies the project (school context) explaining goals, technologies, pipeline, evaluation, and future improvements.