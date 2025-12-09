# Hand Gesture Starter (frontend)

This README covers the frontend: how it runs, how it talks to the backend, and how the two detection models are displayed. For backend details, see `backend/README.md`. The two READMEs together explain how to launch the whole project.

## What you get
- Side-by-side comparison UI: left = your YOLO backend, right = Google/MediaPipe (Tasks Vision) running in-browser.
- Overlays with boxes/landmarks plus small info blocks (hands, fingers, scores) and a global summary comparing both models.

## Quick start
1) Backend: follow `backend/README.md` (needs a YOLO hand weight at `backend/models/hand.pt` or `YOLO_MODEL_PATH`) then run  
   `uvicorn backend.app:app --reload --port 8000`
2) Frontend:  
   ```
   npm install
   npm run dev
   ```  
   Open `http://localhost:5173/`.

## How the frontend works
- Two video panels (`src/App.jsx`):
  - **My model (YOLO backend)**: `useBackendDetections` grabs frames, posts them to `POST /detect` (FastAPI), and draws violet overlays.
  - **Google model (MediaPipe)**: `useDetections` calls the MediaPipe Tasks Vision hand landmarker in-browser and draws cyan/red overlays plus finger counts and fist detection.
- Event console lists recent stable gestures from the MediaPipe side.
- Camera feed is mirrored for a natural webcam feel.

## API calls and configuration
- Backend endpoint: `POST {VITE_API_URL or http://localhost:8000}/detect?min_conf=0.25` with the current frame as `multipart/form-data`.
- Set `VITE_API_URL` in a `.env` if the API is not on `localhost:8000`.
- CORS is enabled in the backend for `http://localhost:5173`.

## Tech stack (frontend)
- React 19 + Vite 7.
- MediaPipe Tasks Vision for the in-browser Google model.
- Fetch-based calls to the FastAPI + Ultralytics YOLO backend for your model.

## Build/preview
- `npm run build` to produce `dist/`.
- `npm run preview` to serve the built app locally.
