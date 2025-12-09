import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

let landmarker = null;

// Compte les doigts levés
function countFingers(landmarks) {
  const fingerTips = [8, 12, 16, 20];
  const fingerPips = [6, 10, 14, 18];
  let count = 0;
  for (let i = 0; i < 4; i++) {
    if (landmarks[fingerTips[i]].y < landmarks[fingerPips[i]].y) count++;
  }
  // Détection du pouce (horizontal)
  if (landmarks[4].x > landmarks[3].x) count++;
  return count;
}

// Détecte si la main est fermée
function isFist(landmarks) {
  const wrist = landmarks[0];
  const tips = [4, 8, 12, 16, 20];
  const avgDist = tips
    .map(i => Math.hypot(
      landmarks[i].x - wrist.x,
      landmarks[i].y - wrist.y
    ))
    .reduce((a, b) => a + b, 0) / tips.length;
  return avgDist < 0.1;
}

// Détecte la direction du mouvement
let lastCenter = null;
let lastTime = performance.now();

function getHandDirection(landmarks) {
  const center = {
    x: landmarks.map(p => p.x).reduce((a,b) => a+b, 0) / landmarks.length,
    y: landmarks.map(p => p.y).reduce((a,b) => a+b, 0) / landmarks.length,
  };

  if (!lastCenter) {
    lastCenter = center;
    return "immobile";
  }

  const dx = center.x - lastCenter.x;
  const dy = center.y - lastCenter.y;
  const now = performance.now();
  const dt = now - lastTime;
  lastCenter = center;
  lastTime = now;

  if (Math.abs(dx) < 0.01 && Math.abs(dy) < 0.01) return "immobile";
  if (Math.abs(dx) > Math.abs(dy)) return dx > 0 ? "droite" : "gauche";
  return dy > 0 ? "bas" : "haut";
}

export async function createFakeDetector() {
  // ✅ En réalité c’est ton vrai modèle MediaPipe
  if (!landmarker) {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      },
      numHands: 2,
      runningMode: "video",
    });
  }

  return {
    detect: async (video) => {
      if (!landmarker) return [];
      const timestamp = performance.now();
      const results = await landmarker.detectForVideo(video, timestamp);
      if (!results?.landmarks) return [];

      // Convertit les landmarks en bounding boxes simples
      return results.landmarks.map((points) => {
            const xs = points.map((p) => p.x);
            const ys = points.map((p) => p.y);
            const x = Math.min(...xs);
            const y = Math.min(...ys);
            const w = Math.max(...xs) - x;
            const h = Math.max(...ys) - y;

            const fingerCount = countFingers(points);
            const fist = isFist(points);
            const direction = getHandDirection(points);

            return {
                x,
                y,
                w,
                h,
                color: fist ? "#f87171" : "#22d3ee", // rouge si poing
                label: `main (${fingerCount} doigts)`,
                direction,
                isFist: fist,
                fingerCount,
                landmarks: points.map((p) => [p.x, p.y]),
            };
        });
    },
  };
}
