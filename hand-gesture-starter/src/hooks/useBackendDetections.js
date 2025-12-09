import { useEffect, useRef, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Capture the current video frame as a JPEG blob.
async function grabFrameBlob(video) {
  if (!video || video.videoWidth === 0 || video.videoHeight === 0) return null;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));
}

export default function useBackendDetections(videoRef) {
  const [detections, setDetections] = useState([]);
  const [fps, setFps] = useState(0);
  const [status, setStatus] = useState({ camera: "init", model: "idle" });
  const lastTime = useRef(performance.now());

  useEffect(() => {
    let running = true;

    const loop = async () => {
      if (!running) return;
      const now = performance.now();
      const delta = now - lastTime.current;
      setFps(1000 / delta);
      lastTime.current = now;

      const video = videoRef.current;
      if (!video || video.readyState < 2 || video.videoWidth === 0) {
        setStatus((s) => ({ ...s, camera: "en attente" }));
        setTimeout(loop, 250);
        return;
      }

      setStatus((s) => ({ ...s, camera: "ok" }));
      try {
        const blob = await grabFrameBlob(video);
        if (!blob) {
          setTimeout(loop, 250);
          return;
        }

        const formData = new FormData();
        formData.append("file", new File([blob], "frame.jpg", { type: "image/jpeg" }));

        const resp = await fetch(`${API_URL}/detect?min_conf=0.25`, {
          method: "POST",
          body: formData,
        });

        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}`);
        }

        const data = await resp.json();
        const mapped = (data?.detections || []).map((d) => ({
          ...d,
          color: "#a78bfa", // violet pour notre modèle
          label: `${d.label ?? "hand"} ${(d.score * 100).toFixed(0)}%`,
        }));

        setDetections(mapped);
        setStatus({ camera: "ok", model: "ok" });
      } catch (err) {
        console.warn("Backend detection error", err);
        setStatus((s) => ({ ...s, model: "erreur" }));
      }

      setTimeout(loop, 250); // throttle backend calls (~4 FPS)
    };

    loop();
    return () => {
      running = false;
    };
  }, [videoRef]);

  return { detections, fps, status };
}
