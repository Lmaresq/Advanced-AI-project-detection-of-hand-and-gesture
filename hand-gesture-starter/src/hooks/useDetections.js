import { useEffect, useRef, useState } from "react";
import { createFakeDetector } from "../detectors/FakeDetector";

export default function useDetections(videoRef) {
  const [detections, setDetections] = useState([]);
  const [fps, setFps] = useState(0);
  const [status, setStatus] = useState({ camera: "init", model: "loading" });
  const lastTime = useRef(performance.now());
  const detectorRef = useRef(null);
  const [events, setEvents] = useState([]);
  const lastEvent = useRef(null);
  const lastEventTime = useRef(0);

  useEffect(() => {
    let running = true;
    async function loop() {
  if (!running) return;

  const now = performance.now();
  const delta = now - lastTime.current;
  setFps(1000 / delta);
  lastTime.current = now;

  const video = videoRef.current;

  // ✅ Vérifie que la caméra est prête
  if (video && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
    setStatus((s) => ({ ...s, camera: "ok" }));
    try {
      const results = await detectorRef.current.detect(video);
      setDetections(results);

      // Gestion d’événements stables
      if (results.length > 0) {
        const r = results[0]; // une seule main gérée pour simplifier
        const current = `${r.isFist ? "Poing fermé" : `${r.fingerCount} doigts`} ${r.direction ?? ""}`;
        const now = performance.now();

        if (current !== lastEvent.current) {
          // nouveau geste -> reset du chrono
          lastEvent.current = current;
          lastEventTime.current = now;
        } else if (now - lastEventTime.current > 50) {
          // stable pendant 50ms => on le log
          setEvents((ev) => {
            if (ev[ev.length - 1] !== current) {
              return [...ev, `${new Date().toLocaleTimeString()} - ${current}`];
            }
            return ev;
          });
          lastEventTime.current = now + 100; // évite le spam
        }
      }

    } catch (e) {
      console.warn("Détection ignorée (caméra pas encore prête)");
    }
  } else {
    setStatus((s) => ({ ...s, camera: "en attente" }));
  }

  requestAnimationFrame(loop);
}


    async function init() {
      detectorRef.current = await createFakeDetector();
      setStatus({ camera: "init", model: "ok" });
      loop();
    }

    init();

    return () => {
      running = false;
    };
  }, [videoRef]);

  return { detections, fps, status, events };
}
