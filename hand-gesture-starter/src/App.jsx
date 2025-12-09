import React, { useRef } from "react";
import CameraPanel from "./components/CameraPanel";
import OverlayCanvas from "./components/OverlayCanvas";
import EventConsole from "./components/EventConsole";
import useDetections from "./hooks/useDetections";
import useBackendDetections from "./hooks/useBackendDetections";
import "./styles.css";

export default function App() {
  const myVideoRef = useRef(null);
  const googleVideoRef = useRef(null);

  const { detections: googleDetections, fps: googleFps, status: googleStatus, events } =
    useDetections(googleVideoRef);
  const {
    detections: myDetections,
    fps: myFps,
    status: myStatus,
  } = useBackendDetections(myVideoRef);

  const size = 360;

  const summarizeGoogle = (dets = []) => {
    const fingers = dets.map((d) => d.fingerCount).filter((n) => typeof n === "number");
    const fists = dets.filter((d) => d.isFist).length;
    return {
      hands: dets.length,
      fingers,
      fists,
    };
  };

  const summarizeMine = (dets = []) => {
    const bestScore = dets.reduce((m, d) => Math.max(m, d.score ?? 0), 0);
    return {
      hands: dets.length,
      bestScore,
    };
  };

  const googleStats = summarizeGoogle(googleDetections);
  const myStats = summarizeMine(myDetections);

  return (
    <div className="app">
      <h1>Comparaison de modèles de détection de main</h1>

      <div className="comparison-grid">
        <div className="panel">
          <h2>Mon modèle (YOLO backend)</h2>
          <div className="camera-stack" style={{ width: size, height: size }}>
            <CameraPanel ref={myVideoRef} size={size} />
            <OverlayCanvas
              videoRef={myVideoRef}
              detections={myDetections}
              fps={myFps}
              status={myStatus}
              size={size}
            />
          </div>
          <div className="info-block">
            <div className="info-title">Mon modèle</div>
            <div>Mains détectées : {myStats.hands}</div>
            <div>Score max : {(myStats.bestScore * 100).toFixed(0)}%</div>
          </div>
        </div>

        <div className="panel">
          <h2>Modèle Google (MediaPipe)</h2>
          <div className="camera-stack" style={{ width: size, height: size }}>
            <CameraPanel ref={googleVideoRef} size={size} />
            <OverlayCanvas
              videoRef={googleVideoRef}
              detections={googleDetections}
              fps={googleFps}
              status={googleStatus}
              size={size}
            />
          </div>
          <div className="info-block">
            <div className="info-title">Google</div>
            <div>Mains détectées : {googleStats.hands}</div>
            <div>
              Doigts comptés :{" "}
              {googleStats.fingers.length > 0 ? googleStats.fingers.join(", ") : "N/A"}
            </div>
            <div>Poings détectés : {googleStats.fists}</div>
          </div>
        </div>
      </div>

      <div className="summary-card">
        <div className="info-title">Bilan global</div>
        <div>
          Mon modèle voit {myStats.hands} main(s), Google en voit {googleStats.hands}.
        </div>
        <div>
          Scores YOLO (max) : {(myStats.bestScore * 100).toFixed(0)}% | Doigts Google :{" "}
          {googleStats.fingers.length > 0 ? googleStats.fingers.join(", ") : "N/A"}
        </div>
      </div>

      <EventConsole events={events} />
    </div>
  );
}
