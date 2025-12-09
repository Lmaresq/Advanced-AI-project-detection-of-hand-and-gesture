import React, { useEffect, useRef } from "react";

export default function OverlayCanvas({ videoRef, detections, fps, status, size }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, size, size);
    ctx.save();

    // --- Miroir pour rectangles et points ---
    ctx.scale(-1, 1);
    ctx.translate(-size, 0);

    const labels = []; // on stocke le texte à dessiner plus tard

    for (const det of detections) {
      const { x, y, w, h, color, label } = det;
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x * size, y * size, w * size, h * size);

      // on ne dessine pas le texte ici
      labels.push({
        text: `${label} ${det.isFist ? "👊" : ""} ${det.direction}`,
        x, y, w, color
      });

      if (det.landmarks) {
        ctx.fillStyle = det.color;
        for (const [lx, ly] of det.landmarks) {
          ctx.beginPath();
          ctx.arc(lx * size, ly * size, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    ctx.restore(); // stop le miroir ici

    // --- Texte lisible ---
    ctx.font = "16px sans-serif";
    for (const lab of labels) {
      ctx.fillStyle = lab.color;
      const mirroredX = size - (lab.x + lab.w) * size; // reposition horizontale miroirée
      ctx.fillText(lab.text, mirroredX + 5, lab.y * size - 5);
    }

    // --- HUD FPS et statut ---
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fillRect(10, 10, 130, 55);
    ctx.fillStyle = "white";
    ctx.font = "13px monospace";
    ctx.fillText(`Cam: ${status.camera}`, 20, 30);
    ctx.fillText(`Model: ${status.model}`, 20, 45);
    ctx.fillText(`FPS: ${fps.toFixed(1)}`, 20, 60);
  }, [detections, fps, status, size]);

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        pointerEvents: "none",
      }}
    />
  );
}
