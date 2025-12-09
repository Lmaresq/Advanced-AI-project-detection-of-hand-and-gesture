import React, { forwardRef, useEffect } from "react";

const CameraPanel = forwardRef(function CameraPanel({ size }, videoRef) {
  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch((err) => {
        console.error("Erreur accès caméra :", err);
      });
  }, [videoRef]);

  return (
    <video
  ref={videoRef}
  autoPlay
  playsInline
  muted
  width={size}
  height={size}
  style={{
    borderRadius: "12px",
    border: "3px solid #4ade80",
    boxShadow: "0 0 10px #4ade80",
    objectFit: "cover",
    transform: "scaleX(-1)", // 👈 effet miroir
  }}
/>

  );
});

export default CameraPanel;
