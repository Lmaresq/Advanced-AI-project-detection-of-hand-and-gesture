import React, { useEffect, useRef } from "react";

export default function EventConsole({ events }) {
  const ref = useRef(null);

  // 🔹 Scroll automatique vers le bas quand un nouvel event arrive
  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [events]);

  return (
    <div
      ref={ref}
      style={{
        width: "360px",
        height: "120px",
        backgroundColor: "#0a0a0a",
        color: "#00ff9d",
        fontFamily: "monospace",
        fontSize: "13px",
        overflowY: "auto",
        borderRadius: "8px",
        marginTop: "10px",
        padding: "8px",
        boxShadow: "0 0 10px #00ff9d40",
      }}
    >
      {events.slice(-10).map((e, i) => (
        <div key={i}>$ {e}</div>
      ))}
    </div>
  );
}
