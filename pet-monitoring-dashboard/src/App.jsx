import React, { useEffect, useRef, useState } from "react";
import Hls from "hls.js";
import "./App.css";

const mockIncidents = [
  {
    id: "INC-001",
    type: "Pet Barking",
    camera: "Living Room",
    timestamp: "2026-05-20 2:32 PM",
    confidence: 91,
    videoUrl: "https://www.w3schools.com/html/mov_bbb.mp4",
  },
  {
    id: "INC-002",
    type: "Pet Fighting",
    camera: "Playroom",
    timestamp: "2026-05-20 3:10 PM",
    confidence: 84,
    videoUrl: "https://www.w3schools.com/html/movie.mp4",
  },
  {
    id: "INC-003",
    type: "Bathroom Event",
    camera: "Litter Area",
    timestamp: "2026-05-20 4:05 PM",
    confidence: 88,
    videoUrl: "https://www.w3schools.com/html/mov_bbb.mp4",
  },
  {
    id: "INC-004",
    type: "Suspicious Human",
    camera: "Front Door",
    timestamp: "2026-05-20 5:18 PM",
    confidence: 93,
    videoUrl: "https://www.w3schools.com/html/movie.mp4",
  },
];

function LiveStreamPlayer({ streamUrl }) {
  const videoRef = useRef(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !streamUrl) return;

    let hls;

    if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = streamUrl;
    } else if (Hls.isSupported()) {
      hls = new Hls();
      hls.loadSource(streamUrl);
      hls.attachMedia(video);
    } else {
      alert("HLS stream is not supported in this browser.");
    }

    return () => {
      if (hls) hls.destroy();
    };
  }, [streamUrl]);

  return (
    <video className="video-player" ref={videoRef} controls autoPlay muted />
  );
}

function App() {
  const [streamInput, setStreamInput] = useState("");
  const [streamUrl, setStreamUrl] = useState("");
  const [selectedIncident, setSelectedIncident] = useState(mockIncidents[0]);

  function loadStream() {
    setStreamUrl(streamInput.trim());
  }

  return (
    <div className="app">
      <h1>Pet Monitoring Dashboard</h1>

      <section className="section">
        <h2>Live Feed</h2>

        <div className="input-row">
          <input
            type="text"
            placeholder="Paste .m3u8 stream link here"
            value={streamInput}
            onChange={(e) => setStreamInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") loadStream();
            }}
          />

          <button onClick={loadStream}>Load Stream</button>
        </div>

        {streamUrl ? (
          <LiveStreamPlayer streamUrl={streamUrl} />
        ) : (
          <p>No stream loaded yet.</p>
        )}
      </section>

      <section className="section">
        <h2>Alerts</h2>

        <div className="layout">
          <div className="alert-list">
            {mockIncidents.map((incident) => (
              <button
                key={incident.id}
                className={
                  selectedIncident.id === incident.id
                    ? "alert-card selected"
                    : "alert-card"
                }
                onClick={() => setSelectedIncident(incident)}
              >
                <strong>{incident.type}</strong>
                <span>{incident.camera}</span>
                <span>{incident.timestamp}</span>
                <span>{incident.confidence}% confidence</span>
              </button>
            ))}
          </div>

          <div className="clip-viewer">
            <h3>{selectedIncident.type}</h3>
            <p>Camera: {selectedIncident.camera}</p>
            <p>Time: {selectedIncident.timestamp}</p>
            <p>Confidence: {selectedIncident.confidence}%</p>

            <video className="video-player" controls>
              <source src={selectedIncident.videoUrl} type="video/mp4" />
              Your browser does not support video playback.
            </video>
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;