import React, { useEffect, useRef, useState } from "react";
import Hls from "hls.js";
import "./App.css";
import {collection,query,orderBy,limit,onSnapshot} from "firebase/firestore";
import { db } from "./firebase";

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
  const DEFAULT_STREAM_URL = "PUT_YOUR_M3U8_LINK_HERE";
  const [streamInput, setStreamInput] = useState(DEFAULT_STREAM_URL);
  const [streamUrl, setStreamUrl] = useState(DEFAULT_STREAM_URL);
  const [alerts, setAlerts] = useState([]);
  const [selectedAlert, setSelectedAlert] = useState(null);

  useEffect(() => {
    const alertsQuery = query(
      collection(db, "alerts"),
      orderBy("received_at", "desc"),
      limit(20)
    );

    const unsubscribe = onSnapshot(alertsQuery, (snapshot) => {
      const alertData = snapshot.docs.map((doc) => ({
        id: doc.id,
        ...doc.data(),
      }));

      setAlerts(alertData);

      if (alertData.length > 0) {
        setSelectedAlert((currentSelected) => currentSelected || alertData[0]);
      }
    });

    return () => unsubscribe();
  }, []);

  function loadStream() {
    setStreamUrl(streamInput.trim());
  }

  function formatTime(alert) {
    const timeValue = alert.received_at || alert.timestamp;

    if (!timeValue) {
      return "Unknown";
    }

    return new Date(timeValue).toLocaleString();
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

        {alerts.length === 0 ? (
          <p>No alerts received yet.</p>
        ) : (
          <div className="layout">
            <div className="alert-list">
              {alerts.map((alert) => (
                <button
                  key={alert.id}
                  className={
                    selectedAlert?.id === alert.id
                      ? "alert-card selected"
                      : "alert-card"
                  }
                  onClick={() => setSelectedAlert(alert)}
                >
                  <strong>{alert.label || alert.event_type || "Alert"}</strong>
                  <span>
                    Camera: {alert.camera_id || alert.device_id || "Unknown"}
                  </span>
                  <span>Time: {formatTime(alert)}</span>
                  <span>Zone Count: {alert.zone_count ?? "N/A"}</span>
                </button>
              ))}
            </div>

            {selectedAlert && (
              <div className="clip-viewer">
                <h3>
                  {selectedAlert.label ||
                    selectedAlert.event_type ||
                    "Alert Details"}
                </h3>

                <p>
                  Camera:{" "}
                  {selectedAlert.camera_id ||
                    selectedAlert.device_id ||
                    "Unknown"}
                </p>

                <p>Time: {formatTime(selectedAlert)}</p>

                <p>Zone Count: {selectedAlert.zone_count ?? "N/A"}</p>

                <p>Max Capacity: {selectedAlert.max_capacity ?? "N/A"}</p>

                <p>
                  Event Type: {selectedAlert.event_type || "N/A"}
                </p>

                {selectedAlert.detections &&
                  selectedAlert.detections.length > 0 && (
                    <div>
                      <p>Detections:</p>
                      <ul>
                        {selectedAlert.detections.map((detection, index) => (
                          <li key={index}>{detection}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                {selectedAlert.image_url && (
                  <img
                    className="alert-image"
                    src={selectedAlert.image_url}
                    alt="Alert snapshot"
                  />
                )}

                {/*{selectedAlert.snapshot_uri && (
                  <p className="snapshot-text">
                    Snapshot: {selectedAlert.snapshot_uri}
                  </p>
                )}*/}
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

export default App;