import os
from flask import Flask, request, jsonify
from google.cloud import firestore
from datetime import datetime, timezone

app = Flask(__name__)

# Firestore setup
db = firestore.Client()
ALERTS_COLLECTION = "alerts"


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "CS131 Cloud Run backend is online"
    }), 200


@app.route("/alert", methods=["POST"])
def alert():
    data = request.get_json(silent=True)

    if data is None:
        app.logger.warning("No JSON received")
        return jsonify({"error": "No JSON received"}), 400

    # Add server-side timestamp
    data["received_at"] = datetime.now(timezone.utc).isoformat()

    # Save alert to Firestore
    doc_ref = db.collection(ALERTS_COLLECTION).document()
    doc_ref.set(data)

    app.logger.warning(f"Received alert JSON: {data}")
    app.logger.warning(f"Saved alert to Firestore with ID: {doc_ref.id}")

    return jsonify({
        "status": "success",
        "message": "Alert received and saved",
        "alert_id": doc_ref.id,
        "data": data
    }), 200


@app.route("/alerts", methods=["GET"])
def get_alerts():
    docs = (
        db.collection(ALERTS_COLLECTION)
        .order_by("received_at", direction=firestore.Query.DESCENDING)
        .limit(20)
        .stream()
    )

    alerts = []

    for doc in docs:
        alert_data = doc.to_dict()
        alert_data["id"] = doc.id
        alerts.append(alert_data)

    return jsonify({
        "status": "success",
        "alerts": alerts
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)