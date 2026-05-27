import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "CS131 Cloud Run backend is online"
    })

@app.route("/alert", methods=["POST"])
def alert():
    data = request.get_json(silent=True)

    if data is None:
        app.logger.warning("No JSON received")
        return jsonify({"error": "No JSON received"}), 400

    app.logger.warning(f"Received alert JSON: {data}")

    return jsonify({
        "status": "success",
        "message": "Alert received",
        "data": data
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
