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
        return jsonify({"error": "No JSON received"}), 400

    print("Received alert:", data)

    return jsonify({
        "status": "success",
        "message": "Alert received",
        "data": data
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
