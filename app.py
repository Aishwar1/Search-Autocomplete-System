from flask import Flask, request, jsonify, render_template
from model.inference import autocomplete
from model.train import load_model
import json
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    model = load_model()
    query = request.json.get("query", "")
    return jsonify(autocomplete(query))


@app.route("/metrics")
def metrics():
    with open(METRICS_PATH) as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
