from flask import Flask, request, jsonify, render_template
from model.inference import autocomplete
import json
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    query = data.get("query", "")
    return jsonify({"suggestions": autocomplete(query)})

@app.route("/metrics")
def metrics():
    metrics_path = os.path.join(BASE_DIR, "model", "metrics.json")
    with open(metrics_path) as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    app.run(debug=True)
