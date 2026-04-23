# app.py
from flask import Flask, request, jsonify
from src.predict import predict_email
from src.anomaly import detect_anomaly, train_autoencoder

app = Flask(__name__)

# Ensure anomaly model exists for API usage
try:
    train_autoencoder()
except Exception:
    # Training will be handled when user runs src/anomaly.py
    pass

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = predict_email(data['text'])
    return jsonify({'prediction': result})


@app.route('/anomaly', methods=['POST'])
def anomaly():
    data = request.json
    features = data.get('features', [])
    result = detect_anomaly(features)
    return jsonify(result)

app.run(debug=True)