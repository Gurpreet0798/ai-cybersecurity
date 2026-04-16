# app.py
from flask import Flask, request, jsonify
from src.predict import predict_email

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = predict_email(data['text'])
    return jsonify({'prediction': result})

app.run(debug=True)