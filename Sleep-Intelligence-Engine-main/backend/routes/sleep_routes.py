from flask import Blueprint, request, jsonify
import numpy as np
import joblib
import os

sleep_bp = Blueprint("sleep_bp", __name__)

# Load model
MODEL_PATH = os.path.join("model", "sleep_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@sleep_bp.route("/predict", methods=["POST"])
def predict_sleep():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON data received"}), 400

        # ✅ Read EXACT keys sent by frontend
        sleep_duration = float(data.get("sleep_duration"))
        physical_activity = int(data.get("physical_activity_level"))
        stress_level = int(data.get("stress_level"))
        heart_rate = int(data.get("heart_rate"))
        daily_steps = int(data.get("daily_steps"))

        # Create input array
        features = np.array([[
            sleep_duration,
            physical_activity,
            stress_level,
            heart_rate,
            daily_steps
        ]])

        # Scale + Predict
        scaled_features = scaler.transform(features)
        sleep_score = model.predict(scaled_features)[0]

        # Simple interpretation
        if sleep_score >= 7:
            pattern = "Good"
            recommendations = [
                "Maintain your current sleep routine",
                "Keep stress levels low",
                "Stay physically active"
            ]
        else:
            pattern = "Poor"
            recommendations = [
                "Increase sleep duration",
                "Reduce stress before bedtime",
                "Limit screen usage at night"
            ]

        return jsonify({
            "sleep_score": round(float(sleep_score), 2),
            "sleep_pattern": pattern,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 400
