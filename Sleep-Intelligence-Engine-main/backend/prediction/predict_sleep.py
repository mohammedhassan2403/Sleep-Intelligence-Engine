import joblib
import pandas as pd
import os

# --------------------------------------------------
# Absolute path handling (PRODUCTION SAFE)
# --------------------------------------------------

# Get absolute path to backend directory
BACKEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(BACKEND_DIR, "model", "sleep_model.pkl")
SCALER_PATH = os.path.join(BACKEND_DIR, "model", "scaler.pkl")

print("Loading model from:", MODEL_PATH)

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --------------------------------------------------
# Business Logic
# --------------------------------------------------

def classify_sleep_pattern(score):
    if score >= 8:
        return "Deep Sleep"
    elif score >= 6:
        return "Moderate Sleep"
    else:
        return "Poor Sleep"


def generate_recommendations(input_data, score):
    tips = []

    if input_data["Sleep Duration"] < 7:
        tips.append("Try to sleep at least 7–8 hours per night.")

    if input_data["Stress Level"] > 6:
        tips.append("Practice relaxation techniques before bedtime.")

    if input_data["Physical Activity Level"] < 40:
        tips.append("Increase daily physical activity for better sleep.")

    if input_data["Daily Steps"] < 5000:
        tips.append("Aim for at least 6,000–8,000 steps daily.")

    if input_data["Heart Rate"] > 80:
        tips.append("Avoid caffeine and heavy meals before sleep.")

    if not tips:
        tips.append("Great sleep habits! Keep maintaining consistency.")

    return tips


def predict_sleep(input_data):
    """
    input_data example:
    {
        "Sleep Duration": 6.5,
        "Physical Activity Level": 45,
        "Stress Level": 7,
        "Heart Rate": 78,
        "Daily Steps": 6000
    }
    """

    # Convert input to DataFrame (avoids sklearn warning)
    input_df = pd.DataFrame([input_data])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict sleep score
    sleep_score = model.predict(input_scaled)[0]
    sleep_score = round(float(sleep_score), 1)

    # Pattern classification
    pattern = classify_sleep_pattern(sleep_score)

    # Personalized recommendations
    tips = generate_recommendations(input_data, sleep_score)

    return {
        "sleep_score": sleep_score,
        "sleep_pattern": pattern,
        "recommendations": tips
    }


# --------------------------------------------------
# Standalone testing
# --------------------------------------------------

if __name__ == "__main__":
    sample_input = {
        "Sleep Duration": 6.2,
        "Physical Activity Level": 35,
        "Stress Level": 8,
        "Heart Rate": 82,
        "Daily Steps": 4200
    }

    result = predict_sleep(sample_input)
    print(result)

