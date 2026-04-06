import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Paths
DATA_PATH = "../data/Sleep_health_and_lifestyle_dataset.csv"
MODEL_PATH = "../model/sleep_model.pkl"
SCALER_PATH = "../model/scaler.pkl"

def train_model():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Select relevant columns
    df = df[
        [
            "Sleep Duration",
            "Physical Activity Level",
            "Stress Level",
            "Heart Rate",
            "Daily Steps",
            "Quality of Sleep"
        ]
    ]

    # Drop missing values
    df.dropna(inplace=True)

    # Features and target
    X = df.drop("Quality of Sleep", axis=1)
    y = df["Quality of Sleep"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Train
    model.fit(X_train_scaled, y_train)

    # Evaluation
    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("✅ Model Training Completed")
    print(f"📉 Mean Absolute Error: {mae:.2f}")
    print(f"📈 R2 Score: {r2:.2f}")

    # Save model and scaler
    os.makedirs("../model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("💾 Model saved as sleep_model.pkl")
    print("💾 Scaler saved as scaler.pkl")

if __name__ == "__main__":
    train_model()
