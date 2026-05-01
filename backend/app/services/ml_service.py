"""
ML Service — loads and serves all trained models
"""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("models/artifacts")


class MLService:
    def __init__(self):
        self.scaler = None
        self.feature_cols = None
        self.prediction_features = None
        self.cluster_features = None
        self.habit_model = None
        self.churn_model = None
        self.kmeans_model = None
        self.cluster_profiles = []
        self.cluster_metrics = {}
        self.model_metrics = {}

    def load_models(self):
        self.scaler = joblib.load(BASE / "scaler.pkl")
        self.feature_cols = joblib.load(BASE / "feature_cols.pkl")
        self.prediction_features = joblib.load(BASE / "prediction_features.pkl")
        self.cluster_features = joblib.load(BASE / "cluster_features.pkl")
        self.habit_model = joblib.load(BASE / "habit_formation_model.pkl")
        self.churn_model = joblib.load(BASE / "churn_prediction_model.pkl")
        self.kmeans_model = joblib.load(BASE / "kmeans_model.pkl")

        with open(BASE / "cluster_profiles.json") as f:
            self.cluster_profiles = json.load(f)
        with open(BASE / "cluster_metrics.json") as f:
            self.cluster_metrics = json.load(f)
        with open(BASE / "model_metrics.json") as f:
            self.model_metrics = json.load(f)

    def _vectorize(self, features: dict) -> np.ndarray:
        vec = pd.DataFrame([features])
        for col in self.feature_cols:
            if col not in vec.columns:
                vec[col] = 0
        vec_scaled = self.scaler.transform(vec[self.feature_cols])
        return pd.DataFrame(vec_scaled, columns=self.feature_cols)

    def predict(self, features: dict) -> dict:
        vec_df = self._vectorize(features)
        X = vec_df[self.prediction_features].values

        habit_prob = float(self.habit_model.predict_proba(X)[0][1])
        churn_prob = float(self.churn_model.predict_proba(X)[0][1])

        milestones = []
        if features.get("consistency_rate", 0) > 0.4:
            milestones.append({"milestone": "7_day_streak", "probability": round(min(0.95, habit_prob + 0.2), 2)})
        if features.get("workout_density", 0) > 0.5:
            milestones.append({"milestone": "10th_workout", "probability": round(min(0.9, habit_prob + 0.1), 2)})
        if features.get("social_ratio", 0) > 0.1:
            milestones.append({"milestone": "first_social_share", "probability": round(min(0.85, habit_prob), 2)})
        if habit_prob > 0.6:
            milestones.append({"milestone": "21_day_habit_lock", "probability": round(habit_prob * 0.9, 2)})

        return {
            "habit_formation_probability": round(habit_prob, 3),
            "churn_probability": round(churn_prob, 3),
            "risk_level": "high" if churn_prob > 0.6 else "medium" if churn_prob > 0.35 else "low",
            "predicted_milestones": milestones,
        }

    def cluster(self, features: dict) -> dict:
        vec_df = self._vectorize(features)
        X = vec_df[self.cluster_features].values
        cluster_id = int(self.kmeans_model.predict(X)[0])
        distances = self.kmeans_model.transform(X)[0]
        confidence = round(1 - (distances[cluster_id] / (distances.sum() + 1e-9)), 3)

        labels = {0: "Habit Champions", 1: "Steady Builders", 2: "Casual Dabblers", 3: "At-Risk Dropouts"}
        return {
            "cluster_id": cluster_id,
            "cluster_label": labels.get(cluster_id, f"Segment {cluster_id}"),
            "confidence": confidence,
        }