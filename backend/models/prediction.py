"""
Prediction Model — Habit Formation Engine
Random Forest + Gradient Boosting ensemble for:
  1. Habit formation probability (will this user form a lasting habit?)
  2. Churn prediction (will this user drop off in 30 days?)
  3. Milestone prediction (which milestones will they hit?)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, f1_score
)
from sklearn.calibration import CalibratedClassifierCV
import joblib
import json
import os

os.makedirs("models/artifacts", exist_ok=True)

PREDICTION_FEATURES = [
    "w1_sessions", "w2_sessions", "engagement_velocity",
    "total_sessions_30d", "total_active_days_30d",
    "consistency_rate", "streak_to_days_ratio",
    "sessions_per_active_day", "session_gap_std", "max_streak_30d",
    "workout_density", "duration_per_session",
    "feature_adoption_score", "workouts_logged",
    "social_ratio", "social_actions",
    "milestone_rate", "milestones_hit",
    "routine_encoded", "age",
]


def build_ensemble():
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    lr = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft",
        weights=[3, 3, 1],
    )
    return ensemble


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
        "f1_score": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    return metrics


def get_feature_importance(model, feature_names: list) -> list:
    """Extract feature importances from the ensemble."""
    importances = {}
    
    # From RF
    rf_est = model.estimators_[0][1] if hasattr(model, 'estimators_') else model
    if hasattr(rf_est, 'feature_importances_'):
        for fname, imp in zip(feature_names, rf_est.feature_importances_):
            importances[fname] = float(imp)
    
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return [{"feature": k, "importance": round(v, 4)} for k, v in sorted_imp]


def train_models(feats_df: pd.DataFrame):
    print("🤖 Training prediction models...")

    X = feats_df[PREDICTION_FEATURES].values
    y_habit = feats_df["retained_long_term"].values
    y_churn = feats_df["churned_30d"].values

    all_metrics = {}

    for task, y in [("habit_formation", y_habit), ("churn_prediction", y_churn)]:
        print(f"\n   Training: {task}")
        print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = build_ensemble()
        model.fit(X_train, y_train)

        # Calibrate probabilities
        calibrated = CalibratedClassifierCV(model, cv=3, method="isotonic")
        calibrated.fit(X_train, y_train)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        print(f"   CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        metrics = evaluate_model(calibrated, X_test, y_test, task)
        metrics["cv_roc_auc_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_roc_auc_std"] = round(cv_scores.std(), 4)
        all_metrics[task] = metrics

        print(f"   Test ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"   Avg Precision: {metrics['avg_precision']:.4f}")

        # Feature importance (from uncalibrated for access to base estimators)
        importances = get_feature_importance(model, PREDICTION_FEATURES)
        metrics["feature_importances"] = importances[:10]  # top 10

        joblib.dump(calibrated, f"models/artifacts/{task}_model.pkl")

    joblib.dump(PREDICTION_FEATURES, "models/artifacts/prediction_features.pkl")

    with open("models/artifacts/model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n✅ Models trained and saved.")
    return all_metrics


def predict_user(user_features: dict) -> dict:
    """Run full prediction pipeline for a single user."""
    scaler = joblib.load("models/artifacts/scaler.pkl")
    feature_cols = joblib.load("models/artifacts/feature_cols.pkl")
    pred_features = joblib.load("models/artifacts/prediction_features.pkl")

    habit_model = joblib.load("models/artifacts/habit_formation_model.pkl")
    churn_model = joblib.load("models/artifacts/churn_prediction_model.pkl")

    vec = pd.DataFrame([user_features])
    for col in feature_cols:
        if col not in vec.columns:
            vec[col] = 0

    vec_scaled = scaler.transform(vec[feature_cols])
    vec_df = pd.DataFrame(vec_scaled, columns=feature_cols)
    X = vec_df[pred_features].values

    habit_prob = float(habit_model.predict_proba(X)[0][1])
    churn_prob = float(churn_model.predict_proba(X)[0][1])

    # Derive milestone predictions based on feature signals
    milestones = []
    raw = user_features
    if raw.get("consistency_rate", 0) > 0.4:
        milestones.append({"milestone": "7_day_streak", "probability": round(min(0.95, habit_prob + 0.2), 2)})
    if raw.get("workout_density", 0) > 0.5:
        milestones.append({"milestone": "10th_workout", "probability": round(min(0.9, habit_prob + 0.1), 2)})
    if raw.get("social_ratio", 0) > 0.1:
        milestones.append({"milestone": "first_social_share", "probability": round(min(0.85, habit_prob), 2)})
    if habit_prob > 0.6:
        milestones.append({"milestone": "21_day_habit_lock", "probability": round(habit_prob * 0.9, 2)})

    return {
        "habit_formation_probability": round(habit_prob, 3),
        "churn_probability": round(churn_prob, 3),
        "risk_level": "high" if churn_prob > 0.6 else "medium" if churn_prob > 0.35 else "low",
        "predicted_milestones": milestones,
    }


if __name__ == "__main__":
    feats_df = pd.read_csv("data/processed/features_raw.csv")

    scaler = joblib.load("models/artifacts/scaler.pkl")
    feature_cols = joblib.load("models/artifacts/feature_cols.pkl")
    feats_scaled = pd.read_csv("data/processed/features_scaled.csv")

    train_models(feats_df)