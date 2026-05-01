"""
Feature Engineering Pipeline — Habit Formation Engine
Transforms raw event logs into ML-ready feature vectors.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models/artifacts", exist_ok=True)


def load_raw_data():
    users = pd.read_csv("data/raw/users.csv")
    events = pd.read_csv("data/raw/events.csv")
    return users, events


def engineer_features(users_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the primary feature matrix from user + event data.
    
    Feature groups:
    1. Engagement velocity   — how fast they ramp up
    2. Consistency signals   — streak patterns, regularity
    3. Depth signals         — feature breadth, workout density
    4. Social signals        — community engagement
    5. Milestone cadence     — which milestones hit & when
    """
    feats = users_df.copy()

    # ── 1. Engagement velocity (first-week vs. second-week) ────────────────
    week1 = events_df[events_df["day_of_onboarding"] <= 7]
    week2 = events_df[(events_df["day_of_onboarding"] > 7) & (events_df["day_of_onboarding"] <= 14)]

    w1_agg = (
        week1.groupby("user_id")
        .agg(
            w1_sessions=("session_duration_min", "count"),
            w1_duration=("session_duration_min", "sum"),
            w1_workouts=("workout_logged", "sum"),
            w1_features=("n_features_used", "mean"),
        )
        .reset_index()
    )
    w2_agg = (
        week2.groupby("user_id")
        .agg(
            w2_sessions=("session_duration_min", "count"),
            w2_duration=("session_duration_min", "sum"),
        )
        .reset_index()
    )

    feats = feats.merge(w1_agg, on="user_id", how="left")
    feats = feats.merge(w2_agg, on="user_id", how="left")
    feats[["w1_sessions", "w2_sessions"]] = feats[["w1_sessions", "w2_sessions"]].fillna(0)

    # Week-over-week velocity
    feats["engagement_velocity"] = (
        (feats["w2_sessions"] - feats["w1_sessions"]) /
        (feats["w1_sessions"] + 1)
    )

    # ── 2. Consistency signals ─────────────────────────────────────────────
    feats["consistency_rate"] = feats["total_active_days_30d"] / 30.0
    feats["streak_to_days_ratio"] = feats["max_streak_30d"] / (feats["total_active_days_30d"] + 1)
    feats["sessions_per_active_day"] = feats["total_sessions_30d"] / (feats["total_active_days_30d"] + 1)

    # Standard deviation of session gaps (regularity)
    def session_gap_std(uid):
        u_events = events_df[events_df["user_id"] == uid].copy()
        if len(u_events) < 2:
            return 10.0
        u_events = u_events.drop_duplicates("event_date").sort_values("day_of_onboarding")
        gaps = u_events["day_of_onboarding"].diff().dropna()
        return float(gaps.std()) if len(gaps) > 1 else 10.0

    feats["session_gap_std"] = feats["user_id"].apply(session_gap_std)

    # ── 3. Depth signals ───────────────────────────────────────────────────
    feats["workout_density"] = feats["workouts_logged"] / (feats["total_sessions_30d"] + 1)
    feats["duration_per_session"] = feats["avg_session_duration_min"]
    feats["feature_adoption_score"] = np.log1p(feats["w1_features"].fillna(0))

    # ── 4. Social signals ──────────────────────────────────────────────────
    feats["social_ratio"] = feats["social_actions"] / (feats["total_sessions_30d"] + 1)

    # ── 5. Milestone signals ───────────────────────────────────────────────
    feats["milestone_rate"] = feats["milestones_hit"] / 6.0  # max possible milestones

    # ── 6. Categorical encoding ────────────────────────────────────────────
    le = LabelEncoder()
    feats["routine_encoded"] = le.fit_transform(feats["fitness_routine"].fillna("mixed"))
    joblib.dump(le, "models/artifacts/routine_encoder.pkl")

    # ── 7. Composite habit score (business KPI) ────────────────────────────
    feats["habit_score"] = (
        feats["consistency_rate"] * 0.30 +
        feats["workout_density"] * 0.25 +
        feats["milestone_rate"] * 0.20 +
        feats["social_ratio"] * 0.10 +
        feats["feature_adoption_score"].clip(0, 1) * 0.15
    )

    # ── Select final feature columns ───────────────────────────────────────
    FEATURE_COLS = [
        "user_id",
        # Engagement
        "w1_sessions", "w2_sessions", "engagement_velocity",
        "total_sessions_30d", "total_active_days_30d",
        # Consistency
        "consistency_rate", "streak_to_days_ratio",
        "sessions_per_active_day", "session_gap_std", "max_streak_30d",
        # Depth
        "workout_density", "duration_per_session",
        "feature_adoption_score", "workouts_logged",
        # Social
        "social_ratio", "social_actions",
        # Milestones
        "milestone_rate", "milestones_hit",
        # Categorical
        "routine_encoded", "age",
        # Target & composite
        "habit_score", "churned_30d", "retained_long_term",
        "persona",  # for analysis only
    ]

    feats = feats[FEATURE_COLS].fillna(0)
    return feats


def scale_features(feats: pd.DataFrame):
    ML_FEATURES = [
        "w1_sessions", "w2_sessions", "engagement_velocity",
        "total_sessions_30d", "total_active_days_30d",
        "consistency_rate", "streak_to_days_ratio",
        "sessions_per_active_day", "session_gap_std", "max_streak_30d",
        "workout_density", "duration_per_session",
        "feature_adoption_score", "workouts_logged",
        "social_ratio", "social_actions",
        "milestone_rate", "milestones_hit",
        "routine_encoded", "age", "habit_score",
    ]

    scaler = StandardScaler()
    feats_scaled = feats.copy()
    feats_scaled[ML_FEATURES] = scaler.fit_transform(feats[ML_FEATURES])

    joblib.dump(scaler, "models/artifacts/scaler.pkl")
    joblib.dump(ML_FEATURES, "models/artifacts/feature_cols.pkl")

    return feats_scaled, ML_FEATURES


def run_pipeline():
    print("🔧 Running feature engineering pipeline...")
    users, events = load_raw_data()
    print(f"   Loaded {len(users):,} users, {len(events):,} events")

    feats = engineer_features(users, events)
    feats.to_csv("data/processed/features_raw.csv", index=False)
    print(f"   Raw features saved → data/processed/features_raw.csv")

    feats_scaled, feature_cols = scale_features(feats)
    feats_scaled.to_csv("data/processed/features_scaled.csv", index=False)
    print(f"   Scaled features saved → data/processed/features_scaled.csv")
    print(f"   Feature count: {len(feature_cols)}")
    print("✅ Pipeline complete.")

    return feats, feats_scaled, feature_cols


if __name__ == "__main__":
    run_pipeline()