"""
API Routes — Habit Formation Engine
All prediction, clustering, and recommendation endpoints.
"""

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import pandas as pd
import numpy as np
import json
import time

router = APIRouter()


# ── Request / Response Schemas ─────────────────────────────────────────────

class UserEngagementInput(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    
    # Week 1 signals
    w1_sessions: float = Field(0, ge=0, description="Sessions in first week")
    w2_sessions: float = Field(0, ge=0, description="Sessions in second week")
    
    # Consistency
    total_sessions_30d: float = Field(0, ge=0)
    total_active_days_30d: float = Field(0, ge=0, le=30)
    max_streak_30d: float = Field(0, ge=0, le=30)
    
    # Depth
    workouts_logged: float = Field(0, ge=0)
    avg_session_duration_min: float = Field(0, ge=0)
    milestones_hit: int = Field(0, ge=0)
    social_actions: int = Field(0, ge=0)
    
    # Demographics
    age: int = Field(25, ge=13, le=100)
    fitness_routine: str = Field("mixed", description="strength|cardio|yoga|hiit|mixed|rehabilitation")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "usr_demo_001",
                "w1_sessions": 5,
                "w2_sessions": 6,
                "total_sessions_30d": 22,
                "total_active_days_30d": 18,
                "max_streak_30d": 12,
                "workouts_logged": 15,
                "avg_session_duration_min": 28.5,
                "milestones_hit": 3,
                "social_actions": 4,
                "age": 28,
                "fitness_routine": "strength"
            }
        }


class PredictionResponse(BaseModel):
    user_id: str
    habit_formation_probability: float
    churn_probability: float
    risk_level: str
    cluster_id: int
    cluster_label: str
    habit_score: float
    predicted_milestones: List[dict]
    recommendations: List[dict]
    inference_time_ms: float


class ClusterSummaryResponse(BaseModel):
    clusters: List[dict]
    total_users: int
    model_metrics: dict


# ── Helper: derive features ────────────────────────────────────────────────

def derive_features(inp: UserEngagementInput) -> dict:
    routine_map = {"strength": 0, "cardio": 1, "yoga": 2, "hiit": 3, "mixed": 4, "rehabilitation": 5}
    s30 = max(inp.total_sessions_30d, 1)
    a30 = max(inp.total_active_days_30d, 1)

    engagement_velocity = (inp.w2_sessions - inp.w1_sessions) / (inp.w1_sessions + 1)
    consistency_rate = inp.total_active_days_30d / 30.0
    streak_ratio = inp.max_streak_30d / (a30 + 1)
    sessions_per_active = s30 / a30
    workout_density = inp.workouts_logged / s30
    social_ratio = inp.social_actions / s30
    milestone_rate = inp.milestones_hit / 6.0
    feature_adoption = np.log1p(inp.w1_sessions / 7 * 3)  # approx
    
    habit_score = (
        consistency_rate * 0.30 +
        workout_density * 0.25 +
        milestone_rate * 0.20 +
        social_ratio * 0.10 +
        min(feature_adoption, 1) * 0.15
    )

    return {
        "w1_sessions": inp.w1_sessions,
        "w2_sessions": inp.w2_sessions,
        "engagement_velocity": engagement_velocity,
        "total_sessions_30d": inp.total_sessions_30d,
        "total_active_days_30d": inp.total_active_days_30d,
        "consistency_rate": consistency_rate,
        "streak_to_days_ratio": streak_ratio,
        "sessions_per_active_day": sessions_per_active,
        "session_gap_std": max(1.0, 7.0 - consistency_rate * 5),  # estimated
        "max_streak_30d": inp.max_streak_30d,
        "workout_density": workout_density,
        "duration_per_session": inp.avg_session_duration_min,
        "feature_adoption_score": feature_adoption,
        "workouts_logged": inp.workouts_logged,
        "social_ratio": social_ratio,
        "social_actions": inp.social_actions,
        "milestone_rate": milestone_rate,
        "milestones_hit": inp.milestones_hit,
        "routine_encoded": routine_map.get(inp.fitness_routine, 4),
        "age": inp.age,
        "habit_score": habit_score,
    }


def generate_recommendations(
    habit_prob: float,
    churn_prob: float,
    cluster_label: str,
    features: dict,
) -> List[dict]:
    """Rule-based + heuristic recommendation engine."""
    recs = []

    if churn_prob > 0.6:
        recs.append({
            "type": "intervention",
            "priority": "critical",
            "title": "Re-engagement Campaign",
            "message": "User shows high dropout risk. Trigger a personalized 'We miss you' push notification with a 3-day streak challenge.",
            "action": "send_reengagement_push",
        })

    if features["max_streak_30d"] < 3:
        recs.append({
            "type": "gamification",
            "priority": "high",
            "title": "Streak Starter Challenge",
            "message": "No meaningful streak yet. Surface a 3-day mini-challenge with a badge reward to build initial habit loop.",
            "action": "trigger_streak_challenge",
        })
    elif features["max_streak_30d"] >= 7:
        recs.append({
            "type": "reinforcement",
            "priority": "medium",
            "title": "Streak Milestone Celebration",
            "message": f"User hit a {int(features['max_streak_30d'])}-day streak. Send milestone badge + share prompt to boost social retention.",
            "action": "trigger_milestone_celebration",
        })

    if features["social_ratio"] < 0.05:
        recs.append({
            "type": "social",
            "priority": "medium",
            "title": "Community Onboarding",
            "message": "Low social engagement. Invite to join a fitness challenge group matching their routine.",
            "action": "invite_to_community_challenge",
        })

    if features["workout_density"] < 0.3 and habit_prob > 0.4:
        recs.append({
            "type": "content",
            "priority": "medium",
            "title": "Workout Plan Suggestion",
            "message": "Good engagement but low workout logging. Surface a structured 4-week plan to drive deeper feature adoption.",
            "action": "recommend_workout_plan",
        })

    if habit_prob > 0.75:
        recs.append({
            "type": "upsell",
            "priority": "low",
            "title": "Premium Conversion Candidate",
            "message": "High habit formation probability. Optimal window for premium upgrade offer — show personalized coach matching.",
            "action": "show_premium_upgrade_prompt",
        })

    return recs[:4]  # cap at 4


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_user(request: Request, body: UserEngagementInput):
    """
    Full prediction pipeline for a user.
    Returns habit formation probability, churn risk, cluster assignment, and interventions.
    """
    start = time.time()

    ml = request.app.state.ml_service
    if ml is None:
        raise HTTPException(503, "ML models not loaded. Run training pipeline first.")

    features = derive_features(body)

    try:
        pred = ml.predict(features)
        cluster = ml.cluster(features)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {str(e)}")

    recs = generate_recommendations(
        pred["habit_formation_probability"],
        pred["churn_probability"],
        cluster["cluster_label"],
        features,
    )

    return PredictionResponse(
        user_id=body.user_id,
        habit_formation_probability=pred["habit_formation_probability"],
        churn_probability=pred["churn_probability"],
        risk_level=pred["risk_level"],
        cluster_id=cluster["cluster_id"],
        cluster_label=cluster["cluster_label"],
        habit_score=round(features["habit_score"], 3),
        predicted_milestones=pred["predicted_milestones"],
        recommendations=recs,
        inference_time_ms=round((time.time() - start) * 1000, 2),
    )


@router.get("/clusters", response_model=ClusterSummaryResponse, tags=["Clustering"])
async def get_clusters(request: Request):
    """Returns cluster profiles and model quality metrics."""
    ml = request.app.state.ml_service
    if ml is None:
        raise HTTPException(503, "ML models not loaded.")

    return ClusterSummaryResponse(
        clusters=ml.cluster_profiles,
        total_users=sum(c["size"] for c in ml.cluster_profiles),
        model_metrics=ml.cluster_metrics,
    )


@router.get("/metrics", tags=["System"])
async def get_model_metrics(request: Request):
    """Returns prediction model evaluation metrics (ROC-AUC, F1, etc.)."""
    ml = request.app.state.ml_service
    if ml is None:
        raise HTTPException(503, "ML models not loaded.")
    return ml.model_metrics


@router.get("/demo-users", tags=["Demo"])
async def get_demo_users():
    """Returns pre-built demo user profiles for portfolio demo."""
    return {
        "users": [
            {
                "label": "Power User",
                "user_id": "demo_power",
                "w1_sessions": 6, "w2_sessions": 7,
                "total_sessions_30d": 26, "total_active_days_30d": 24,
                "max_streak_30d": 21, "workouts_logged": 20,
                "avg_session_duration_min": 35, "milestones_hit": 5,
                "social_actions": 8, "age": 29, "fitness_routine": "strength",
            },
            {
                "label": "Casual User",
                "user_id": "demo_casual",
                "w1_sessions": 2, "w2_sessions": 2,
                "total_sessions_30d": 8, "total_active_days_30d": 7,
                "max_streak_30d": 3, "workouts_logged": 4,
                "avg_session_duration_min": 12, "milestones_hit": 1,
                "social_actions": 1, "age": 35, "fitness_routine": "cardio",
            },
            {
                "label": "At-Risk User",
                "user_id": "demo_atrisk",
                "w1_sessions": 3, "w2_sessions": 1,
                "total_sessions_30d": 4, "total_active_days_30d": 4,
                "max_streak_30d": 2, "workouts_logged": 1,
                "avg_session_duration_min": 6, "milestones_hit": 0,
                "social_actions": 0, "age": 42, "fitness_routine": "mixed",
            },
        ]
    }