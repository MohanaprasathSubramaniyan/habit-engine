"""
Synthetic Data Generator — Habit Formation Engine
Generates realistic user engagement data for 30-day onboarding windows.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json
import os

random.seed(42)
np.random.seed(42)

# ── Persona archetypes ──────────────────────────────────────────────────────
PERSONAS = {
    "power_user": {
        "login_freq": (5, 7),       # sessions/week
        "session_dur": (25, 45),    # minutes
        "feature_depth": (7, 12),   # features used
        "churn_prob": 0.04,
        "weight": 0.15,
    },
    "consistent_user": {
        "login_freq": (3, 5),
        "session_dur": (15, 30),
        "feature_depth": (4, 8),
        "churn_prob": 0.18,
        "weight": 0.30,
    },
    "casual_user": {
        "login_freq": (1, 3),
        "session_dur": (5, 15),
        "feature_depth": (2, 5),
        "churn_prob": 0.45,
        "weight": 0.35,
    },
    "at_risk_user": {
        "login_freq": (0, 1),
        "session_dur": (2, 8),
        "feature_depth": (1, 3),
        "churn_prob": 0.78,
        "weight": 0.20,
    },
}

FEATURES = [
    "workout_logged", "goal_set", "streak_maintained", "social_share",
    "nutrition_tracked", "challenge_joined", "coach_messaged",
    "progress_photo", "milestone_reached", "plan_customized",
    "video_watched", "community_post",
]

FITNESS_ROUTINES = ["strength", "cardio", "yoga", "hiit", "mixed", "rehabilitation"]


def generate_user(user_id: int, start_date: datetime) -> dict:
    persona_name = random.choices(
        list(PERSONAS.keys()),
        weights=[p["weight"] for p in PERSONAS.values()],
    )[0]
    persona = PERSONAS[persona_name]

    age = random.randint(18, 65)
    routine = random.choice(FITNESS_ROUTINES)
    churned = random.random() < persona["churn_prob"]

    # Day-level events
    events = []
    streak = 0
    max_streak = 0
    total_sessions = 0
    milestone_days = []

    for day in range(30):
        event_date = start_date + timedelta(days=day)
        sessions_today = max(
            0,
            int(np.random.normal(persona["login_freq"][1] / 7, 0.8)),
        )
        # Decay for at-risk users over time
        if persona_name == "at_risk_user" and day > 7:
            sessions_today = max(0, sessions_today - random.randint(0, 1))

        if churned and day > random.randint(5, 20):
            sessions_today = 0

        if sessions_today > 0:
            streak += 1
            max_streak = max(max_streak, streak)
            total_sessions += sessions_today

            # Features used this day
            n_features = random.randint(*persona["feature_depth"])
            used = random.sample(FEATURES, min(n_features, len(FEATURES)))

            for session in range(sessions_today):
                duration = random.uniform(*persona["session_dur"])
                events.append({
                    "user_id": user_id,
                    "event_date": event_date.strftime("%Y-%m-%d"),
                    "day_of_onboarding": day + 1,
                    "session_duration_min": round(duration, 2),
                    "features_used": json.dumps(used),
                    "n_features_used": len(used),
                    "workout_logged": int("workout_logged" in used),
                    "goal_set": int("goal_set" in used),
                    "streak_day": streak,
                    "social_action": int(
                        "social_share" in used or "community_post" in used
                    ),
                    "coach_interaction": int("coach_messaged" in used),
                })

            # Milestone detection
            if streak in [3, 7, 14, 21, 30]:
                milestone_days.append({"day": day + 1, "milestone": f"{streak}_day_streak"})
            if total_sessions == 10:
                milestone_days.append({"day": day + 1, "milestone": "10th_session"})
        else:
            streak = 0

    # User-level summary
    all_features = []
    total_duration = 0
    total_workouts = 0
    social_actions = 0

    for e in events:
        total_duration += e["session_duration_min"]
        total_workouts += e["workout_logged"]
        social_actions += e["social_action"]

    user_record = {
        "user_id": user_id,
        "persona": persona_name,
        "age": age,
        "fitness_routine": routine,
        "signup_date": start_date.strftime("%Y-%m-%d"),
        "total_sessions_30d": total_sessions,
        "total_active_days_30d": len(set(e["event_date"] for e in events)),
        "avg_session_duration_min": round(total_duration / max(total_sessions, 1), 2),
        "total_duration_min": round(total_duration, 2),
        "max_streak_30d": max_streak,
        "workouts_logged": total_workouts,
        "social_actions": social_actions,
        "milestones_hit": len(milestone_days),
        "milestone_details": json.dumps(milestone_days),
        "churned_30d": int(churned),
        "retained_long_term": int(not churned and persona_name in ["power_user", "consistent_user"]),
    }

    return user_record, events


def generate_dataset(n_users: int = 2000):
    os.makedirs("data/raw", exist_ok=True)

    users, all_events = [], []
    base_date = datetime(2024, 1, 1)

    for uid in range(1, n_users + 1):
        start = base_date + timedelta(days=random.randint(0, 180))
        user, events = generate_user(uid, start)
        users.append(user)
        all_events.extend(events)

        if uid % 500 == 0:
            print(f"  Generated {uid}/{n_users} users...")

    users_df = pd.DataFrame(users)
    events_df = pd.DataFrame(all_events)

    users_df.to_csv("data/raw/users.csv", index=False)
    events_df.to_csv("data/raw/events.csv", index=False)

    print(f"\n✅ Dataset generated:")
    print(f"   Users : {len(users_df):,}")
    print(f"   Events: {len(events_df):,}")
    print(f"   Churn rate: {users_df['churned_30d'].mean():.1%}")
    print(f"   Files saved to data/raw/")

    return users_df, events_df


if __name__ == "__main__":
    generate_dataset(2000)