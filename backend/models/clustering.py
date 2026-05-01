"""
Clustering Model — Habit Formation Engine
Segments users by fitness routine consistency using KMeans + silhouette analysis.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import joblib
import json
import os

os.makedirs("models/artifacts", exist_ok=True)

CLUSTER_FEATURES = [
    "consistency_rate",
    "streak_to_days_ratio",
    "workout_density",
    "engagement_velocity",
    "social_ratio",
    "milestone_rate",
    "feature_adoption_score",
    "duration_per_session",
]

CLUSTER_LABELS = {
    0: "Habit Champions",
    1: "Steady Builders",
    2: "Casual Dabblers",
    3: "At-Risk Dropouts",
}


def find_optimal_k(X: np.ndarray, k_range=(2, 8)) -> dict:
    results = {}
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        results[k] = {"silhouette": round(sil, 4), "davies_bouldin": round(db, 4)}
        print(f"   k={k}: silhouette={sil:.4f}, davies_bouldin={db:.4f}")
    return results


def train_clustering(feats_scaled: pd.DataFrame, n_clusters: int = 4):
    print("🔵 Training clustering model...")

    X = feats_scaled[CLUSTER_FEATURES].values

    print("   Sweeping k values...")
    k_results = find_optimal_k(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    cluster_labels = km.fit_predict(X)

    sil = silhouette_score(X, cluster_labels)
    db = davies_bouldin_score(X, cluster_labels)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    feats_scaled = feats_scaled.copy()
    feats_scaled["cluster_id"] = cluster_labels
    feats_scaled["pca_x"] = X_2d[:, 0]
    feats_scaled["pca_y"] = X_2d[:, 1]

    raw_feats = pd.read_csv("data/processed/features_raw.csv")
    raw_feats["cluster_id"] = cluster_labels

    cluster_profiles = []
    for cid in range(n_clusters):
        subset = raw_feats[raw_feats["cluster_id"] == cid]
        profile = {
            "cluster_id": cid,
            "label": CLUSTER_LABELS.get(cid, f"Segment {cid}"),
            "size": int(len(subset)),
            "pct_of_total": round(len(subset) / len(raw_feats) * 100, 1),
            "avg_consistency_rate": round(subset["consistency_rate"].mean(), 3),
            "avg_max_streak": round(subset["max_streak_30d"].mean(), 1),
            "avg_habit_score": round(subset["habit_score"].mean(), 3),
            "churn_rate": round(subset["churned_30d"].mean(), 3),
            "retention_rate": round(subset["retained_long_term"].mean(), 3),
            "avg_sessions_30d": round(subset["total_sessions_30d"].mean(), 1),
            "avg_workouts_logged": round(subset["workouts_logged"].mean(), 1),
            "avg_social_actions": round(subset["social_actions"].mean(), 1),
        }
        cluster_profiles.append(profile)
        print(f"   Cluster {cid} ({profile['label']}): n={profile['size']}, "
              f"habit_score={profile['avg_habit_score']:.3f}, churn={profile['churn_rate']:.1%}")

    metrics = {
        "silhouette_score": round(sil, 4),
        "davies_bouldin_score": round(db, 4),
        "n_clusters": n_clusters,
        "k_sweep": k_results,
    }

    joblib.dump(km, "models/artifacts/kmeans_model.pkl")
    joblib.dump(pca, "models/artifacts/pca_model.pkl")
    joblib.dump(CLUSTER_FEATURES, "models/artifacts/cluster_features.pkl")

    with open("models/artifacts/cluster_profiles.json", "w") as f:
        json.dump(cluster_profiles, f, indent=2)

    with open("models/artifacts/cluster_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    feats_scaled.to_csv("data/processed/features_clustered.csv", index=False)

    print(f"\n✅ Clustering complete — silhouette={sil:.4f}, DB={db:.4f}")
    return km, cluster_labels, cluster_profiles, metrics


def predict_cluster(user_features: dict) -> dict:
    km = joblib.load("models/artifacts/kmeans_model.pkl")
    scaler = joblib.load("models/artifacts/scaler.pkl")
    feature_cols = joblib.load("models/artifacts/feature_cols.pkl")
    cluster_features = joblib.load("models/artifacts/cluster_features.pkl")

    with open("models/artifacts/cluster_profiles.json") as f:
        profiles = json.load(f)

    vec = pd.DataFrame([user_features])
    for col in feature_cols:
        if col not in vec.columns:
            vec[col] = 0

    vec_scaled = scaler.transform(vec[feature_cols])
    vec_df = pd.DataFrame(vec_scaled, columns=feature_cols)

    cluster_vec = vec_df[cluster_features].values
    cluster_id = int(km.predict(cluster_vec)[0])
    distances = km.transform(cluster_vec)[0]
    confidence = round(1 - (distances[cluster_id] / distances.sum()), 3)

    profile = next(p for p in profiles if p["cluster_id"] == cluster_id)

    return {
        "cluster_id": cluster_id,
        "cluster_label": CLUSTER_LABELS.get(cluster_id, f"Segment {cluster_id}"),
        "confidence": confidence,
        "churn_risk": profile["churn_rate"],
        "retention_rate": profile["retention_rate"],
        "avg_habit_score": profile["avg_habit_score"],
    }


if __name__ == "__main__":
    feats_scaled = pd.read_csv("data/processed/features_scaled.csv")
    train_clustering(feats_scaled)