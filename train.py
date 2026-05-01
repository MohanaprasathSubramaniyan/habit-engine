"""
Master Training Script — Run this to train all models end-to-end.
Usage: python train.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate import generate_dataset
from backend.pipeline.feature_engineering import run_pipeline
from backend.models.clustering import train_clustering
from backend.models.prediction import train_models
import pandas as pd
import json

def main():
    print("=" * 60)
    print("  HABIT FORMATION ENGINE — TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Generate data
    print("\n📊 STEP 1: Generating synthetic dataset...")
    generate_dataset(n_users=2000)

    # Step 2: Feature engineering
    print("\n🔧 STEP 2: Feature engineering...")
    feats_raw, feats_scaled, feature_cols = run_pipeline()

    # Step 3: Clustering
    print("\n🔵 STEP 3: Training clustering model...")
    km, labels, profiles, cluster_metrics = train_clustering(feats_scaled)

    # Step 4: Prediction models
    print("\n🤖 STEP 4: Training prediction models...")
    model_metrics = train_models(feats_raw)

    # Summary
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  Cluster Silhouette Score : {cluster_metrics['silhouette_score']:.4f}")
    print(f"  Habit Model ROC-AUC      : {model_metrics['habit_formation']['roc_auc']:.4f}")
    print(f"  Churn Model ROC-AUC      : {model_metrics['churn_prediction']['roc_auc']:.4f}")
    print(f"\n  Cluster breakdown:")
    for p in profiles:
        print(f"    [{p['cluster_id']}] {p['label']:<20} n={p['size']:>4}  "
              f"habit={p['avg_habit_score']:.3f}  churn={p['churn_rate']:.1%}")
    print("\n  All artifacts saved to models/artifacts/")
    print("  Start API: cd backend && uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()