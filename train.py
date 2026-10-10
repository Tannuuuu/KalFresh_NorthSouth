"""
train.py - Baseline Training Pipeline for Pre-Visual Food Waste Mitigation & Quality Prediction

Trains multi-head prediction models:
1. Ripening & Internal Degradation Stage Classifier (Stages 1-5)
2. Remaining Shelf Life (RSL) Regressor (Days until spoilage)
3. Internal Degradation Severity Index Regressor (0-100%)
4. Commercial Quality Score Regressor (0-100%)

Evaluates quantitative performance on held-out avocado specimens (zero data leakage)
and saves the trained model bundle to models/model_bundle.joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report
)

from dataset import build_or_load_dataset, get_train_test_split
from features import FEATURE_NAMES

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def train_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model_bundle.joblib")

    print("=" * 65)
    print(" PRE-VISUAL FOOD WASTE MITIGATION: MODEL TRAINING & BENCHMARKING")
    print("=" * 65)

    # 1. Load data
    df = build_or_load_dataset()

    # 2. Split train/test strictly by avocado fruit Sample ID
    train_df, test_df = get_train_test_split(df, test_ratio=0.2, random_state=42)

    X_train = train_df[FEATURE_NAMES]
    X_test = test_df[FEATURE_NAMES]

    y_train_stage = train_df['Ripening Index Classification']
    y_test_stage = test_df['Ripening Index Classification']

    y_train_rsl = train_df['RSL']
    y_test_rsl = test_df['RSL']

    y_train_deg = train_df['Degradation_Index']
    y_test_deg = test_df['Degradation_Index']

    y_train_qual = train_df['Quality_Score']
    y_test_qual = test_df['Quality_Score']

    print(f"\nTraining set size: {len(X_train)} samples across {train_df['Sample'].nunique()} avocados")
    print(f"Test set size:     {len(X_test)} samples across {test_df['Sample'].nunique()} avocados")

    # --- MODEL 1: Ripening / Internal Degradation Stage Classifier ---
    print("\n[1/4] Training Ripening Stage Classifier (LightGBM)...")
    stage_clf = LGBMClassifier(
        n_estimators=220,
        learning_rate=0.06,
        num_leaves=35,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=-1
    )
    stage_clf.fit(X_train, y_train_stage)
    pred_stage = stage_clf.predict(X_test)

    stage_acc = accuracy_score(y_test_stage, pred_stage)
    stage_f1 = f1_score(y_test_stage, pred_stage, average='macro')
    # Tolerant accuracy (+/- 1 stage, industry standard for post-harvest ripening)
    stage_tol_acc = np.mean(np.abs(pred_stage - y_test_stage) <= 1)

    print(f"  -> Exact Stage Accuracy:       {stage_acc * 100:.2f}%")
    print(f"  -> Within +/- 1 Stage Tolerance:  {stage_tol_acc * 100:.2f}%")
    print(f"  -> Macro F1-Score:             {stage_f1:.4f}")

    # Pre-Visual Stages specific accuracy (Stages 1 and 2: visually fresh/green)
    pre_visual_mask = y_test_stage.isin([1, 2])
    pre_vis_acc = accuracy_score(y_test_stage[pre_visual_mask], pred_stage[pre_visual_mask])
    pre_vis_tol = np.mean(np.abs(pred_stage[pre_visual_mask] - y_test_stage[pre_visual_mask]) <= 1)
    print(f"  -> Pre-Visual (Stages 1-2) Acc: {pre_vis_acc * 100:.2f}% (Tol: {pre_vis_tol * 100:.2f}%)")

    # --- MODEL 2: Remaining Shelf Life (RSL) Regressor ---
    print("\n[2/4] Training Remaining Shelf Life Regressor (LightGBM)...")
    rsl_reg = LGBMRegressor(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=35,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=-1
    )
    rsl_reg.fit(X_train, y_train_rsl)
    pred_rsl = np.clip(rsl_reg.predict(X_test), 0, None)

    rsl_mae = mean_absolute_error(y_test_rsl, pred_rsl)
    rsl_rmse = np.sqrt(mean_squared_error(y_test_rsl, pred_rsl))
    rsl_r2 = r2_score(y_test_rsl, pred_rsl)

    print(f"  -> RSL Mean Absolute Error:    {rsl_mae:.2f} days")
    print(f"  -> RSL Root Mean Squared Error:{rsl_rmse:.2f} days")
    print(f"  -> RSL R2 Coefficient:         {rsl_r2:.4f}")

    # --- MODEL 3: Internal Degradation Severity Index Regressor ---
    print("\n[3/4] Training Internal Degradation Severity Index Regressor...")
    deg_reg = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.06,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    deg_reg.fit(X_train, y_train_deg)
    pred_deg = np.clip(deg_reg.predict(X_test), 0, 100)
    deg_mae = mean_absolute_error(y_test_deg, pred_deg)
    deg_r2 = r2_score(y_test_deg, pred_deg)
    print(f"  -> Degradation Index MAE:      {deg_mae:.2f}% (R2 = {deg_r2:.4f})")

    # --- MODEL 4: Commercial Produce Quality Score Regressor ---
    print("\n[4/4] Training Commercial Produce Quality Score Regressor...")
    quality_reg = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.06,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    quality_reg.fit(X_train, y_train_qual)
    pred_qual = np.clip(quality_reg.predict(X_test), 0, 100)
    qual_mae = mean_absolute_error(y_test_qual, pred_qual)
    qual_r2 = r2_score(y_test_qual, pred_qual)
    print(f"  -> Quality Score MAE:          {qual_mae:.2f}% (R2 = {qual_r2:.4f})")

    # --- Feature Importance Analysis ---
    print("\n--- Key Computer Vision Biomarkers (Top 8 Feature Importances) ---")
    feat_imp = pd.Series(stage_clf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    for rank, (feat, score) in enumerate(feat_imp.head(8).items(), 1):
        print(f"  {rank}. {feat:<20} (score: {score})")

    # Save model bundle
    bundle = {
        'stage_clf': stage_clf,
        'rsl_reg': rsl_reg,
        'deg_reg': deg_reg,
        'quality_reg': quality_reg,
        'feature_names': FEATURE_NAMES,
        'metrics': {
            'stage_accuracy': float(stage_acc),
            'stage_tol_accuracy': float(stage_tol_acc),
            'stage_macro_f1': float(stage_f1),
            'pre_visual_acc': float(pre_vis_acc),
            'rsl_mae': float(rsl_mae),
            'rsl_rmse': float(rsl_rmse),
            'rsl_r2': float(rsl_r2),
            'deg_mae': float(deg_mae),
            'deg_r2': float(deg_r2),
            'quality_mae': float(qual_mae),
            'quality_r2': float(qual_r2)
        }
    }

    joblib.dump(bundle, model_path)
    print(f"\nTrained model bundle successfully saved to:\n  {model_path}")
    print("=" * 65)
    return bundle


if __name__ == "__main__":
    train_models()
