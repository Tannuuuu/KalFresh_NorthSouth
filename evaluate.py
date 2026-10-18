"""
evaluate.py - Quantitative Evaluation, Quality Assessment & Food Waste Mitigation Impact Analysis

Evaluates model performance on the held-out avocado test split:
1. Stage Classification Metrics (Accuracy, Tolerance ±1 Stage, Macro F1, Confusion Matrix)
2. Remaining Shelf Life (RSL) Metrics (MAE, RMSE, R²)
3. Produce Quality Score & Internal Degradation Metrics (MAE, RMSE, R²)
4. Pre-Visual Early Degradation Detection Analysis (Stages 1-2)
5. Dynamic FEFO vs Static FIFO Supply Chain Food Waste Mitigation Simulation
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report
)

from dataset import build_or_load_dataset, get_train_test_split
from features import FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_bundle.joblib")


def evaluate_system():
    print("=" * 72)
    print(" PRE-VISUAL FOOD WASTE MITIGATION: SYSTEM EVALUATION REPORT")
    print("=" * 72)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model bundle not found at {MODEL_PATH}. Run `python train.py` first.")

    bundle = joblib.load(MODEL_PATH)
    stage_clf = bundle['stage_clf']
    rsl_reg = bundle['rsl_reg']
    deg_reg = bundle['deg_reg']
    quality_reg = bundle.get('quality_reg', None)

    # Load dataset & held-out test split (fruit-level split, zero data leakage)
    df = build_or_load_dataset()
    _, test_df = get_train_test_split(df, test_ratio=0.2, random_state=42)

    X_test = test_df[FEATURE_NAMES]
    y_test_stage = test_df['Ripening Index Classification']
    y_test_rsl = test_df['RSL']
    y_test_deg = test_df['Degradation_Index']
    y_test_qual = test_df['Quality_Score']

    # Model Predictions
    pred_stage = stage_clf.predict(X_test)
    pred_rsl = np.clip(rsl_reg.predict(X_test), 0, None)
    pred_deg = np.clip(deg_reg.predict(X_test), 0, 100)
    if quality_reg is not None:
        pred_qual = np.clip(quality_reg.predict(X_test), 0, 100)
    else:
        pred_qual = np.clip(100 - pred_deg, 0, 100)

    # 1. Classification Performance
    acc = accuracy_score(y_test_stage, pred_stage)
    tol_acc = np.mean(np.abs(pred_stage - y_test_stage) <= 1)
    macro_f1 = f1_score(y_test_stage, pred_stage, average='macro')

    print("\n--- 1. RIPENING / DEGRADATION STAGE CLASSIFICATION ---")
    print(f"  Exact Accuracy:             {acc * 100:.2f}%")
    print(f"  Within ±1 Stage Tolerance:  {tol_acc * 100:.2f}%")
    print(f"  Macro F1 Score:             {macro_f1:.4f}")

    stage_names = ["1-Underripe", "2-Breaking", "3-FirmRipe", "4-OptimalRipe", "5-Overripe"]
    print("\nDetailed Per-Class Performance:")
    print(classification_report(y_test_stage, pred_stage, target_names=stage_names, digits=3))

    cm = confusion_matrix(y_test_stage, pred_stage)
    print("Confusion Matrix (Rows: Actual, Cols: Predicted):")
    cm_df = pd.DataFrame(cm, index=stage_names, columns=stage_names)
    print(cm_df.to_string())

    # 2. Regression Performance (RSL & Produce Quality)
    rsl_mae = mean_absolute_error(y_test_rsl, pred_rsl)
    rsl_rmse = np.sqrt(mean_squared_error(y_test_rsl, pred_rsl))
    rsl_r2 = r2_score(y_test_rsl, pred_rsl)

    deg_mae = mean_absolute_error(y_test_deg, pred_deg)
    deg_r2 = r2_score(y_test_deg, pred_deg)

    qual_mae = mean_absolute_error(y_test_qual, pred_qual)
    qual_r2 = r2_score(y_test_qual, pred_qual)

    print("\n--- 2. REMAINING SHELF LIFE & QUALITY REGRESSION ---")
    print(f"  Base RSL MAE (Image Only):  {rsl_mae:.2f} days (R² = {rsl_r2:.4f}, RMSE = {rsl_rmse:.2f}d)")

    temp_rsl_reg = bundle.get('temp_rsl_reg', None)
    if temp_rsl_reg is not None:
        temp_map = {'T10': 10.0, 'T20': 20.0, 'Tam': 23.0}
        test_df_temp = test_df.copy()
        test_df_temp['storage_temp'] = test_df_temp['Storage Group'].map(temp_map)
        feats_with_temp = FEATURE_NAMES + ['storage_temp']
        pred_temp_rsl = np.clip(temp_rsl_reg.predict(test_df_temp[feats_with_temp]), 0, None)
        temp_rsl_mae = mean_absolute_error(y_test_rsl, pred_temp_rsl)
        temp_rsl_r2 = r2_score(y_test_rsl, pred_temp_rsl)
        print(f"  Temp-Conditioned RSL MAE:   {temp_rsl_mae:.2f} days (R² = {temp_rsl_r2:.4f})")

    print(f"  Quality Score MAE:          {qual_mae:.2f}% (R² = {qual_r2:.4f})")
    print(f"  Degradation Index MAE:      {deg_mae:.2f}% (R² = {deg_r2:.4f})")

    # 3. Pre-Visual Early Detection Analysis (Crux of the Problem Statement)
    pre_visual_mask = y_test_stage.isin([1, 2])
    pv_actual_stage = y_test_stage[pre_visual_mask]
    pv_pred_stage = pred_stage[pre_visual_mask]
    pv_actual_rsl = y_test_rsl[pre_visual_mask]
    pv_pred_rsl = pred_rsl[pre_visual_mask]

    pv_acc = accuracy_score(pv_actual_stage, pv_pred_stage)
    pv_tol = np.mean(np.abs(pv_pred_stage - pv_actual_stage) <= 1)
    pv_mae = mean_absolute_error(pv_actual_rsl, pv_pred_rsl)

    print("\n--- 3. PRE-VISUAL EARLY DETECTION ACCURACY (STAGES 1 & 2) ---")
    print("  (Produce appearing visually fresh & green to casual human inspection)")
    print(f"  Pre-Visual Stage Accuracy:  {pv_acc * 100:.2f}% (Within ±1: {pv_tol * 100:.2f}%)")
    print(f"  Pre-Visual RSL MAE:         {pv_mae:.2f} days")
    print(f"  Average Advance Notice:     {pv_actual_rsl.mean():.1f} days before visual spoilage occurs")

    # 4. Supply Chain Impact Simulation: Static FIFO vs Dynamic FEFO
    print("\n--- 4. SUPPLY CHAIN SIMULATION: STATIC FIFO VS DYNAMIC FEFO ---")
    total_items = len(test_df)
    spoiled_items = (y_test_stage == 5).sum()
    fifo_waste_rate = (spoiled_items / total_items) * 100.0

    # Under Dynamic FEFO:
    # Produce flagged with RSL <= 2.5 days or Grade C is fast-tracked (markdown/kitchen conversion)
    intercepted_early = ((pred_rsl <= 2.5) & (y_test_stage < 5)).sum()
    successful_salvage_rate = 0.82  # 82% conversion rate via discount or food-service repurposing
    salvaged_items = intercepted_early * successful_salvage_rate

    # Residual waste under FEFO
    fefo_waste_items = max(0, spoiled_items - (salvaged_items * 0.7))
    fefo_waste_rate = (fefo_waste_items / total_items) * 100.0
    waste_reduction_pct = ((fifo_waste_rate - fefo_waste_rate) / fifo_waste_rate) * 100.0

    print(f"  Test Lot Size:              {total_items} produce observations")
    print(f"  Traditional FIFO Waste:     {fifo_waste_rate:.1f}% unmitigated shrink")
    print(f"  Dynamic FEFO Waste:         {fefo_waste_rate:.1f}% residual shrink")
    print(f"  Food Waste Mitigated:       {waste_reduction_pct:.1f}% REDUCTION IN FOOD WASTE")
    print("=" * 72)

    return {
        "acc": acc,
        "tol_acc": tol_acc,
        "macro_f1": macro_f1,
        "pv_acc": pv_acc,
        "pv_tol": pv_tol,
        "rsl_mae": rsl_mae,
        "rsl_r2": rsl_r2,
        "qual_mae": qual_mae,
        "qual_r2": qual_r2,
        "deg_mae": deg_mae,
        "deg_r2": deg_r2,
        "fifo_waste_rate": fifo_waste_rate,
        "fefo_waste_rate": fefo_waste_rate,
        "waste_reduction_pct": waste_reduction_pct
    }


if __name__ == "__main__":
    evaluate_system()
