"""
predict.py - Standalone Computer Vision Inference for Pre-Visual Food Waste Mitigation & Quality Prediction

Performs inference on a produce image:
- Extracts CV colorimetric, spectral, and micro-textural features
- Predicts Ripening / Degradation Stage (1-5)
- Predicts Remaining Shelf Life (RSL in Days)
- Predicts Internal Degradation Severity Index (0-100%)
- Predicts Commercial Produce Quality Score (0-100%) and Grade (Grade A, B, C, Reject)
- Issues Pre-Visual Spoilage Alerts before human-visible decay appears
- Generates Dynamic FEFO (First-Expired, First-Out) Supply Chain Routing Actions
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from features import extract_features, FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_bundle.joblib")

STAGE_METADATA = {
    1: {
        "name": "Underripe (Hard/Intact)",
        "visual_appearance": "Dark green, firm, intact exocarp",
        "internal_state": "Intact cellular walls, high turgor pressure, negligible degradation",
        "pre_visual_alert": "NORMAL: No internal degradation detected"
    },
    2: {
        "name": "Breaking (Pre-Visual Degradation)",
        "visual_appearance": "Appears green/fresh to casual inspection",
        "internal_state": "Enzymatic pectin breakdown initiated; chlorophyll degradation underway",
        "pre_visual_alert": "ALERT: Pre-visual internal softening initiated before overt browning"
    },
    3: {
        "name": "Firm-Ripe (Stage 1)",
        "visual_appearance": "Slight hue transition, predominantly green-olive",
        "internal_state": "Active cell wall breakdown, early edible state",
        "pre_visual_alert": "ACTIVE TRANSITION: Approaching peak edible maturity"
    },
    4: {
        "name": "Optimal Ripe (Stage 2)",
        "visual_appearance": "Dark brown-green, yields to gentle pressure",
        "internal_state": "Peak organoleptic quality, maximum sensory appeal",
        "pre_visual_alert": "PEAK QUALITY: Must be sold/consumed within 24-48 hours"
    },
    5: {
        "name": "Overripe / Spoilage",
        "visual_appearance": "Blackened skin, overt visual senescence",
        "internal_state": "Internal structural collapse, vascular browning, rancidity risk",
        "pre_visual_alert": "SPOILED: Product unsellable, high waste risk"
    }
}


def compute_quality_grade(stage: int, rsl_days: float, quality_score: float, decay_pct: float = 0.0) -> dict:
    """
    Calibrate commercial produce quality grade from stage, RSL, quality score, and localized surface necrosis.
    """
    if stage == 5 or quality_score < 25.0 or rsl_days <= 0.5 or decay_pct >= 35.0:
        return {
            "grade": "Reject (Bio-Divert)",
            "tier_code": "REJECT",
            "market_value": "0% (Write-off)",
            "summary": "Senescent/decayed with significant necrotic breakdown. Unsafe/unfit for retail sale."
        }
    elif stage == 4 or quality_score < 45.0 or rsl_days <= 2.0 or decay_pct >= 18.0:
        return {
            "grade": "Grade C (Markdown / Deli Repurpose)",
            "tier_code": "GRADE_C",
            "market_value": "50% - 70% of standard retail",
            "summary": "Soft ripe / early necrosis. Route immediately to kitchen/deli or flash discount."
        }
    elif stage == 3 or quality_score < 70.0 or rsl_days <= 5.0 or decay_pct >= 8.0:
        return {
            "grade": "Grade B (Standard Retail Display)",
            "tier_code": "GRADE_B",
            "market_value": "85% - 100% of standard retail",
            "summary": "Firm-ripe edible maturity. Ideal for eye-level promotional displays."
        }
    else:
        return {
            "grade": "Grade A (Premium Fresh)",
            "tier_code": "GRADE_A",
            "market_value": "100% Full Retail Value",
            "summary": "Pristine exocarp turgor, intact cellular walls, maximum shelf stability."
        }


def get_fefo_action(rsl_days: float, stage: int, quality_tier: str) -> dict:
    """
    Map predicted RSL, stage, and quality tier to dynamic FEFO supply chain actions.
    """
    if quality_tier == "REJECT" or stage == 5 or rsl_days <= 0.5:
        return {
            "tier": "Decommission / Bio-Divert",
            "action": "Immediate removal from retail display",
            "discount_percent": 100,
            "routing": "Divert to composting or anaerobic digestion to prevent spore transmission",
            "urgency": "Immediate (0 hours)"
        }
    elif quality_tier == "GRADE_C" or rsl_days <= 2.0 or stage == 4:
        return {
            "tier": "Critical Shelf-Life (Flash Markdown / Kitchen Repurposing)",
            "action": "Apply dynamic 30% - 50% discount OR route directly to deli/food service",
            "discount_percent": 40,
            "routing": "Repurpose into processed goods (e.g., guacamole, prepared bowls) before spoilage",
            "urgency": "High (within 12-24 hours)"
        }
    elif quality_tier == "GRADE_B" or rsl_days <= 5.0 or stage == 3:
        return {
            "tier": "Priority Sale (FEFO Fast-Track)",
            "action": "Move to front-of-store eye-level promotional displays with 15% incentive",
            "discount_percent": 15,
            "routing": "Fast-track retail floor placement ahead of older FIFO lots",
            "urgency": "Medium (within 48-72 hours)"
        }
    else:
        return {
            "tier": "Standard Retail Inventory",
            "action": "Maintain at standard price; standard shelf storage",
            "discount_percent": 0,
            "routing": "Normal retail display or cool storage retention",
            "urgency": "Low (> 5 days buffer)"
        }


def predict(image_path: str, model_bundle_path: str = MODEL_PATH) -> dict:
    """
    Run full prediction pipeline on a single image.
    """
    if not os.path.exists(model_bundle_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_bundle_path}. Please run `python train.py` first."
        )

    bundle = joblib.load(model_bundle_path)
    stage_clf = bundle['stage_clf']
    rsl_reg = bundle['rsl_reg']
    deg_reg = bundle['deg_reg']
    quality_reg = bundle.get('quality_reg', None)

    # Extract CV features
    feats_dict = extract_features(image_path)
    feat_df = pd.DataFrame([feats_dict])[FEATURE_NAMES]

    # Predict Stage
    pred_stage = int(stage_clf.predict(feat_df)[0])
    stage_probs = stage_clf.predict_proba(feat_df)[0]
    stage_conf = float(np.max(stage_probs) * 100.0)

    # Predict Remaining Shelf Life
    pred_rsl = float(np.clip(rsl_reg.predict(feat_df)[0], 0.0, None))

    # Predict Internal Degradation Index
    pred_deg = float(np.clip(deg_reg.predict(feat_df)[0], 0.0, 100.0))

    # Predict Commercial Quality Score
    if quality_reg is not None:
        pred_qual = float(np.clip(quality_reg.predict(feat_df)[0], 0.0, 100.0))
    else:
        pred_qual = float(np.clip(100.0 - pred_deg, 0.0, 100.0))

    decay_pct = float(feats_dict.get('decay_pct', 0.0))

    # Post-harvest pathological override: If surface necrosis / rot exceeds 35%,
    # product is objectively in advanced senescent collapse regardless of localized green patches
    if decay_pct >= 50.0:
        pred_stage = 5
        pred_rsl = 0.0
        pred_qual = min(pred_qual, 10.0)
        pred_deg = max(pred_deg, 90.0)
    elif decay_pct >= 30.0:
        pred_stage = max(pred_stage, 4)
        pred_rsl = min(pred_rsl, 1.5)
        pred_qual = min(pred_qual, 30.0)
        pred_deg = max(pred_deg, 75.0)

    stage_info = STAGE_METADATA.get(pred_stage, {})
    quality_info = compute_quality_grade(pred_stage, pred_rsl, pred_qual, decay_pct)
    fefo_action = get_fefo_action(pred_rsl, pred_stage, quality_info['tier_code'])

    # Pre-Visual Detection flag:
    # Triggered when produce looks outwardly fresh (Stage 1 or 2) but has internal degradation
    is_pre_visual = (pred_stage in [1, 2]) and (pred_deg >= 22.0 or pred_rsl < 8.0)

    result = {
        "image_path": image_path,
        "predicted_stage": pred_stage,
        "stage_name": stage_info.get("name", "Unknown"),
        "confidence_pct": round(stage_conf, 1),
        "quality_score_pct": round(pred_qual, 1),
        "quality_grade": quality_info["grade"],
        "quality_tier_code": quality_info["tier_code"],
        "quality_summary": quality_info["summary"],
        "market_value_tier": quality_info["market_value"],
        "remaining_shelf_life_days": round(pred_rsl, 1),
        "internal_degradation_index_pct": round(pred_deg, 1),
        "surface_decay_pct": round(decay_pct, 1),
        "pre_visual_alert": stage_info.get("pre_visual_alert", ""),
        "is_pre_visual_degradation": bool(is_pre_visual),
        "visual_appearance": stage_info.get("visual_appearance", ""),
        "internal_biological_state": stage_info.get("internal_state", ""),
        "fefo_recommendation": fefo_action,
        "cv_biomarkers": {
            "cie_lightness_L": round(feats_dict['l_mean'], 1),
            "cie_chromaticity_a": round(feats_dict['a_mean'], 1),
            "delta_e_color_drift": round(feats_dict['delta_e'], 1),
            "excess_green_chlorophyll": round(feats_dict['exg'], 3),
            "excess_red_browning": round(feats_dict['exr'], 3),
            "browning_ratio": round(feats_dict['browning_ratio'], 3),
            "surface_necrosis_decay_pct": round(decay_pct, 1),
            "texture_energy": round(feats_dict['glcm_energy'], 4),
            "surface_roughness_laplacian": round(feats_dict['laplacian_var'], 1)
        }
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Predict Produce Quality, Internal Degradation & Remaining Shelf Life"
    )
    parser.add_argument("--image", type=str, help="Path to input produce image")
    parser.add_argument("--sample", type=int, help="Sample avocado ID from dataset to test")
    parser.add_argument("--json", action="store_true", help="Output results in raw JSON format")
    args = parser.parse_args()

    image_path = args.image

    # If --sample is passed instead of direct path, locate an image from that sample
    if image_path is None and args.sample is not None:
        dataset_dir = os.path.join(os.path.dirname(__file__), "Hass Avocado Ripening Photographic Dataset", "Avocado Ripening Dataset")
        files = [f for f in os.listdir(dataset_dir) if f"_{args.sample:03d}_" in f and f.endswith(".jpg")]
        if not files:
            print(f"Error: No images found for sample {args.sample}")
            sys.exit(1)
        image_path = os.path.join(dataset_dir, sorted(files)[0])
        print(f"Testing sample {args.sample}: using image {image_path}")

    if image_path is None:
        # Default fallback to first sample image in dataset
        dataset_dir = os.path.join(os.path.dirname(__file__), "Hass Avocado Ripening Photographic Dataset", "Avocado Ripening Dataset")
        if os.path.exists(dataset_dir):
            files = [f for f in os.listdir(dataset_dir) if f.endswith(".jpg")]
            if files:
                image_path = os.path.join(dataset_dir, files[0])
                print(f"No image specified. Using default sample image:\n  {image_path}\n")

    if not image_path or not os.path.exists(image_path):
        print("Error: Please provide a valid image path via --image <path>.")
        sys.exit(1)

    res = predict(image_path)

    if args.json:
        print(json.dumps(res, indent=2))
        return

    print("=" * 68)
    print(" PRE-VISUAL FOOD QUALITY & SHELF-LIFE INFERENCE REPORT")
    print("=" * 68)
    print(f"Target Image:      {os.path.basename(res['image_path'])}")
    print(f"Produce Quality:   {res['quality_grade']} (Score: {res['quality_score_pct']} / 100)")
    print(f"Commercial Value:  {res['market_value_tier']}")
    print(f"Ripening Stage:    Stage {res['predicted_stage']} - {res['stage_name']} ({res['confidence_pct']}% conf)")
    print(f"Remaining Life:    {res['remaining_shelf_life_days']} DAYS remaining before spoilage")
    print(f"Degradation Index: {res['internal_degradation_index_pct']}% internal cellular breakdown")
    print(f"Surface Necrosis:  {res['surface_decay_pct']}% damaged/decaying peel")
    print("-" * 68)
    print(f"Quality Summary:   {res['quality_summary']}")
    print(f"Pre-Visual Alert:  {res['pre_visual_alert']}")
    print(f"Visual Appearance: {res['visual_appearance']}")
    print(f"Internal Biology:  {res['internal_biological_state']}")
    print("-" * 68)
    print(" DYNAMIC FEFO (FIRST-EXPIRED, FIRST-OUT) INVENTORY ROUTING:")
    fefo = res['fefo_recommendation']
    print(f"  Action Tier:     {fefo['tier']}")
    print(f"  Directive:       {fefo['action']}")
    print(f"  Dynamic Pricing: {fefo['discount_percent']}% markdown incentive")
    print(f"  Logistics Route: {fefo['routing']}")
    print(f"  Execution SLA:   {fefo['urgency']}")
    print("-" * 68)
    print(" KEY CV BIOMARKERS:")
    for k, v in res['cv_biomarkers'].items():
        print(f"  * {k:<30}: {v}")
    print("=" * 68)


if __name__ == "__main__":
    main()
