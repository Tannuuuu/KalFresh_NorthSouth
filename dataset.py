"""
dataset.py - Dataset Ingestion, RSL Ground Truth Generation, and Feature Extraction

Loads the Hass Avocado Ripening Photographic Dataset:
- Computes longitudinal Remaining Shelf Life (RSL in days) for each fruit trajectory
- Computes Internal Degradation Severity Index (%) and Commercial Quality Score (0-100%)
- Extracts calibrated computer vision features using multithreaded processing
- Performs fruit-level (Sample-based) train/test splitting to prevent data leakage
- Caches extracted features for instantaneous reloading
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

from features import extract_features, FEATURE_NAMES

DATASET_ROOT = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(DATASET_ROOT, "Hass Avocado Ripening Photographic Dataset", "Avocado Ripening Dataset.xlsx")
IMAGE_DIR = os.path.join(DATASET_ROOT, "Hass Avocado Ripening Photographic Dataset", "Avocado Ripening Dataset")
CACHE_DIR = os.path.join(DATASET_ROOT, "data")
CACHE_FILE = os.path.join(CACHE_DIR, "avocado_features.parquet")


def compute_ground_truth_rsl(excel_path: str = EXCEL_PATH) -> pd.DataFrame:
    """
    Load dataset metadata and compute longitudinal Remaining Shelf Life (RSL).
    RSL = Days remaining until the specific fruit reached Stage 5 (Overripe/Decay).
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel metadata not found at {excel_path}")

    df = pd.read_excel(excel_path)

    # Clean string columns
    df['File Name'] = df['File Name'].astype(str).str.strip()
    df['Storage Group'] = df['Storage Group'].astype(str).str.strip()

    # Find the earliest day each fruit sample reached Stage 5 (Overripe/Spoilage)
    stage5_onset = (
        df[df['Ripening Index Classification'] == 5]
        .groupby(['Storage Group', 'Sample'])['Day of Experiment']
        .min()
        .reset_index()
    )
    stage5_onset.rename(columns={'Day of Experiment': 'Onset_Day_5'}, inplace=True)

    # Merge onset day back to all observations of those samples
    df_merged = df.merge(stage5_onset, on=['Storage Group', 'Sample'], how='inner')

    # Compute Remaining Shelf Life in days
    df_merged['RSL'] = (df_merged['Onset_Day_5'] - df_merged['Day of Experiment']).clip(lower=0)

    # Filter only rows whose image files actually exist on disk
    df_merged['image_path'] = df_merged['File Name'].apply(
        lambda fn: os.path.join(IMAGE_DIR, f"{fn}.jpg")
    )
    df_merged = df_merged[df_merged['image_path'].apply(os.path.exists)].copy()
    df_merged.reset_index(drop=True, inplace=True)

    # Add descriptive stage label
    stage_labels = {
        1: "Underripe (Hard/Intact)",
        2: "Breaking (Pre-Visual Softening)",
        3: "Ripe Stage 1 (Firm-Edible)",
        4: "Ripe Stage 2 (Optimal Ready-to-Eat)",
        5: "Overripe (Senescent/Decayed)"
    }
    df_merged['Stage_Label'] = df_merged['Ripening Index Classification'].map(stage_labels)

    # Continuous internal degradation benchmark (% of life elapsed)
    df_merged['Degradation_Index'] = np.where(
        df_merged['Onset_Day_5'] > 0,
        (df_merged['Day of Experiment'] / df_merged['Onset_Day_5'] * 100.0).clip(upper=100.0),
        100.0
    ).round(1)

    # Commercial Produce Quality Score (0 - 100%)
    df_merged['Quality_Score'] = (100.0 - df_merged['Degradation_Index']).clip(lower=0.0, upper=100.0).round(1)

    # Commercial Quality Grade
    quality_grades = {
        1: "Grade A (Premium Fresh)",
        2: "Grade A (Fresh Breaking)",
        3: "Grade B (Standard Retail)",
        4: "Grade C (Markdown / Deli)",
        5: "Reject (Bio-Divert)"
    }
    df_merged['Quality_Grade'] = df_merged['Ripening Index Classification'].map(quality_grades)

    return df_merged


def _extract_single(row_tuple):
    idx, fn, img_path = row_tuple
    try:
        feats = extract_features(img_path)
        return (idx, feats)
    except Exception as e:
        return (idx, None)


def build_or_load_dataset(force_recompute: bool = False, max_samples: int = None) -> pd.DataFrame:
    """
    Builds or loads cached dataset with extracted features.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not force_recompute and os.path.exists(CACHE_FILE):
        print(f"Loading cached dataset from {CACHE_FILE}...")
        df = pd.read_parquet(CACHE_FILE)
        print(f"Loaded {len(df)} records with {len(FEATURE_NAMES)} CV features.")
        return df

    print("Computing ground truth RSL and metadata...")
    meta_df = compute_ground_truth_rsl()

    if max_samples:
        meta_df = meta_df.head(max_samples).copy()

    print(f"Extracting CV features for {len(meta_df)} images across multiple threads...")
    t0 = time.time()

    tasks = [
        (idx, row['File Name'], row['image_path'])
        for idx, row in meta_df.iterrows()
    ]

    extracted_records = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        for idx, feats in executor.map(_extract_single, tasks):
            if feats is not None:
                extracted_records[idx] = feats

    print(f"Feature extraction completed in {time.time() - t0:.1f}s.")

    # Convert features to DataFrame
    feats_rows = [extracted_records[idx] for idx in sorted(extracted_records.keys())]
    feats_df = pd.DataFrame(feats_rows, index=sorted(extracted_records.keys()))

    valid_meta = meta_df.loc[sorted(extracted_records.keys())].copy().reset_index(drop=True)
    feats_df.reset_index(drop=True, inplace=True)

    combined_df = pd.concat([valid_meta, feats_df], axis=1)

    print(f"Saving extracted features to {CACHE_FILE}...")
    combined_df.to_parquet(CACHE_FILE, index=False)
    print(f"Dataset successfully built and cached: {len(combined_df)} records.")
    return combined_df


def get_train_test_split(df: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42):
    """
    Splits dataset at the fruit specimen level ('Sample') so the same individual
    avocado fruit is never present in both training and test sets (zero data leakage).
    """
    unique_samples = df['Sample'].unique()
    rng = np.random.RandomState(random_state)
    test_samples = rng.choice(
        unique_samples,
        size=int(len(unique_samples) * test_ratio),
        replace=False
    )

    train_mask = ~df['Sample'].isin(test_samples)
    test_mask = df['Sample'].isin(test_samples)

    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    print(f"Sample-stratified Split: {len(train_df)} train images ({len(unique_samples) - len(test_samples)} fruits) | {len(test_df)} test images ({len(test_samples)} fruits)")
    return train_df, test_df


if __name__ == "__main__":
    df = build_or_load_dataset(force_recompute=True)
    print("Dataset summary:")
    print(df[['File Name', 'Storage Group', 'Day of Experiment', 'Ripening Index Classification', 'RSL', 'Degradation_Index', 'Quality_Score']].head())
