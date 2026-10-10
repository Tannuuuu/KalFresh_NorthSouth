"""
features.py - Computer Vision Feature Extraction for Pre-Visual Food Degradation & Produce Quality

Extracts spectral, colorimetric, and micro-textural features from produce images:
- Adaptive Foreground Segmentation: Otsu thresholding + morphology + contour verification
- Color Spaces: Calibrated CIE L*a*b*, HSV, Normalized RGB
- Chlorophyll & Senescence Pigment Indices (Excess Green, Excess Red, NDCI, Browning Ratio)
- Perceptual Color Drift (Delta E from Stage 1 Baseline, Color Decay Index)
- Spatial Necrosis & Decay Heterogeneity (decay_pct, p10_l, p25_l, p90_a, heterogeneity_index)
- Micro-Texture & Surface Roughness (GLCM Contrast, Homogeneity, Energy, Laplacian Variance)
- Pre-Visual Degradation Heatmap Generation
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Calibrated reference baseline for fresh underripe Hass avocado exocarp (Stage 1) in 8-bit OpenCV Lab
REF_LAB_L = 124.4
REF_LAB_A = 116.9
REF_LAB_B = 152.5

FEATURE_NAMES = [
    'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b',
    'norm_r', 'norm_g', 'norm_b',
    'exg', 'exr', 'ndci', 'rg_ratio',
    'h_mean', 'h_std', 's_mean', 's_std', 'v_mean', 'v_std',
    'l_mean', 'l_std', 'a_mean', 'a_std', 'b_mean', 'b_std',
    'chroma', 'delta_e',
    'browning_ratio', 'color_decay_index',
    'decay_pct', 'p10_l', 'p25_l', 'p90_a', 'heterogeneity_index',
    'glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'laplacian_var'
]


def segment_fruit(img: np.ndarray) -> np.ndarray:
    """
    Robustly segment the produce fruit foreground from the background.
    Handles studio neutral backgrounds (white/grey) as well as varied field/retail backgrounds.
    Returns a clean binary mask (uint8, 0 or 255) of fruit pixels.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Check 4 corner areas to determine if background is predominantly light or dark
    corner_samples = [
        blurred[:10, :10],
        blurred[:10, -10:],
        blurred[-10:, :10],
        blurred[-10:, -10:]
    ]
    mean_corner = float(np.mean([np.mean(c) for c in corner_samples]))
    is_bright_bg = mean_corner > 130.0

    if is_bright_bg:
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup to remove noise specks and fill interior crevices
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Extract largest contour to ensure solid fruit exocarp without background noise
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        solid_mask = np.zeros_like(mask)
        cv2.drawContours(solid_mask, [c], -1, 255, -1)
        if solid_mask.sum() > (0.02 * mask.size * 255):
            mask = solid_mask

    # Fallback in case mask is completely empty
    if np.sum(mask > 0) < 100:
        mask = np.ones_like(gray, dtype=np.uint8) * 255

    return mask


def extract_features(img_or_path, target_size=(192, 192)) -> dict:
    """
    Extract comprehensive computer vision features from an image or file path.
    Returns a dictionary of feature name -> float value.
    """
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {img_or_path}")
    else:
        img = img_or_path

    # Resize for fast, consistent multi-scale processing
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    mask = segment_fruit(resized)
    fruit_mask = mask > 0

    fruit_bgr = resized[fruit_mask]
    if len(fruit_bgr) == 0:
        fruit_bgr = resized.reshape(-1, 3)

    # --- 1. RGB Statistics & Normalized Pigment Indices ---
    b = fruit_bgr[:, 0].astype(np.float32)
    g = fruit_bgr[:, 1].astype(np.float32)
    r = fruit_bgr[:, 2].astype(np.float32)

    mean_r, mean_g, mean_b = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
    std_r, std_g, std_b = float(np.std(r)), float(np.std(g)), float(np.std(b))

    total_rgb = mean_r + mean_g + mean_b + 1e-6
    norm_r = mean_r / total_rgb
    norm_g = mean_g / total_rgb
    norm_b = mean_b / total_rgb

    # Excess Green (Chlorophyll Index) & Excess Red (Senescence Browning Index)
    exg = float(2.0 * norm_g - norm_r - norm_b)
    exr = float(1.4 * norm_r - norm_g)
    ndci = float((mean_g - mean_r) / (mean_g + mean_r + 1e-6))
    rg_ratio = float(mean_r / (mean_g + 1e-6))

    # --- 2. HSV Color Space ---
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    fruit_hsv = hsv[fruit_mask].astype(np.float32)
    h_mean = float(np.mean(fruit_hsv[:, 0]))
    h_std = float(np.std(fruit_hsv[:, 0]))
    s_mean = float(np.mean(fruit_hsv[:, 1]))
    s_std = float(np.std(fruit_hsv[:, 1]))
    v_mean = float(np.mean(fruit_hsv[:, 2]))
    v_std = float(np.std(fruit_hsv[:, 2]))

    # --- 3. CIE L*a*b* Perceptual Color Space ---
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    fruit_lab = lab[fruit_mask].astype(np.float32)
    l_channel = fruit_lab[:, 0]
    a_channel = fruit_lab[:, 1]
    b_channel = fruit_lab[:, 2]

    l_mean = float(np.mean(l_channel))
    l_std = float(np.std(l_channel))
    a_mean = float(np.mean(a_channel))
    a_std = float(np.std(a_channel))
    b_mean = float(np.mean(b_channel))
    b_std = float(np.std(b_channel))

    # Chroma and Total Color Deviation from Fresh Baseline (Delta E)
    chroma = float(np.sqrt(a_mean**2 + b_mean**2))
    delta_e = float(np.sqrt((l_mean - REF_LAB_L)**2 + (a_mean - REF_LAB_A)**2 + (b_mean - REF_LAB_B)**2))

    # Physiological Degradation Ratios
    browning_ratio = float(exr / (exg + 0.5))
    color_decay_index = float((a_mean - REF_LAB_A) / (l_mean + 1e-5))

    # --- 4. Spatial Necrosis, Local Decay & Color Heterogeneity ---
    # Necrotic / decaying pixels: severe darkening (L < 90) or brown pigment shift (a > 123 and exr > 0.13)
    is_decay = (l_channel < 90.0) | ((a_channel > 123.0) & (exr > 0.13))
    decay_pct = float(np.mean(is_decay) * 100.0)
    p10_l = float(np.percentile(l_channel, 10))
    p25_l = float(np.percentile(l_channel, 25))
    p90_a = float(np.percentile(a_channel, 90))
    heterogeneity_index = float((l_std * a_std) / 100.0)

    # --- 5. Micro-Texture & Surface Roughness ---
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    y_idx, x_idx = np.where(fruit_mask)
    if len(y_idx) > 10 and len(x_idx) > 10:
        crop_gray = gray[y_idx.min():y_idx.max() + 1, x_idx.min():x_idx.max() + 1]
    else:
        crop_gray = gray

    # Quantize to 16 gray levels for GLCM computation
    gray_q = (crop_gray // 16).astype(np.uint8)
    glcm = graycomatrix(gray_q, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
    glcm_contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    glcm_homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    glcm_energy = float(graycoprops(glcm, 'energy')[0, 0])

    # Surface roughness (Laplacian variance)
    laplacian_var = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())

    features = {
        'mean_r': mean_r, 'mean_g': mean_g, 'mean_b': mean_b,
        'std_r': std_r, 'std_g': std_g, 'std_b': std_b,
        'norm_r': norm_r, 'norm_g': norm_g, 'norm_b': norm_b,
        'exg': exg, 'exr': exr, 'ndci': ndci, 'rg_ratio': rg_ratio,
        'h_mean': h_mean, 'h_std': h_std,
        's_mean': s_mean, 's_std': s_std,
        'v_mean': v_mean, 'v_std': v_std,
        'l_mean': l_mean, 'l_std': l_std,
        'a_mean': a_mean, 'a_std': a_std,
        'b_mean': b_mean, 'b_std': b_std,
        'chroma': chroma, 'delta_e': delta_e,
        'browning_ratio': browning_ratio,
        'color_decay_index': color_decay_index,
        'decay_pct': decay_pct,
        'p10_l': p10_l,
        'p25_l': p25_l,
        'p90_a': p90_a,
        'heterogeneity_index': heterogeneity_index,
        'glcm_contrast': glcm_contrast,
        'glcm_homogeneity': glcm_homogeneity,
        'glcm_energy': glcm_energy,
        'laplacian_var': laplacian_var
    }
    return features


def generate_degradation_heatmap(img: np.ndarray) -> np.ndarray:
    """
    Generate an interpretability heatmap showing local pre-visual degradation intensity.
    Based on localized chlorophyll breakdown and a* (red/browning) shift inside fruit tissue.
    Returns RGB heatmap overlay image.
    """
    resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    mask = segment_fruit(resized)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)

    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]

    # Pre-visual internal degradation index per pixel:
    # Shift towards positive red/brown (> 116.9) + darkening (< 124.4)
    degradation_map = (a_channel - 116.0) * 2.2 + (125.0 - l_channel) * 1.5
    degradation_map = np.clip(degradation_map, 0, 100)

    # Normalize to 0-255
    norm_deg = cv2.normalize(degradation_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    norm_deg = cv2.bitwise_and(norm_deg, norm_deg, mask=mask)

    # Apply colormap (JET: Blue=Fresh/Intact, Green/Yellow=Breaking, Red=High Degradation)
    heatmap = cv2.applyColorMap(norm_deg, cv2.COLORMAP_JET)

    # Blend with original image inside the mask
    overlay = resized.copy()
    alpha = 0.55
    fruit_pixels = mask > 0
    overlay[fruit_pixels] = cv2.addWeighted(resized[fruit_pixels], 1 - alpha, heatmap[fruit_pixels], alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
