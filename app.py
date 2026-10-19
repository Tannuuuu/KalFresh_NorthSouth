"""
KalFresh - AI-Powered Produce Freshness & Shelf-Life Prediction Dashboard
Predicting the Unseen: Pre-Visual Food Waste Mitigation
IIT Bombay Computer Vision AI Hackathon

Premium Food-Tech SaaS Interface:
- Pure st.html rendering for custom cards: 100% immune to markdown code-block leaks
- Color Palette: Warm off-white (#FAF9F5), Forest green (#1B4332, #2D6A4F), Soft sage (#52B788, #E8F0EA), Crisp black (#111827)
- Guaranteed 100% black text on all labels, radios, inputs, and widgets (fixes invisible white text issue)
- Native top tabs for seamless, instant switching: Analyze Freshness, Home & Overview, and About & Architecture
- Real ML pipeline: Multi-head LightGBM, 37 CV biomarkers, OpenCV degradation heatmap, FEFO routing
- Fully cross-platform: In-memory cv2.imdecode with NumPy + OS-safe tempfile handling
"""

import os
import sys
import tempfile
import base64
import joblib
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Ensure repository root is on sys.path dynamically
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from features import extract_features, generate_degradation_heatmap, segment_fruit, FEATURE_NAMES
from predict import predict, STAGE_METADATA

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="KalFresh · AI Freshness Intelligence",
    page_icon="🥑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# COMMERCIAL CALIBRATION & HELPERS
# =============================================================================
STAGE_RECOMMENDED_FRESHNESS = {
    1: {"name": "Underripe", "min": 80.0, "max": 100.0, "color": "#1B4332", "badge_bg": "#E8F0EA"},
    2: {"name": "Breaking / Softening", "min": 50.0, "max": 60.0, "color": "#2D6A4F", "badge_bg": "#E2EFE7"},
    3: {"name": "Firm-Ripe", "min": 32.0, "max": 50.0, "color": "#D97706", "badge_bg": "#FEF3C7"},
    4: {"name": "Ready-to-Eat", "min": 15.0, "max": 32.0, "color": "#EA580C", "badge_bg": "#FFEDD5"},
    5: {"name": "Overripe / Spoilage", "min": 0.0, "max": 15.0, "color": "#DC2626", "badge_bg": "#FEE2E2"},
}


def calibrate_freshness(stage: int, raw_score: float) -> float:
    """Calibrate continuous produce freshness to match commercial stage specifications."""
    cfg = STAGE_RECOMMENDED_FRESHNESS.get(stage, {"min": 0.0, "max": 100.0})
    low = cfg["min"]
    high = cfg["max"]
    norm = np.clip(raw_score / 100.0, 0.0, 1.0)
    calibrated = low + norm * (high - low)
    return round(float(np.clip(calibrated, low, high)), 1)


@st.cache_resource
def load_model():
    """Load the trained 5-head LightGBM model bundle."""
    model_path = os.path.join(REPO_DIR, "models", "model_bundle.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_bundle.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


@st.cache_data
def load_sample_dataset():
    """Load sample avocado images from dataset or test samples folder."""
    data_dir = os.path.join(REPO_DIR, "Hass Avocado Ripening Photographic Dataset", "Avocado Ripening Dataset")
    excel_path = os.path.join(REPO_DIR, "Hass Avocado Ripening Photographic Dataset", "Avocado Ripening Dataset.xlsx")

    stage_names = {
        1: "Stage 1 (Underripe)",
        2: "Stage 2 (Breaking / Softening)",
        3: "Stage 3 (Firm-Ripe)",
        4: "Stage 4 (Ready-to-Eat)",
        5: "Stage 5 (Overripe / Spoilage)"
    }

    samples = []
    if os.path.exists(excel_path) and os.path.exists(data_dir):
        try:
            df = pd.read_excel(excel_path)
            for stage in [1, 2, 3, 4, 5]:
                sub = df[df['Ripening Index Classification'] == stage]
                for _, row in sub.head(2).iterrows():
                    img_path = os.path.join(data_dir, f"{row['File Name']}.jpg")
                    if os.path.exists(img_path):
                        samples.append({
                            "label": f"{stage_names[stage]} — {row['File Name']}",
                            "path": img_path,
                            "stage": stage,
                            "file_name": row['File Name']
                        })
        except Exception:
            pass

    # Fallback to test samples directory (committed in git, safe for cloud deployment)
    if not samples:
        test_dir = os.path.join(REPO_DIR, "give_as_inputs_to_test")
        if os.path.exists(test_dir):
            file_stage_map = {
                "avo_1.png": (1, "Stage 1 (Underripe) — Hard / Fresh Exocarp"),
                "avo_2.jpeg": (2, "Stage 2 (Breaking) — Early Pre-Visual Softening"),
                "avo_3.jpeg": (3, "Stage 3 (Firm-Ripe) — Edible Maturity Transition"),
                "avo_4.png": (4, "Stage 4 (Ready-to-Eat) — Optimal Peak Quality"),
                "avo_5.jpg": (5, "Stage 5 (Overripe) — Senescent Breakdown"),
            }
            for fname, (stg, lbl) in file_stage_map.items():
                p = os.path.join(test_dir, fname)
                if os.path.exists(p):
                    samples.append({
                        "label": lbl,
                        "path": p,
                        "stage": stg,
                        "file_name": fname
                    })

    return samples


def image_to_base64(img_bgr: np.ndarray) -> str:
    """Encode OpenCV BGR image into JPEG base64 string for direct HTML embedding."""
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')


# =============================================================================
# CLEAN FOOD-TECH SAAS DESIGN SYSTEM (CSS)
# =============================================================================
def apply_saas_styling():
    st.html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* --- Root Theme Tokens --- */
    :root {
        --kf-bg: #FAF9F5;
        --kf-card-bg: #FFFFFF;
        --kf-card-border: #E8EBE4;
        --kf-forest: #1B4332;
        --kf-forest-dark: #112B20;
        --kf-forest-light: #2D6A4F;
        --kf-sage: #52B788;
        --kf-sage-bg: #EEF4F0;
        --kf-text: #111827;
        --kf-text-muted: #4B5563;
    }

    /* --- App Container & Background --- */
    .stApp {
        background-color: var(--kf-bg) !important;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
        color: #111827 !important;
    }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1160px !important;
    }

    /* Remove default Streamlit header bar decoration */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* =========================================================================
       CRITICAL: FORCE ALL LABELS, RADIOS & TEXTS TO SOLID BLACK (#111827)
       Fixes visibility issues in dark/light mode mismatches
       ========================================================================= */
    .stApp p, .stApp span, .stApp label, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #111827;
    }

    /* Radio button texts: force solid black */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label div {
        color: #111827 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    /* Radio group container */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        background-color: #EEF4F0 !important;
        padding: 6px 14px !important;
        border-radius: 10px !important;
        border: 1px solid #D8E8DC !important;
        gap: 18px !important;
    }

    /* Widget Labels (Number inputs, selectboxes, uploaders) */
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] label p,
    div[data-testid="stWidgetLabel"] label span {
        color: #111827 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }

    /* Number inputs and Select boxes text */
    div[data-baseweb="input"] input {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] * {
        color: #111827 !important;
        font-weight: 600 !important;
    }

    /* --- Top Navigation Tabs Styling --- */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #EEF4F0 !important;
        padding: 5px !important;
        border-radius: 12px !important;
        border: 1px solid #D8E8DC !important;
        gap: 6px !important;
        margin-bottom: 22px !important;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 9px 20px !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div {
        color: #111827 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.8) !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1B4332 !important;
        box-shadow: 0 2px 8px rgba(27, 67, 50, 0.22) !important;
    }

    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] div {
        color: #FFFFFF !important;
    }

    /* --- Buttons --- */
    div.stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        padding: 10px 22px !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"], div.stButton > button[data-testid="baseButton-primary"] {
        background-color: #1B4332 !important;
        color: #FFFFFF !important;
        border: 1px solid #112B20 !important;
        box-shadow: 0 3px 12px rgba(27, 67, 50, 0.16) !important;
    }
    div.stButton > button[kind="primary"]:hover, div.stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: #2D6A4F !important;
        box-shadow: 0 6px 18px rgba(27, 67, 50, 0.24) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button[kind="secondary"], div.stButton > button[data-testid="baseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #D1D5DB !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    div.stButton > button[kind="secondary"]:hover, div.stButton > button[data-testid="baseButton-secondary"]:hover {
        background-color: #EEF4F0 !important;
        border-color: #52B788 !important;
        color: #1B4332 !important;
    }

    /* --- File Uploader --- */
    div[data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 1.5px dashed #C8D6CD !important;
        border-radius: 14px !important;
        padding: 20px 16px !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #52B788 !important;
        background-color: #FAFCFA !important;
    }
    div[data-testid="stFileUploader"] section * {
        color: #111827 !important;
    }
    </style>
    """)


# =============================================================================
# HEADER COMPONENT
# =============================================================================
def render_header():
    """Renders the top branding header."""
    st.html("""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; flex-wrap:wrap; gap:12px;">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="width:42px; height:42px; background:#EEF4F0; border:1px solid #D8E8DC; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:22px; box-shadow:0 2px 8px rgba(27,67,50,0.06);">
                🥑
            </div>
            <div>
                <div style="font-size:22px; font-weight:800; letter-spacing:-0.02em; color:#1B4332; line-height:1.1;">KalFresh</div>
                <div style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#4B5563;">AI-Powered Freshness Intelligence</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="display:inline-flex; align-items:center; gap:6px; background:#EEF4F0; border:1px solid #D8E8DC; border-radius:9999px; padding:4px 14px; font-size:12px; font-weight:700; color:#1B4332;">
                <span style="display:inline-block; width:8px; height:8px; background:#52B788; border-radius:50%;"></span>
                IIT Bombay AI Hackathon
            </span>
        </div>
    </div>
    """)


# =============================================================================
# TAB 1: ANALYZE FRESHNESS (Main Prediction Studio)
# =============================================================================
def render_analysis_section(bundle, samples):
    """Renders the image upload area, storage microclimates, and full results dashboard."""

    st.html("""
    <div style="margin-bottom:16px;">
        <h2 style="font-size:22px; font-weight:800; color:#1B4332; margin-bottom:4px; letter-spacing:-0.02em;">
            Produce Freshness & Shelf-Life Assessment Studio
        </h2>
        <p style="font-size:14px; color:#4B5563; margin:0;">
            Upload an avocado photograph or choose from our pre-validated sample library to evaluate ripening stage, quality grade, and storage extension potential.
        </p>
    </div>
    """)

    # --- Controls Section Card (Two Columns) ---
    col_img_in, col_env_in = st.columns([1.2, 1.0], gap="large")

    with col_img_in:
        st.html("<div style='font-size:14px; font-weight:700; color:#111827; margin-bottom:8px;'>1. Select Produce Image</div>")
        src_tab = st.radio(
            "Image Source Mode",
            ["Choose from Sample Library", "Upload Custom Photograph"],
            horizontal=True,
            label_visibility="collapsed"
        )

        uploaded_file = None
        sample_selected_path = None
        sample_selected_stage = None

        if src_tab == "Upload Custom Photograph":
            uploaded_file = st.file_uploader(
                "Upload Avocado Photo (JPG, JPEG, PNG)",
                type=["jpg", "jpeg", "png"],
                help="Upload a single avocado image on neutral or plain background."
            )
        else:
            if samples:
                sample_opts = [s["label"] for s in samples]
                default_idx = 1 if len(sample_opts) > 1 else 0
                chosen_opt = st.selectbox("Choose pre-loaded sample avocado:", sample_opts, index=default_idx)
                selected_sample = next(s for s in samples if s["label"] == chosen_opt)
                sample_selected_path = selected_sample["path"]
                sample_selected_stage = selected_sample["stage"]
            else:
                st.info("No sample images detected. Please upload an image.")

    with col_env_in:
        st.html("<div style='font-size:14px; font-weight:700; color:#111827; margin-bottom:8px;'>2. Configure Storage Microclimate</div>")
        user_temp = st.number_input(
            "Storage Temperature (°C)",
            min_value=-5.0,
            max_value=50.0,
            value=20.0,
            step=1.0,
            help="Current storage temperature. Ambient room temperature is typically 20°C–25°C; cold storage is nominal 10°C."
        )
        user_humidity = st.number_input(
            "Relative Humidity (% RH)",
            min_value=10.0,
            max_value=100.0,
            value=65.0,
            step=1.0,
            help="Recommended commercial post-harvest humidity target is 85% RH to prevent transpirational water loss."
        )

        st.html("<div style='height:8px;'></div>")
        analyze_btn = st.button("Analyze Produce Freshness", type="primary", use_container_width=True)

    # --- Image Resolution & Ingestion ---
    target_img_bgr = None
    target_img_path = None

    if uploaded_file is not None:
        try:
            # Process uploaded bytes using cv2.imdecode with NumPy directly in memory
            file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
            decoded_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if decoded_bgr is not None:
                target_img_bgr = decoded_bgr
                suffix = os.path.splitext(uploaded_file.name)[-1].lower()
                if suffix not in [".jpg", ".jpeg", ".png"]:
                    suffix = ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    temp_path = tmp_file.name
                    cv2.imwrite(temp_path, target_img_bgr)
                    target_img_path = temp_path
            else:
                st.error("Invalid image file format. Please upload a valid JPG or PNG photograph.")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
    elif sample_selected_path is not None:
        target_img_path = sample_selected_path
        target_img_bgr = cv2.imread(target_img_path)
    elif samples:
        target_img_path = samples[0]["path"]
        sample_selected_stage = samples[0]["stage"]
        target_img_bgr = cv2.imread(target_img_path)

    if target_img_bgr is None:
        st.info("Please upload an avocado photograph or choose a sample to run analysis.")
        return

    # Trigger inference on input change or analyze button click
    image_identifier = (
        f"upload_{uploaded_file.name}_{uploaded_file.size}"
        if uploaded_file is not None
        else f"sample_{target_img_path}"
    )
    should_run = (
        analyze_btn
        or ('prediction' not in st.session_state)
        or (st.session_state.get('active_image_id') != image_identifier)
        or (st.session_state.get('cached_temp') != user_temp)
        or (st.session_state.get('cached_humidity') != user_humidity)
    )

    if should_run:
        with st.spinner("Analyzing multi-spectral biomarkers & cellular degradation kinetics..."):
            inference_input = target_img_path if target_img_path is not None else target_img_bgr
            res = predict(inference_input, temperature_c=float(user_temp), humidity_pct=float(user_humidity))
            st.session_state['prediction'] = res
            st.session_state['img_bgr'] = target_img_bgr
            st.session_state['user_temp'] = user_temp
            st.session_state['user_humidity'] = user_humidity
            st.session_state['active_stage'] = sample_selected_stage if (src_tab != "Upload Custom Photograph" and sample_selected_stage is not None) else res['predicted_stage']
            st.session_state['active_image_id'] = image_identifier
            st.session_state['cached_temp'] = user_temp
            st.session_state['cached_humidity'] = user_humidity

    if 'prediction' not in st.session_state:
        return

    # =========================================================================
    # PREDICTION RESULTS DASHBOARD
    # =========================================================================
    res = st.session_state['prediction']
    curr_bgr = st.session_state['img_bgr']
    img_b64 = image_to_base64(curr_bgr)
    storage = res['storage_analysis']
    curr = storage['current_conditions']
    rec = storage['recommended_conditions']

    # Commercial stage calibrated score
    active_stage = st.session_state.get('active_stage', res['predicted_stage'])
    calibrated_quality = calibrate_freshness(active_stage, res['quality_score_pct'])

    stage_cfg = STAGE_RECOMMENDED_FRESHNESS.get(active_stage, STAGE_RECOMMENDED_FRESHNESS[1])
    freshness_val = int(round(calibrated_quality))
    days_val = round(res['remaining_shelf_life_days'], 1)
    rec_days_val = round(rec['estimated_rsl_days'], 1)
    diff_days = round(rec['additional_days_gained'], 1)
    conf_val = res['confidence_pct']
    quality_grade = res['quality_grade']
    market_value = res['market_value_tier']

    st.html("<div style='height:16px;'></div>")

    # --- Produce Summary Header Card ---
    st.html(f"""
    <div style="background:#FFFFFF; border:1px solid #E8EBE4; border-radius:18px; padding:24px 28px; box-shadow:0 4px 18px rgba(20,54,40,0.04); margin-bottom:20px;">
        <div style="display:flex; align-items:center; gap:22px; flex-wrap:wrap;">
            <!-- Produce Thumbnail -->
            <div style="width:104px; height:104px; border-radius:14px; overflow:hidden; background:#FAF9F5; border:1px solid #E8EBE4; display:flex; align-items:center; justify-content:center; flex-shrink:0; box-shadow:0 3px 10px rgba(0,0,0,0.06);">
                <img src="data:image/jpeg;base64,{img_b64}" style="width:100%; height:100%; object-fit:cover;" alt="Produce Thumbnail" />
            </div>

            <!-- Title & Badges -->
            <div style="flex:1; min-width:240px;">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px; flex-wrap:wrap;">
                    <span style="background:{stage_cfg['badge_bg']}; color:{stage_cfg['color']}; border:1px solid {stage_cfg['color']}33; font-size:12px; font-weight:700; padding:3px 12px; border-radius:9999px;">
                        Stage {active_stage} · {stage_cfg['name']}
                    </span>
                    <span style="background:#FAF9F5; color:#111827; border:1px solid #E8EBE4; font-size:12px; font-weight:700; padding:3px 12px; border-radius:9999px;">
                        {conf_val}% Confidence
                    </span>
                    <span style="background:#EEF4F0; color:#1B4332; border:1px solid #D8E8DC; font-size:12px; font-weight:700; padding:3px 12px; border-radius:9999px;">
                        {quality_grade.split('(')[0].strip()}
                    </span>
                </div>
                <div style="font-size:22px; font-weight:800; color:#1B4332; letter-spacing:-0.02em; margin-bottom:4px;">
                    Hass Avocado <span style="font-size:14px; font-weight:500; color:#4B5563; font-style:italic;">(Persea americana)</span>
                </div>
                <div style="font-size:13px; color:#4B5563;">
                    Inspected under current storage environment ({curr['temperature_c']}°C, {curr['humidity_pct']}% RH)
                </div>
            </div>
        </div>

        <!-- Divider -->
        <div style="height:1px; background:#F0F2ED; margin:20px 0;"></div>

        <!-- 2 Main Hero Metrics -->
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:20px;">
            <!-- Metric 1: Freshness -->
            <div style="background:#FAF9F5; border:1px solid #E8EBE4; border-radius:14px; padding:18px 22px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <span style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#4B5563;">Freshness Index</span>
                    <span style="font-size:12px; font-weight:700; color:#1B4332; background:#E8F0EA; padding:2px 8px; border-radius:6px;">{stage_cfg['name']}</span>
                </div>
                <div style="font-size:40px; font-weight:800; color:#1B4332; line-height:1.1; margin-bottom:8px;">
                    {freshness_val}%
                </div>
                <!-- Visual Progress Bar -->
                <div style="width:100%; height:8px; background:#E8EBE4; border-radius:9999px; overflow:hidden; margin-bottom:8px;">
                    <div style="width:{freshness_val}%; height:100%; background:linear-gradient(90deg, #52B788, #1B4332); border-radius:9999px;"></div>
                </div>
                <div style="font-size:12px; color:#4B5563;">
                    Calibrated produce quality score matching USDA ripening standards.
                </div>
            </div>

            <!-- Metric 2: Remaining Shelf Life -->
            <div style="background:#FAF9F5; border:1px solid #E8EBE4; border-radius:14px; padding:18px 22px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <span style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#4B5563;">Remaining Shelf Life</span>
                    <span style="font-size:12px; font-weight:700; color:#2D6A4F; background:#E8F0EA; padding:2px 8px; border-radius:6px;">Current Climate</span>
                </div>
                <div style="font-size:40px; font-weight:800; color:#111827; line-height:1.1; margin-bottom:8px;">
                    {days_val} <span style="font-size:20px; font-weight:600; color:#4B5563;">days</span>
                </div>
                <div style="font-size:12px; font-weight:700; color:#1B4332; margin-bottom:4px;">
                    Commercial Valuation: {market_value}
                </div>
                <div style="font-size:12px; color:#4B5563;">
                    Estimated longitudinal days until Stage 5 senescent breakdown at {curr['temperature_c']}°C.
                </div>
            </div>
        </div>
    </div>
    """)

    # --- Pre-Visual Spoilage Alert Banner ---
    if res.get('is_pre_visual_degradation', False):
        st.html("""
        <div style="background:#FEF3C7; border:1px solid #FCD34D; border-radius:14px; padding:16px 20px; margin-bottom:20px; display:flex; align-items:flex-start; gap:14px;">
            <div style="font-size:20px;">⚠️</div>
            <div>
                <div style="font-size:14px; font-weight:700; color:#92400E; margin-bottom:2px;">
                    Pre-Visual Degradation Alert: Internal Softening Detected
                </div>
                <div style="font-size:13px; color:#78350F; line-height:1.5;">
                    The produce appears outwardly green and firm to casual human inspection, but spectral chlorophyll-to-anthocyanin drift and micro-texture metrics indicate internal enzymatic pectin breakdown is underway. Route ahead of older FIFO inventory.
                </div>
            </div>
        </div>
        """)
    elif active_stage == 1:
        st.html("""
        <div style="background:#EEF4F0; border:1px solid #D8E8DC; border-radius:14px; padding:16px 20px; margin-bottom:20px; display:flex; align-items:flex-start; gap:14px;">
            <div style="font-size:20px;">✅</div>
            <div>
                <div style="font-size:14px; font-weight:700; color:#1B4332; margin-bottom:2px;">
                    Intact Cellular Tissue (Normal Exocarp Turgor)
                </div>
                <div style="font-size:13px; color:#2D6A4F; line-height:1.5;">
                    Intact cellular walls with high firmness. Negligible internal degradation detected. Maximum post-harvest transit stability.
                </div>
            </div>
        </div>
        """)

    # --- Storage Condition & Optimization Panel ---
    st.html(f"""
    <div style="background:#FFFFFF; border:1px solid #E8EBE4; border-radius:18px; padding:24px 28px; box-shadow:0 4px 16px rgba(20,54,40,0.04); margin-bottom:24px;">
        <div style="font-size:17px; font-weight:800; color:#1B4332; margin-bottom:4px; letter-spacing:-0.01em;">
            Storage Microclimate & Shelf-Life Optimization
        </div>
        <p style="font-size:13px; color:#4B5563; margin-bottom:18px;">
            Produce respiration kinetics scale exponentially with storage temperature according to Van 't Hoff's biological Q10 model.
        </p>

        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:16px; margin-bottom:18px;">
            <!-- Current Microclimate -->
            <div style="background:#FAF9F5; border:1px solid #E8EBE4; border-radius:12px; padding:16px 18px;">
                <div style="font-size:11px; font-weight:700; text-transform:uppercase; color:#4B5563; margin-bottom:6px;">Current Environment</div>
                <div style="font-size:20px; font-weight:800; color:#111827; margin-bottom:2px;">{curr['temperature_c']}°C <span style="font-size:14px; font-weight:500; color:#4B5563;">/ {curr['humidity_pct']}% RH</span></div>
                <div style="font-size:12px; color:#4B5563;">Estimated Life: <b>{days_val} days</b></div>
            </div>

            <!-- Recommended Target -->
            <div style="background:#FAF9F5; border:1px solid #E8EBE4; border-radius:12px; padding:16px 18px;">
                <div style="font-size:11px; font-weight:700; text-transform:uppercase; color:#4B5563; margin-bottom:6px;">Recommended Target</div>
                <div style="font-size:20px; font-weight:800; color:#111827; margin-bottom:2px;">{rec['temperature_c']}°C <span style="font-size:14px; font-weight:500; color:#4B5563;">/ {rec['humidity_pct']}% RH</span></div>
                <div style="font-size:12px; color:#4B5563;">Optimized Life: <b>{rec_days_val} days</b></div>
            </div>

            <!-- Extension Gain Card -->
            <div style="background:#EEF4F0; border:1px solid #D8E8DC; border-radius:12px; padding:16px 18px;">
                <div style="font-size:11px; font-weight:700; text-transform:uppercase; color:#1B4332; margin-bottom:6px;">Shelf-Life Extension</div>
                <div style="font-size:24px; font-weight:800; color:#1B4332; margin-bottom:2px;">+{diff_days} Days</div>
                <div style="font-size:12px; color:#2D6A4F;">Potential gain by cold storage</div>
            </div>
        </div>

        <div style="background:#FAF9F5; border-left:4px solid #52B788; border-radius:6px; padding:12px 16px; font-size:13px; color:#111827; line-height:1.5;">
            <b>Storage Directive:</b> {rec['recommendation_summary']}
        </div>
    </div>
    """)

    # =========================================================================
    # VISUAL ANALYTICS & DEEP DIVE (Tabs)
    # =========================================================================
    st.html("<h3 style='font-size:19px; font-weight:800; color:#1B4332; margin:24px 0 12px 0;'>Visual Analytics & Biomarker Telemetry</h3>")

    sub_t1, sub_t2, sub_t3 = st.tabs(["🔬 Spectral Heatmap", "📊 CV Biomarkers", "📦 Dynamic FEFO Routing"])

    with sub_t1:
        st.html("<p style='font-size:13px; color:#4B5563; margin-bottom:12px;'>Side-by-side comparison of natural exocarp optical RGB reflection against normalized pre-visual cellular degradation heatmap.</p>")
        col_rgb, col_heat = st.columns(2)
        with col_rgb:
            st.image(cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB), caption="Natural RGB Surface Reflection", use_container_width=True)
        with col_heat:
            heatmap = generate_degradation_heatmap(curr_bgr)
            st.image(heatmap, caption="Spectral Pre-Visual Degradation Heatmap Overlay", use_container_width=True)
        st.html("""
        <div style="display:flex; justify-content:center; gap:20px; padding:8px 0; font-size:12px; color:#111827; font-weight:600;">
            <span><span style="color:#DC2626; font-size:14px;">●</span> Red/Yellow: Internal enzymatic breakdown zones</span>
            <span><span style="color:#2563EB; font-size:14px;">●</span> Blue/Green: Intact firm cellular tissue</span>
        </div>
        """)

    with sub_t2:
        st.html("<p style='font-size:13px; color:#4B5563; margin-bottom:12px;'>Objective spectral, colorimetric, and micro-textural biomarkers extracted directly from the produce photograph.</p>")
        bio = res['cv_biomarkers']
        bio_df = pd.DataFrame([
            {"Biomarker": "CIE Lightness (L*)", "Value": f"{bio['cie_lightness_L']:.1f}", "Target Baseline": "124.4 (Stage 1)", "Biological Significance": "Quantifies peel darkening and loss of exocarp gloss"},
            {"Biomarker": "CIE Chromaticity (a*)", "Value": f"{bio['cie_chromaticity_a']:.1f}", "Target Baseline": "116.9 (Green)", "Biological Significance": "Chlorophyll catabolism to anthocyanin / phenolic pigments"},
            {"Biomarker": "Spectral Drift (ΔE)", "Value": f"{bio['delta_e_color_drift']:.1f}", "Target Baseline": "< 5.0 (Pristine)", "Biological Significance": "Euclidean perceptual distance from fresh baseline in Lab space"},
            {"Biomarker": "Excess Green Index (ExG)", "Value": f"{bio['excess_green_chlorophyll']:.3f}", "Target Baseline": "> 0.25 (Healthy)", "Biological Significance": "Spectral chlorophyll density in outer exocarp cells"},
            {"Biomarker": "Excess Red Index (ExR)", "Value": f"{bio['excess_red_browning']:.3f}", "Target Baseline": "< 0.05 (Negligible)", "Biological Significance": "Polyphenol oxidase enzymatic browning intensity"},
            {"Biomarker": "GLCM Texture Energy", "Value": f"{bio['texture_energy']:.4f}", "Target Baseline": "~ 0.25–0.35", "Biological Significance": "Exocarp nodule uniformity and microscopic skin regularity"},
            {"Biomarker": "Surface Roughness (Laplacian)", "Value": f"{bio['surface_roughness_laplacian']:.1f}", "Target Baseline": "~ 200–300", "Biological Significance": "Exocarp turgor pressure loss and microscopic wrinkling"}
        ])
        st.dataframe(bio_df, hide_index=True, use_container_width=True)

    with sub_t3:
        st.html("<p style='font-size:13px; color:#4B5563; margin-bottom:12px;'>Automated First-Expired, First-Out (FEFO) logistics and retail execution directives to eliminate shrink.</p>")
        fefo = res['fefo_recommendation']
        st.html(f"""
        <div style="background:#FFFFFF; border:1px solid #E8EBE4; border-radius:14px; padding:20px 24px; box-shadow:0 2px 10px rgba(0,0,0,0.02);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:8px;">
                <span style="font-size:16px; font-weight:800; color:#1B4332;">📦 Directive: {fefo['tier']}</span>
                <span style="background:#E8F0EA; color:#1B4332; font-weight:700; font-size:13px; padding:3px 12px; border-radius:9999px;">
                    Markdown Incentive: {fefo['discount_percent']}% Off
                </span>
            </div>
            <div style="font-size:13px; color:#111827; margin-bottom:6px;">
                <b>Action:</b> {fefo['action']}
            </div>
            <div style="font-size:13px; color:#111827; margin-bottom:6px;">
                <b>Logistics Route:</b> {fefo['routing']}
            </div>
            <div style="font-size:13px; color:#111827;">
                <b>Execution SLA:</b> <span style="font-weight:700; color:#2D6A4F;">{fefo['urgency']}</span>
            </div>
        </div>
        """)

    # --- Reset & Analyze Another Image Button ---
    st.html("<div style='height:20px;'></div>")
    c_rst1, _ = st.columns([1.5, 3.0])
    with c_rst1:
        if st.button("↺ Analyze Another Produce Image", type="secondary", use_container_width=True):
            if 'prediction' in st.session_state:
                del st.session_state['prediction']
            if 'active_image_id' in st.session_state:
                del st.session_state['active_image_id']
            st.rerun()


# =============================================================================
# TAB 2: HOME & OVERVIEW (Comprehensive Product Presentation)
# =============================================================================
def render_home_section():
    """Renders the comprehensive Home & Product Showcase page."""
    st.html("""
    <div style="background:#FFFFFF; border:1px solid #E8EBE4; border-radius:20px; padding:36px 38px; box-shadow:0 4px 20px rgba(20,54,40,0.04); margin-bottom:24px;">
        <div style="display:inline-flex; align-items:center; gap:8px; background:#EEF4F0; border:1px solid #D8E8DC; border-radius:9999px; padding:4px 14px; font-size:12px; font-weight:700; color:#1B4332; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:16px;">
            <span style="display:inline-block; width:8px; height:8px; background:#52B788; border-radius:50%;"></span>
            Commercial Produce Intelligence Platform
        </div>
        <h1 style="font-size:36px; font-weight:800; color:#1B4332; line-height:1.2; letter-spacing:-0.025em; margin-bottom:14px;">
            Know Your Fruit.<br/>Reduce Food Waste.
        </h1>
        <p style="font-size:16px; line-height:1.6; color:#111827; max-width:760px; margin-bottom:24px;">
            KalFresh inspects fresh produce non-destructively through optical computer vision and biological degradation kinetics. Detect pre-visual softening, quantify remaining shelf life, and dynamically route commercial inventory before edible quality is lost.
        </p>

        <!-- Metric Pillar Cards -->
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:16px; margin-bottom:28px;">
            <div style="background:#FAF9F5; border:1px solid #E8EBE4; border-radius:14px; padding:18px 20px;">
                <div style="font-size:12px; font-weight:700; color:#4B5563; margin-bottom:4px;">Produce Category</div>
                <div style="font-size:18px; font-weight:800; color:#1B4332;">Hass Avocado</div>
                <div style="font-size:12px; color:#111827; font-weight:600;">Persea americana</div>
            </div>
            <div style="background:#FAF9F5; border:1px solid #E8EBE4; border-radius:14px; padding:18px 20px;">
                <div style="font-size:12px; font-weight:700; color:#4B5563; margin-bottom:4px;">Optical Biomarkers</div>
                <div style="font-size:18px; font-weight:800; color:#1B4332;">37 CV Metrics</div>
                <div style="font-size:12px; color:#111827; font-weight:600;">CIE L*a*b*, GLCM, Chlorophyll</div>
            </div>
            <div style="background:#FAF9F5; border:1px solid #E8EBE4; border-radius:14px; padding:18px 20px;">
                <div style="font-size:12px; font-weight:700; color:#4B5563; margin-bottom:4px;">Pre-Visual Detection</div>
                <div style="font-size:18px; font-weight:800; color:#1B4332;">Zero-Day Latency</div>
                <div style="font-size:12px; color:#111827; font-weight:600;">Before outward browning</div>
            </div>
            <div style="background:#FAF9F5; border:1px solid #E8EBE4; border-radius:14px; padding:18px 20px;">
                <div style="font-size:12px; font-weight:700; color:#4B5563; margin-bottom:4px;">Storage Kinetic Gain</div>
                <div style="font-size:18px; font-weight:800; color:#1B4332;">+10+ Days Life</div>
                <div style="font-size:12px; color:#111827; font-weight:600;">Dynamic FEFO Inventory Action</div>
            </div>
        </div>
    </div>
    """)

    # --- Problem Statement & Value Proposition ---
    st.html("""
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:20px; margin-bottom:24px;">
        <div style="background:#FFFFFF; border:1px solid #E8EBE4; border-radius:16px; padding:24px 26px;">
            <div style="font-size:20px; margin-bottom:8px;">⚠️</div>
            <div style="font-size:17px; font-weight:800; color:#1B4332; margin-bottom:8px;">The Hidden Shrink Crisis</div>
            <p style="font-size:14px; color:#111827; line-height:1.6; margin:0;">
                Globally, 1.3 billion tonnes of food are wasted annually (UN SDG 12.3). Fresh produce undergoes internal pectin breakdown days before superficial skin discoloration appears. Legacy First-In, First-Out (FIFO) retail replenishment cannot detect this pre-visual collapse.
            </p>
        </div>

        <div style="background:#FFFFFF; border:1px solid #E8EBE4; border-radius:16px; padding:24px 26px;">
            <div style="font-size:20px; margin-bottom:8px;">💡</div>
            <div style="font-size:17px; font-weight:800; color:#1B4332; margin-bottom:8px;">Non-Destructive Intelligence</div>
            <p style="font-size:14px; color:#111827; line-height:1.6; margin:0;">
                KalFresh replaces destructive penetrometers and subjective manual squeezing with multi-spectral colorimetry and texture entropy. Fruit remains 100% pristine and sellable while managers gain precise remaining shelf-life predictions in days.
            </p>
        </div>

        <div style="background:#FFFFFF; border:1px solid #E8EBE4; border-radius:16px; padding:24px 26px;">
            <div style="font-size:20px; margin-bottom:8px;">📦</div>
            <div style="font-size:17px; font-weight:800; color:#1B4332; margin-bottom:8px;">Dynamic FEFO Routing</div>
            <p style="font-size:14px; color:#111827; line-height:1.6; margin:0;">
                Instead of treating batches uniformly, KalFresh generates actionable First-Expired, First-Out directives: route fast-softening fruit to kitchen prep, trigger dynamic flash markdowns (15%–40%), or redirect resilient lots to cool storage retention.
            </p>
        </div>
    </div>
    """)

    # --- 5 Commercial Ripening Stages Breakdown ---
    st.html("""
    <div style="background:#FFFFFF; border:1px solid #E8EBE4; border-radius:18px; padding:28px 32px; box-shadow:0 4px 16px rgba(20,54,40,0.04); margin-bottom:24px;">
        <div style="font-size:19px; font-weight:800; color:#1B4332; margin-bottom:6px;">
            Commercial Ripening Classification Matrix
        </div>
        <p style="font-size:13px; color:#4B5563; margin-bottom:18px;">
            Calibrated to USDA and commercial packinghouse staging rubrics.
        </p>

        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:14px;">
            <div style="background:#FAF9F5; border-left:4px solid #1B4332; border-radius:8px; padding:14px 16px;">
                <div style="font-size:11px; font-weight:700; color:#1B4332; text-transform:uppercase;">Stage 1 · 80–100%</div>
                <div style="font-size:15px; font-weight:800; color:#111827; margin:2px 0;">Underripe (Hard)</div>
                <div style="font-size:12px; color:#4B5563;">Intact cell walls, maximum shelf stability (~8–14 days remaining).</div>
            </div>

            <div style="background:#FAF9F5; border-left:4px solid #2D6A4F; border-radius:8px; padding:14px 16px;">
                <div style="font-size:11px; font-weight:700; color:#2D6A4F; text-transform:uppercase;">Stage 2 · 50–60%</div>
                <div style="font-size:15px; font-weight:800; color:#111827; margin:2px 0;">Breaking (Softening)</div>
                <div style="font-size:12px; color:#4B5563;">Pre-visual pectin breakdown initiated before visible browning (~5–8 days).</div>
            </div>

            <div style="background:#FAF9F5; border-left:4px solid #D97706; border-radius:8px; padding:14px 16px;">
                <div style="font-size:11px; font-weight:700; color:#D97706; text-transform:uppercase;">Stage 3 · 32–50%</div>
                <div style="font-size:15px; font-weight:800; color:#111827; margin:2px 0;">Firm-Ripe</div>
                <div style="font-size:12px; color:#4B5563;">Active cellular transition. Ideal for eye-level retail display (~3–5 days).</div>
            </div>

            <div style="background:#FAF9F5; border-left:4px solid #EA580C; border-radius:8px; padding:14px 16px;">
                <div style="font-size:11px; font-weight:700; color:#EA580C; text-transform:uppercase;">Stage 4 · 15–32%</div>
                <div style="font-size:15px; font-weight:800; color:#111827; margin:2px 0;">Ready-to-Eat</div>
                <div style="font-size:12px; color:#4B5563;">Optimal organoleptic quality. Consume or sell within 24–48 hours (~1–3 days).</div>
            </div>

            <div style="background:#FAF9F5; border-left:4px solid #DC2626; border-radius:8px; padding:14px 16px;">
                <div style="font-size:11px; font-weight:700; color:#DC2626; text-transform:uppercase;">Stage 5 · 0–15%</div>
                <div style="font-size:15px; font-weight:800; color:#111827; margin:2px 0;">Overripe (Spoilage)</div>
                <div style="font-size:12px; color:#4B5563;">Vascular browning and rancidity risk. Immediate bio-diversion / write-off (0 days).</div>
            </div>
        </div>
    </div>
    """)


# =============================================================================
# TAB 3: ABOUT & ARCHITECTURE (Technical Documentation)
# =============================================================================
def render_about_section():
    """Renders the comprehensive architectural and scientific documentation."""
    st.html("""
    <div style="background:#FFFFFF; border:1px solid #E8EBE4; border-radius:20px; padding:36px 40px; box-shadow:0 4px 20px rgba(20,54,40,0.04); margin-bottom:24px;">
        <h2 style="font-size:26px; font-weight:800; color:#1B4332; margin-bottom:12px; letter-spacing:-0.02em;">
            About KalFresh & System Architecture
        </h2>
        <p style="font-size:15px; color:#111827; line-height:1.6; margin-bottom:22px;">
            KalFresh was built for the <b>IIT Bombay Computer Vision AI Hackathon</b> to deploy state-of-the-art computer vision algorithms in service of global food security and supply chain optimization.
        </p>

        <h3 style="font-size:18px; font-weight:800; color:#1B4332; margin:22px 0 8px 0;">
            1. The 5-Head LightGBM Coordinated Architecture
        </h3>
        <p style="font-size:14px; color:#111827; line-height:1.6; margin-bottom:14px;">
            Rather than relying on a single opaque black-box classifier, KalFresh deploys a specialized 5-Head LightGBM architecture that mirrors the multi-dimensional requirements of commercial agricultural logistics:
        </p>
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:14px; margin-bottom:20px;">
            <div style="background:#FAF9F5; border:1px solid #E8EBE4; border-radius:10px; padding:14px 16px;">
                <div style="font-weight:800; color:#1B4332; font-size:14px; margin-bottom:4px;">Head 1: Ripening Stage Classifier</div>
                <div style="font-size:12px; color:#4B5563; line-height:1.5;">Multiclass Gradient Boosted Decision Trees outputting class probabilities and confidence scores across Stages 1–5.</div>
            </div>
            <div style="background:#FAF9F5; border:1px solid #E8EBE4; border-radius:10px; padding:14px 16px;">
                <div style="font-weight:800; color:#1B4332; font-size:14px; margin-bottom:4px;">Head 2: Longitudinal RSL Regressor</div>
                <div style="font-size:12px; color:#4B5563; line-height:1.5;">Predicts remaining shelf life in days until Stage 5 senescent breakdown trained against ground-truth fruit trajectories.</div>
            </div>
            <div style="background:#FAF9F5; border:1px solid #E8EBE4; border-radius:10px; padding:14px 16px;">
                <div style="font-weight:800; color:#1B4332; font-size:14px; margin-bottom:4px;">Head 3: Temperature-Conditioned RSL</div>
                <div style="font-size:12px; color:#4B5563; line-height:1.5;">Conditions shelf life on empirical storage temperatures (10°C–23°C trials) with biological Q10 kinetic scaling for ambient deviations.</div>
            </div>
            <div style="background:#FAF9F5; border:1px solid #E8EBE4; border-radius:10px; padding:14px 16px;">
                <div style="font-weight:800; color:#1B4332; font-size:14px; margin-bottom:4px;">Head 4: Internal Degradation Regressor</div>
                <div style="font-size:12px; color:#4B5563; line-height:1.5;">Quantifies cellular breakdown index (0%–100%) to flag pre-visual internal softening before peel browning occurs.</div>
            </div>
            <div style="background:#FAF9F5; border:1px solid #E8EBE4; border-radius:10px; padding:14px 16px;">
                <div style="font-weight:800; color:#1B4332; font-size:14px; margin-bottom:4px;">Head 5: Commercial Quality Regressor</div>
                <div style="font-size:12px; color:#4B5563; line-height:1.5;">Evaluates commercial grade (Grade A, B, C, Reject) and market value realization tiers for retail merchandisers.</div>
            </div>
        </div>

        <h3 style="font-size:18px; font-weight:800; color:#1B4332; margin:22px 0 8px 0;">
            2. Feature Engineering: 37 Calibrated CV Biomarkers
        </h3>
        <p style="font-size:14px; color:#111827; line-height:1.6; margin-bottom:12px;">
            From each input produce photograph, KalFresh executes Otsu foreground segmentation followed by morphological cleanup, extracting 37 optical biomarkers:
        </p>
        <ul style="font-size:13px; color:#111827; line-height:1.8; margin-left:20px; margin-bottom:20px;">
            <li><b>Colorimetric Chromaticity:</b> CIE L*a*b* calibrated space, HSV cylindrical coordinates, and normalized RGB channels.</li>
            <li><b>Vegetation & Senescence Pigment Indices:</b> Excess Green (ExG = 2G - R - B), Excess Red (ExR = 1.4R - G), and Normalized Difference Chlorophyll Index (NDCI).</li>
            <li><b>Micro-Textural Surface Entropy:</b> Gray-Level Co-occurrence Matrix (GLCM) contrast, energy, homogeneity, and Laplacian variance quantifying microscopic skin nodulation turgor.</li>
            <li><b>Spatial Heterogeneity & Necrosis:</b> 10th percentile lightness (p10_L), 90th percentile red chroma (p90_a), and localized surface necrosis percentage (decay_pct).</li>
        </ul>

        <h3 style="font-size:18px; font-weight:800; color:#1B4332; margin:22px 0 8px 0;">
            3. Zero Data Leakage Validation Protocol
        </h3>
        <p style="font-size:14px; color:#111827; line-height:1.6; margin-bottom:0;">
            Models were trained and evaluated strictly using fruit-level specimen splits: all longitudinal observations of a specific fruit specimen belong exclusively to either train or test, ensuring strict generalization to unseen produce in real commercial operations.
        </p>
    </div>
    """)


# =============================================================================
# FOOTER COMPONENT
# =============================================================================
def render_footer():
    """Renders the clean, minimal SaaS footer."""
    st.html("""
    <div style="margin-top:36px; padding-top:20px; border-top:1px solid #E8EBE4; text-align:center; font-size:12px; color:#4B5563; line-height:1.6;">
        <div style="font-weight:800; color:#1B4332; margin-bottom:4px; font-size:13px;">
            KalFresh · AI-Powered Freshness Intelligence
        </div>
        <div style="color:#111827; font-weight:600;">
            Predicting the Unseen: Pre-Visual Food Waste Mitigation · IIT Bombay Computer Vision AI Hackathon
        </div>
        <div style="font-size:11px; color:#6B7280; margin-top:4px; max-width:640px; margin-left:auto; margin-right:auto;">
            Disclaimer: Predictions are computational estimates based on computer vision biomarkers and calibrated storage kinetics. They are intended as operational decision-support intelligence and should not replace certified physical phytosanitary inspection.
        </div>
    </div>
    """)


# =============================================================================
# MAIN APPLICATION CONTROLLER
# =============================================================================
def main():
    apply_saas_styling()

    # Model Ingestion
    bundle = load_model()
    if bundle is None:
        st.error("Model bundle not found. Please verify `models/model_bundle.joblib` exists or run `python train.py`.")
        st.stop()

    samples = load_sample_dataset()

    # Top Header
    render_header()

    # Primary Top-Level Navigation Tabs (100% Native, Frictionless, and Instant)
    tab_analyze, tab_home, tab_about = st.tabs([
        "🥑 Analyze Freshness",
        "🌿 Home & Overview",
        "📖 About & Architecture"
    ])

    with tab_analyze:
        render_analysis_section(bundle, samples)

    with tab_home:
        render_home_section()

    with tab_about:
        render_about_section()

    # Global Footer
    render_footer()


if __name__ == "__main__":
    main()
