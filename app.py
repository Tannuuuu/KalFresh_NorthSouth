"""
app.py - Avocado Freshness Intelligence Dashboard
Predicting the Unseen: Pre-Visual Food Waste Mitigation
IIT Bombay Computer Vision AI Hackathon

Calibrated Version for Stage-Recommended Freshness:
- Stage 1 (Underripe): 80–100% Freshness
- Stage 2 (Breaking / Softening): 50–60% Freshness
- Stage 3 (Firm-Ripe): 32–50% Freshness
- Stage 4 (Ready-to-Eat): 15–32% Freshness
- Stage 5 (Overripe / Spoilage): 0–15% Freshness

Features:
- Instant, reactive Dark Mode & Light Mode switching with st.rerun (no manual browser refresh needed)
- Robust widget CSS styling for number inputs, selectboxes, and dropzones (fixes dark mode white boxes)
- Exact replica of the user reference card design
- Top card with rounded avocado thumbnail, title, subtitle, and badge
- Two prominent metric columns: Freshness (%) in green and Days remaining (days)
- "Your current storage conditions" with Temperature and Relative Humidity
- "Recommended storage conditions" with 10°C and 85% RH target cards
- "Potential benefit of switching storage conditions" with +X days additional life
- Uses st.html for clean HTML rendering with zero code-block leaks
- Clean input panel and scientific inspection section
"""

import os
import sys
import cv2
import base64
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure repository root is on sys.path even when run directly from Downloads
REPO_DIR = r"c:\Users\RAMYASRI\Desktop\northsouthiitb\KalFresh_NorthSouth"
if os.path.exists(REPO_DIR) and REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from features import extract_features, generate_degradation_heatmap, segment_fruit, FEATURE_NAMES
from predict import predict, STAGE_METADATA

# Page Configuration
st.set_page_config(
    page_title="Avocado Freshness Report",
    page_icon="🥑",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Recommended Freshness Brackets matching Commercial Staging Rubric
STAGE_RECOMMENDED_FRESHNESS = {
    1: {"name": "Underripe", "min": 80.0, "max": 100.0},
    2: {"name": "Breaking / Softening", "min": 50.0, "max": 60.0},
    3: {"name": "Firm-Ripe", "min": 32.0, "max": 50.0},
    4: {"name": "Ready-to-Eat", "min": 15.0, "max": 32.0},
    5: {"name": "Overripe / Spoilage", "min": 0.0, "max": 15.0},
}


def calibrate_freshness(stage: int, raw_score: float) -> float:
    """
    Calibrate continuous produce freshness to match commercial stage specifications:
    - Stage 1 (Underripe): 80–100%
    - Stage 2 (Breaking / Softening): 50–60%
    - Stage 3 (Firm-Ripe): 32–50%
    - Stage 4 (Ready-to-Eat): 15–32%
    - Stage 5 (Overripe / Spoilage): 0–15%
    """
    cfg = STAGE_RECOMMENDED_FRESHNESS.get(stage, {"min": 0.0, "max": 100.0})
    low = cfg["min"]
    high = cfg["max"]
    norm = np.clip(raw_score / 100.0, 0.0, 1.0)
    calibrated = low + norm * (high - low)
    return round(float(np.clip(calibrated, low, high)), 1)


@st.cache_resource
def load_model():
    """Load the trained model bundle."""
    model_path = os.path.join(REPO_DIR, "models", "model_bundle.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), "models", "model_bundle.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


@st.cache_data
def load_sample_dataset():
    """Load sample avocado images from dataset."""
    data_dir = os.path.join(REPO_DIR, "Hass Avocado Ripening Photographic Dataset", "Avocado Ripening Dataset")
    excel_path = os.path.join(REPO_DIR, "Hass Avocado Ripening Photographic Dataset", "Avocado Ripening Dataset.xlsx")
    if not os.path.exists(excel_path) or not os.path.exists(data_dir):
        data_dir = os.path.join(os.path.dirname(__file__), "Hass Avocado Ripening Photographic Dataset", "Avocado Ripening Dataset")
        excel_path = os.path.join(os.path.dirname(__file__), "Hass Avocado Ripening Photographic Dataset", "Avocado Ripening Dataset.xlsx")
    if not os.path.exists(excel_path) or not os.path.exists(data_dir):
        return []

    df = pd.read_excel(excel_path)
    stage_names = {
        1: "Stage 1 (Underripe)",
        2: "Stage 2 (Breaking / Softening)",
        3: "Stage 3 (Firm-Ripe)",
        4: "Stage 4 (Ready-to-Eat)",
        5: "Stage 5 (Overripe / Spoilage)"
    }

    samples = []
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
    return samples


def image_to_base64(img_bgr: np.ndarray) -> str:
    """Encode OpenCV BGR image into JPEG base64 string for direct HTML embedding."""
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')


def _sync_theme_mode():
    """Keep the active theme in sync with the radio widget before rerun."""
    st.session_state["theme_mode"] = st.session_state["theme_radio_selector"]


def main():
    bundle = load_model()
    if bundle is None:
        st.error("Model bundle not found. Please run `python train.py` first.")
        st.stop()

    samples = load_sample_dataset()

    # --- Reactive Theme Mode State Management ---
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "🌙 Dark"

    if "theme_radio_selector" not in st.session_state:
        st.session_state["theme_radio_selector"] = st.session_state["theme_mode"]

    top_col1, top_col2 = st.columns([3.8, 1.4])
    with top_col1:
        st.markdown("<p style='font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#22C55E; margin:6px 0 2px 0;'>KalFresh · Computer Vision AI Hackathon</p>", unsafe_allow_html=True)

    with top_col2:
        st.radio(
            "Theme",
            options=["🌙 Dark", "☀️ Light"],
            horizontal=True,
            key="theme_radio_selector",
            on_change=_sync_theme_mode,
            label_visibility="collapsed"
        )

    is_dark = (st.session_state["theme_mode"] == "🌙 Dark")

    # Complete Design Tokens for Dark & Light
    if is_dark:
        bg_canvas = "#000000"
        panel_bg = "#0B0D11"
        panel_border = "#1E222A"
        report_bg = "#000000"
        report_border = "#1E222A"
        target_card_bg = "#15181F"
        target_card_border = "#232834"
        benefit_card_bg = "#000000"
        benefit_card_border = "#1E222A"
        text_primary = "#FFFFFF"
        text_secondary = "#9CA3AF"
        accent_green = "#22C55E"
        badge_bg = "#062817"
        badge_color = "#22C55E"
        badge_border = "#14532D"
        icon_color = "#9CA3AF"
        input_bg = "#15181F"
        input_border = "#262B36"
        btn_bg = "#15803D"
        btn_hover = "#22C55E"
    else:
        bg_canvas = "#F8FAFC"
        panel_bg = "#FFFFFF"
        panel_border = "#E2E8F0"
        report_bg = "#FFFFFF"
        report_border = "#E2E8F0"
        target_card_bg = "#F1F5F9"
        target_card_border = "#E2E8F0"
        benefit_card_bg = "#F8FAFC"
        benefit_card_border = "#E2E8F0"
        text_primary = "#0F172A"
        text_secondary = "#64748B"
        accent_green = "#16A34A"
        badge_bg = "#DCFCE7"
        badge_color = "#15803D"
        badge_border = "#86EFAC"
        icon_color = "#64748B"
        input_bg = "#F1F5F9"
        input_border = "#CBD5E1"
        btn_bg = "#16A34A"
        btn_hover = "#15803D"

    # Inject theme-aware CSS
    st.markdown(f"""
    <style>
        :root {{
            --app-bg: {bg_canvas};
            --panel-bg: {panel_bg};
            --panel-border: {panel_border};
            --report-bg: {report_bg};
            --report-border: {report_border};
            --target-card-bg: {target_card_bg};
            --target-card-border: {target_card_border};
            --benefit-card-bg: {benefit_card_bg};
            --benefit-card-border: {benefit_card_border};
            --text-primary: {text_primary};
            --text-secondary: {text_secondary};
            --accent-green: {accent_green};
            --input-bg: {input_bg};
            --input-border: {input_border};
            --button-bg: {btn_bg};
            --button-hover: {btn_hover};
        }}

        /* App surfaces */
        html, body, .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {{
            background: var(--app-bg) !important;
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}

        .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 3rem !important;
            max-width: 720px !important;
        }}

        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stCaptionContainer"],
        .stApp label,
        .stApp [data-testid="stWidgetLabel"] p {{
            color: var(--text-primary);
        }}
        .stApp [data-testid="stCaptionContainer"] {{
            color: var(--text-secondary) !important;
        }}

        /* Theme selector */
        div[data-testid="stRadio"] {{
            background: var(--panel-bg);
            padding: 4px 10px;
            border: 1px solid var(--panel-border);
            border-radius: 9999px;
        }}
        div[data-testid="stRadio"] label,
        div[data-testid="stRadio"] label p,
        div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {{
            color: var(--text-primary) !important;
            font-size: 13px !important;
            font-weight: 500 !important;
        }}
        div[data-testid="stRadio"] input {{
            accent-color: var(--accent-green);
        }}

        /* Expanders */
        div[data-testid="stExpander"] {{
            background: var(--panel-bg) !important;
            border: 1px solid var(--panel-border) !important;
            border-radius: 12px !important;
            margin-bottom: 18px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, {'0.25' if is_dark else '0.04'}) !important;
        }}
        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] details summary {{
            background: var(--panel-bg) !important;
            color: var(--text-primary) !important;
            border-radius: 12px !important;
        }}
        div[data-testid="stExpander"] details summary p,
        div[data-testid="stExpander"] details summary span,
        div[data-testid="stExpander"] details summary svg {{
            color: var(--text-primary) !important;
            fill: var(--text-primary) !important;
            font-weight: 600 !important;
        }}
        div[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {{
            color: var(--text-primary);
        }}

        /* Number inputs */
        div[data-testid="stNumberInput"] div[data-baseweb="input"],
        div[data-testid="stNumberInput"] div[data-baseweb="base-input"] {{
            background: var(--input-bg) !important;
            border-color: var(--input-border) !important;
        }}
        div[data-testid="stNumberInput"] input {{
            background: var(--input-bg) !important;
            color: var(--text-primary) !important;
            -webkit-text-fill-color: var(--text-primary) !important;
            border-color: var(--input-border) !important;
        }}
        div[data-testid="stNumberInput"] label p {{
            color: var(--text-primary) !important;
            font-size: 13px !important;
            font-weight: 600 !important;
        }}
        div[data-testid="stNumberInput"] button {{
            background: var(--input-bg) !important;
            color: var(--text-primary) !important;
            border-color: var(--input-border) !important;
        }}
        div[data-testid="stNumberInput"] button svg {{
            fill: var(--text-primary) !important;
            stroke: var(--text-primary) !important;
        }}

        /* Selectbox control */
        div[data-testid="stSelectbox"] label p {{
            color: var(--text-primary) !important;
            font-size: 13px !important;
            font-weight: 600 !important;
        }}
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div {{
            background: var(--input-bg) !important;
            border-color: var(--input-border) !important;
        }}
        div[data-testid="stSelectbox"] [data-baseweb="select"] span,
        div[data-testid="stSelectbox"] [data-baseweb="select"] input {{
            color: var(--text-primary) !important;
            -webkit-text-fill-color: var(--text-primary) !important;
        }}
        div[data-testid="stSelectbox"] [data-baseweb="select"] svg {{
            fill: var(--text-secondary) !important;
            color: var(--text-secondary) !important;
        }}

        /* Dropdown popover menu */
        [data-baseweb="popover"],
        [data-baseweb="popover"] ul[data-baseweb="menu"],
        [data-baseweb="popover"] li[role="option"] {{
            background: var(--panel-bg) !important;
            color: var(--text-primary) !important;
        }}
        [data-baseweb="popover"] li[role="option"] * {{
            color: var(--text-primary) !important;
        }}
        [data-baseweb="popover"] li[role="option"]:hover {{
            background: var(--input-bg) !important;
        }}

        /* File uploader */
        section[data-testid="stFileUploadDropzone"] {{
            background: var(--input-bg) !important;
            border: 1.5px dashed var(--input-border) !important;
            border-radius: 10px !important;
            padding: 14px !important;
        }}
        section[data-testid="stFileUploadDropzone"] p,
        section[data-testid="stFileUploadDropzone"] span,
        section[data-testid="stFileUploadDropzone"] small,
        section[data-testid="stFileUploadDropzone"] button {{
            color: var(--text-primary) !important;
        }}
        div[data-testid="stFileUploaderFile"] {{
            background: var(--target-card-bg) !important;
            border: 1px solid var(--input-border) !important;
            border-radius: 8px !important;
        }}
        div[data-testid="stFileUploaderFile"] p,
        div[data-testid="stFileUploaderFile"] span,
        div[data-testid="stFileUploaderFile"] svg {{
            color: var(--text-primary) !important;
            fill: var(--text-primary) !important;
        }}

        /* Primary Action Button */
        div.stButton > button:first-child {{
            background: var(--button-bg) !important;
            color: #FFFFFF !important;
            height: 44px !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(34, 197, 94, 0.2) !important;
            transition: background-color 0.2s ease !important;
            width: 100% !important;
        }}
        div.stButton > button:first-child:hover {{
            background: var(--button-hover) !important;
            color: #FFFFFF !important;
        }}
        div.stButton > button:first-child p {{
            color: #FFFFFF !important;
        }}

        /* Tabs */
        button[data-baseweb="tab"] {{
            color: var(--text-secondary) !important;
        }}
        button[data-baseweb="tab"] p {{
            color: inherit !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: var(--accent-green) !important;
            border-bottom-color: var(--accent-green) !important;
        }}

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {{
            border-color: var(--panel-border) !important;
        }}

        header[data-testid="stHeader"] {{
            background: transparent !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    # --- Controls Section ---
    with st.expander("📸 Select Image & Configure Storage Conditions", expanded=True):
        col_img_in, col_env_in = st.columns([1.2, 1.0])

        with col_img_in:
            src_tab = st.radio("Image Source", ["Sample Avocado from Dataset", "Upload Custom Photo"], horizontal=True, label_visibility="collapsed")
            uploaded_file = None
            sample_selected_path = None
            sample_selected_stage = None

            if src_tab == "Upload Custom Photo":
                uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
            else:
                if samples:
                    sample_opts = [s["label"] for s in samples]
                    default_idx = 2 if len(sample_opts) > 2 else 0
                    chosen_opt = st.selectbox("Choose sample avocado:", sample_opts, index=default_idx)
                    selected_sample = next(s for s in samples if s["label"] == chosen_opt)
                    sample_selected_path = selected_sample["path"]
                    sample_selected_stage = selected_sample["stage"]

        with col_env_in:
            user_temp = st.number_input("Storage Temperature (°C)", min_value=-5.0, max_value=50.0, value=28.0, step=1.0)
            user_humidity = st.number_input("Relative Humidity (%)", min_value=10.0, max_value=100.0, value=65.0, step=1.0)

        analyze_btn = st.button("Analyze Avocado")

    # Determine image to analyze
    target_img_bgr = None
    target_img_path = None

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        target_img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        temp_path = os.path.join(REPO_DIR, "temp_uploaded.jpg")
        cv2.imwrite(temp_path, target_img_bgr)
        target_img_path = temp_path
    elif sample_selected_path is not None:
        target_img_path = sample_selected_path
        target_img_bgr = cv2.imread(target_img_path)
    elif samples:
        target_img_path = samples[0]["path"]
        sample_selected_stage = samples[0]["stage"]
        target_img_bgr = cv2.imread(target_img_path)

    # Run Prediction or load cached result
    if analyze_btn or 'prediction' not in st.session_state:
        if target_img_bgr is not None and target_img_path is not None:
            res = predict(target_img_path, temperature_c=float(user_temp), humidity_pct=float(user_humidity))
            st.session_state['prediction'] = res
            st.session_state['img_bgr'] = target_img_bgr
            st.session_state['user_temp'] = user_temp
            st.session_state['user_humidity'] = user_humidity
            st.session_state['active_stage'] = sample_selected_stage if (src_tab != "Upload Custom Photo" and sample_selected_stage is not None) else res['predicted_stage']

    if 'prediction' not in st.session_state:
        st.info("Select or upload an avocado image to view the Freshness Report.")
        return

    # Extract Prediction Values
    res = st.session_state['prediction']
    curr_bgr = st.session_state['img_bgr']
    img_b64 = image_to_base64(curr_bgr)
    storage = res['storage_analysis']
    curr = storage['current_conditions']
    rec = storage['recommended_conditions']

    # Stage-calibrated Freshness percentage matching user commercial grading brackets
    active_stage = st.session_state.get('active_stage', res['predicted_stage'])
    calibrated_quality = calibrate_freshness(active_stage, res['quality_score_pct'])

    freshness_val = int(round(calibrated_quality))
    days_val = int(round(res['remaining_shelf_life_days']))
    rec_days_val = int(round(rec['estimated_rsl_days']))
    diff_days = int(round(rec['additional_days_gained']))

    # =========================================================================
    # EXACT AVOCADO FRESHNESS REPORT CARD (Rendered via st.html)
    # =========================================================================
    report_html = f"""
    <div style="background:{report_bg}; border:1px solid {report_border}; border-radius:16px; padding:28px 32px; box-shadow:0 4px 20px rgba(0,0,0,{'0.5' if is_dark else '0.06'}); margin-bottom:24px;">
        
        <!-- Header Row with Avocado Image -->
        <div style="display:flex; align-items:center; gap:22px; padding-bottom:20px;">
            <div style="width:104px; height:104px; border-radius:14px; overflow:hidden; background:#FFFFFF; display:flex; align-items:center; justify-content:center; flex-shrink:0; box-shadow:0 4px 14px rgba(0,0,0,{'0.4' if is_dark else '0.1'}); border:1px solid {report_border};">
                <img src="data:image/jpeg;base64,{img_b64}" style="width:100%; height:100%; object-fit:cover;" />
            </div>
            <div>
                <div style="font-size:24px; font-weight:700; color:{text_primary}; margin-bottom:6px; letter-spacing:-0.01em;">Avocado Freshness Report</div>
                <div style="font-size:14px; color:{text_secondary}; margin-bottom:12px;">Image analysis + current storage conditions</div>
                <span style="display:inline-block; background:{badge_bg}; color:{badge_color}; border:1px solid {badge_border}; font-size:12px; font-weight:600; padding:3px 12px; border-radius:9999px;">
                    Illustrative prediction
                </span>
            </div>
        </div>

        <!-- Metric Values Row -->
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:32px; padding:28px 0;">
            <div>
                <div style="font-size:14px; color:{text_secondary}; margin-bottom:8px;">Freshness</div>
                <div style="font-size:38px; font-weight:700; color:{accent_green}; line-height:1.1; margin-bottom:6px;">{freshness_val}%</div>
                <div style="font-size:13px; color:{text_secondary};">Estimated current quality</div>
            </div>
            <div>
                <div style="font-size:14px; color:{text_secondary}; margin-bottom:8px;">Days remaining</div>
                <div style="font-size:38px; font-weight:700; color:{text_primary}; line-height:1.1; margin-bottom:6px;">{days_val} days</div>
                <div style="font-size:13px; color:{text_secondary};">At your entered conditions</div>
            </div>
        </div>

        <!-- Your Current Storage Conditions -->
        <div style="padding-bottom:28px;">
            <div style="font-size:16px; font-weight:600; color:{text_primary}; margin-bottom:16px;">Your current storage conditions</div>
            <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 0; font-size:15px;">
                <span style="color:{text_primary};">Temperature</span>
                <span style="color:{text_primary}; font-weight:700; font-size:16px;">{int(round(curr['temperature_c']))}°C</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 0; font-size:15px;">
                <span style="color:{text_primary};">Relative humidity</span>
                <span style="color:{text_primary}; font-weight:700; font-size:16px;">{int(round(curr['humidity_pct']))}%</span>
            </div>
            <div style="font-size:12px; color:{text_secondary}; margin-top:8px;">
                Example inputs only, not a prediction for a real avocado.
            </div>
        </div>

        <!-- Recommended Storage Conditions -->
        <div style="padding-bottom:24px;">
            <div style="font-size:16px; font-weight:600; color:{text_primary}; margin-bottom:16px;">Recommended storage conditions</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px;">
                <!-- 10°C Card -->
                <div style="background:{target_card_bg}; border:1px solid {target_card_border}; border-radius:12px; padding:18px 20px;">
                    <div style="margin-bottom:8px;">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/>
                        </svg>
                    </div>
                    <div style="font-size:26px; font-weight:700; color:{text_primary}; margin-bottom:4px; line-height:1.1;">{int(round(rec['temperature_c']))}°C</div>
                    <div style="font-size:13px; color:{text_secondary};">Temperature target</div>
                </div>
                <!-- 85% RH Card -->
                <div style="background:{target_card_bg}; border:1px solid {target_card_border}; border-radius:12px; padding:18px 20px;">
                    <div style="margin-bottom:8px;">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/>
                        </svg>
                    </div>
                    <div style="font-size:26px; font-weight:700; color:{text_primary}; margin-bottom:4px; line-height:1.1;">{int(round(rec['humidity_pct']))}% RH</div>
                    <div style="font-size:13px; color:{text_secondary};">Humidity target</div>
                </div>
            </div>
            <div style="font-size:13px; color:{text_secondary}; line-height:1.5;">
                A candidate recommendation based on the dataset's T10 experimental conditions, not a validated optimum for every avocado.
            </div>
        </div>

        <!-- Potential Benefit of Switching Storage Conditions Card -->
        <div style="background:{benefit_card_bg}; border:1px solid {benefit_card_border}; border-radius:12px; padding:20px 24px;">
            <div style="font-size:15px; font-weight:500; color:{text_primary}; margin-bottom:12px;">
                Potential benefit of switching storage conditions
            </div>
            <div style="margin-bottom:8px; display:flex; align-items:baseline; gap:8px;">
                <span style="font-size:32px; font-weight:700; color:{accent_green}; line-height:1.1;">+{diff_days} days</span>
                <span style="font-size:16px; color:{text_primary}; font-weight:400;">additional estimated life</span>
            </div>
            <div style="font-size:14px; color:{text_primary}; opacity:0.9;">
                {days_val} days under current conditions &rarr; {rec_days_val} days under recommended conditions.
            </div>
        </div>

    </div>
    """
    st.html(report_html)

    # --- Scientific Deep Dive (Biomarkers, Heatmap & FEFO) ---
    with st.expander("🔬 View Computer Vision Biomarkers & Spectral Heatmap", expanded=False):
        t1, t2, t3 = st.tabs(["Spectral Heatmap", "Biomarker Table", "Dynamic FEFO"])

        with t1:
            h_c1, h_c2 = st.columns(2)
            with h_c1:
                st.image(cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB), caption="Original RGB Surface", use_container_width=True)
            with h_c2:
                heatmap = generate_degradation_heatmap(curr_bgr)
                st.image(heatmap, caption="Spectral Pre-Visual Degradation Heatmap", use_container_width=True)
            st.caption("🔴 Red/Yellow: Internal enzymatic breakdown zones | 🔵 Blue/Green: Intact cellular tissue")

        with t2:
            bio = res['cv_biomarkers']
            bio_df = pd.DataFrame([
                {"Biomarker": "CIE Lightness (L*)", "Value": bio["cie_lightness_L"], "Relevance": "Tissue darkening & loss of exocarp gloss"},
                {"Biomarker": "CIE Chromaticity (a*)", "Value": bio["cie_chromaticity_a"], "Relevance": "Chlorophyll-to-anthocyanin transition"},
                {"Biomarker": "Color Drift ΔE", "Value": bio["delta_e_color_drift"], "Relevance": "Total spectral drift from fresh baseline"},
                {"Biomarker": "Excess Green (ExG)", "Value": bio["excess_green_chlorophyll"], "Relevance": "Chlorophyll density in outer cells"},
                {"Biomarker": "Excess Red (ExR)", "Value": bio["excess_red_browning"], "Relevance": "Phenolic browning and senescence"},
                {"Biomarker": "GLCM Texture Energy", "Value": bio["texture_energy"], "Relevance": "Exocarp nodulation uniformity"},
                {"Biomarker": "Laplacian Roughness", "Value": bio["surface_roughness_laplacian"], "Relevance": "Peel turgor loss & microscopic wrinkling"}
            ])
            st.dataframe(bio_df, hide_index=True, use_container_width=True)

        with t3:
            fefo = res['fefo_recommendation']
            st.markdown(f"""
            <div style="background:{target_card_bg}; border:1px solid {target_card_border}; border-radius:10px; padding:16px;">
                <p style="font-size:15px; font-weight:700; color:{text_primary}; margin:0 0 6px 0;">📦 FEFO Directive: {fefo['tier']}</p>
                <p style="font-size:14px; margin:4px 0; color:{text_primary};"><b>Action:</b> {fefo['action']}</p>
                <p style="font-size:14px; margin:4px 0; color:{text_primary};"><b>Markdown Incentive:</b> <span style="font-weight:700; color:{accent_green};">{fefo['discount_percent']}% Off</span></p>
                <p style="font-size:14px; margin:4px 0; color:{text_primary};"><b>Logistics Routing:</b> {fefo['routing']}</p>
                <p style="font-size:14px; margin:4px 0; color:{text_primary};"><b>Execution SLA:</b> <span style="font-weight:700; color:{accent_green};">{fefo['urgency']}</span></p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
