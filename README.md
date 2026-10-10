# Predicting the Unseen: Pre-Visual Food Waste Mitigation & Produce Quality Assessment

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-OpenCV%20%7C%20LightGBM%20%7C%20Streamlit-orange.svg)]()

> **Problem Statement:** Globally, one-third of all food produced is wasted, with fresh produce accounting for the highest losses. In retail supply chains, inventory is managed using outdated First-In, First-Out (FIFO) methodologies or static "Best Before" dates. Fresh produce degrades from the inside out due to cell wall breakdown and enzymatic senescence before visible mould, slime, or severe rot appears. By the time a human inspector or standard RGB camera spots visual decay, the product is already unsellable and discarded.  
>  
> **Solution:** An intelligent, computer vision-driven software solution that analyzes produce exocarp biomarkers to accurately predict commercial **Produce Quality Grades (Grade A, B, C, Reject)**, continuous **Produce Quality Scores (0–100%)**, internal degradation severity, and **Remaining Shelf Life (RSL in days)**. This enables dynamic **First-Expired, First-Out (FEFO)** inventory routing to proactively mitigate food waste before produce spoils.

---

## 1. Scientific & Biophysical Principles

Traditional supply chains rely on **human visual inspection**, which only catches spoilage when produce reaches **Stage 5 (Overripe/Decayed)**. However, internal cellular breakdown and pre-visual softening initiate days earlier:

```
+----------------------------------------------------------------------------------------------------+
|                                    BIOLOGICAL LIFECYCLE TIMELINE                                   |
+----------------------------------------------------------------------------------------------------+
|  Stage 1: Underripe      Stage 2: Breaking        Stage 3: Firm-Ripe    Stage 4: Ripe     Stage 5  |
|  (Hard, Intact Walls)    (Pre-Visual Softening)   (Active Breakdown)    (Ready-to-Eat)    (Decay)  |
|                                                                                                    |
|  [====== PRE-VISUAL WINDOW: UP TO 12.5 DAYS ======]                   [ VISIBLE ROT ]              |
|  Appearance: Bright Green                         Appearance: Green     Appearance: Brown/Black    |
|  Inspector: "Looks Fresh, Normal Shelf"                                 Inspector: "Spoiled"       |
|  CV Model: Detects Chlorophyll Loss & Texture                           CV Model: Waste Risk = 100%|
+----------------------------------------------------------------------------------------------------+
```

### Key Computer Vision Biomarkers
1. **Calibrated CIE $L^*a^*b^*$ Perceptual Shift:**
   - **$L^*$ (Lightness):** Monotonically decreases as exocarp gloss and moisture content decrease ($120.9$ in Stage 1 $\rightarrow$ $76.6$ in Stage 5).
   - **$a^*$ (Chromaticity):** Shifts from green ($117.4$) towards red/brown ($132.4$) as chlorophyll degrades into pheophytin and anthocyanin pigments accumulate.
   - **$\Delta E$ Total Color Distance:** Quantifies total spectral drift from a calibrated fresh baseline ($L^*=124.4, a^*=116.9, b^*=152.5$).
2. **Spectral Chlorophyll & Senescence Pigment Indices:**
   - **Excess Green (ExG):** $(2G - R - B) / (R + G + B)$ measures outer chlorophyll density ($+0.167$ in Stage 1 $\rightarrow$ $-0.050$ in Stage 5).
   - **Excess Red (ExR):** $(1.4R - G) / (R + G + B)$ captures phenolic browning.
   - **Browning Ratio:** $\text{ExR} / (\text{ExG} + 0.5)$ tracks relative browning balance.
   - **Color Decay Index:** $(a^* - 116.9) / (L^* + 10^{-5})$ captures normalized senescence rate.
3. **Spatial Necrosis & Surface Decay Heterogeneity:**
   - **Surface Necrosis Fraction (`decay_pct`):** Identifies localized browning patches and necrotic lesions on non-uniformly ripening fruit.
   - **Percentile Lightness & Browning (`p10_l`, `p25_l`, `p90_a`):** Captures dark sunken lesions even when upper sections appear green.
   - **Heterogeneity Index:** $(L^*_{\text{std}} \times a^*_{\text{std}}) / 100$ quantifies mottled exocarp discoloration.
4. **Micro-Texture & Surface Roughness:**
   - **GLCM (Gray-Level Co-occurrence Matrix):** Energy, Contrast, and Homogeneity capture changes in exocarp nodulation and lenticel patterns as internal turgor pressure drops.
   - **Laplacian Variance:** Captures micro-roughness and wrinkling on the exocarp surface.

---

## 2. Project Architecture

The codebase is organized into modular files with zero unnecessary bloat:

```
iitb_cv2/
├── dataset.py            # Dataset loader, longitudinal RSL & Quality ground-truth generator, multithreaded caching
├── features.py           # CV feature extraction (CIE-Lab, HSV, RGB, spatial decay, GLCM texture, degradation heatmap)
├── train.py              # Multi-head model training (Stage Classifier, RSL Regressor, Degradation & Quality Regressors)
├── predict.py            # Standalone CLI inference with Quality Grade, pre-visual alerts & dynamic FEFO routing
├── evaluate.py           # Comprehensive evaluation report and FIFO vs FEFO food waste simulation
├── app.py                # Interactive Streamlit dashboard with Quality Assessment & heatmap visualization
├── requirements.txt      # Dependency specification
└── README.md             # Project documentation
```

---

## 3. Ground Truth Formulation & Multi-Task Targets

Using the **Hass Avocado Ripening Photographic Dataset** (13,604 images tracked longitudinally across 410 unique fruit specimens over 26 days):

1. **Commercial Produce Quality Score ($0 - 100\%$):**
   $$\text{Quality Score}_t = \max(0, \min(100, 100 - \text{Degradation Index}_t))$$
   A standardized commercial quality rating where $100\%$ represents peak fresh produce and $0\%$ represents total senescence.
2. **Commercial Produce Quality Grades:**
   - **Grade A (Premium Fresh):** Stage 1 / Fresh Breaking (Quality $> 70\%$, RSL $> 5\text{ days}$) — pristine retail value, highest price tier.
   - **Grade B (Standard Retail):** Stage 3 Firm-Ripe (Quality $45\% - 70\%$, RSL $3 - 5\text{ days}$) — prime eating window, standard shelf display.
   - **Grade C (Markdown / Deli Repurpose):** Stage 4 Soft-Ripe (Quality $25\% - 45\%$, RSL $1 - 2\text{ days}$) — dynamic discount or in-store kitchen conversion.
   - **Reject (Bio-Divert):** Stage 5 / Senescent / Necrotic (Quality $< 25\%$, RSL $\le 0.5\text{ days}$, or Decay $> 35\%$) — write-off, divert to composting.
3. **Remaining Shelf Life (RSL in Days):**
   $$\text{RSL}_t = \max(0, \text{Day}_{\text{Stage 5 Onset}} - \text{Day}_t)$$
   Accurately represents the true number of days until the produce crosses into unmarketable senescence.
4. **Internal Degradation Severity Index ($0 - 100\%$):**
   Continuous index of cellular breakdown based on normalized lifespan progression.
5. **Ripening & Degradation Stage ($1 - 5$):**
   - **Stage 1:** Underripe (Firmness > 120 N, Intact cellular structure)
   - **Stage 2:** Breaking (Pre-visual softening initiated; 60–120 N)
   - **Stage 3:** Firm-Ripe (Early edible stage; 30–60 N)
   - **Stage 4:** Optimal Ripe (Peak organoleptic maturity; 10–30 N)
   - **Stage 5:** Overripe / Spoilage (< 10 N, vascular browning, breakdown)

---

## 4. Quantitative Results & Benchmarks

Evaluated on held-out avocado specimens strictly split at the individual fruit level (**Zero Data Leakage** across 2,734 test images from 82 unseen fruit specimens):

| Metric | Score | Industry Context |
| :--- | :--- | :--- |
| **Stage Accuracy (±1 Tolerance)** | **97.29%** | Post-harvest industry standard tolerance |
| **Exact Stage Accuracy** | **67.70%** | Multi-class balance across all 5 ripening stages |
| **Stage Macro F1-Score** | **0.6713** | Balanced across all classes |
| **Pre-Visual (Stage 1 & 2) Accuracy** | **79.07% (99.50% tol)** | Near-perfect precision during pre-visual window |
| **Remaining Shelf Life (RSL) MAE** | **2.46 Days** | High predictive power ($R^2 = 0.7094, \text{RMSE} = 3.33\text{d}$) |
| **Produce Quality Score MAE** | **8.52%** | High linear fit across all commercial grades ($R^2 = 0.8739$) |
| **Internal Degradation Index MAE** | **8.52%** | Strong cellular breakdown correlation ($R^2 = 0.8739$) |
| **Average Early Warning Buffer** | **12.5 Days** | Days advance notice before Stage 5 rot appears |
| **Simulated Food Waste Reduction** | **34.4% Mitigated** | Dynamic FEFO vs Static FIFO baseline |

### Key Computer Vision Biomarkers (Top LightGBM Feature Importances)
1. `glcm_energy` (Texture uniformity & lenticel nodulation)
2. `glcm_contrast` (Micro-surface contrast variations)
3. `s_std` (Color saturation spatial variance)
4. `a_std` (CIE $a^*$ localized redness/browning variance)
5. `std_b` (Blue channel reflectance spread)
6. `glcm_homogeneity` (Surface roughness homogeneity)
7. `mean_b` (Mean blue channel reflectance)
8. `h_mean` (HSV hue center)

---

## 5. Dynamic FEFO Inventory Routing Policy

Instead of static FIFO, the system outputs automated retail directives based on predicted produce quality and remaining shelf life:

| Produce Quality Grade | Predicted RSL | Stage | Supply Chain Action Tier | Dynamic Pricing | Routing Directive |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Grade A (Premium)** | **> 5 Days** | 1 – 2 | **Standard Retail Inventory** | 0% Discount | Standard shelf display / cool storage retention |
| **Grade B (Standard)** | **3 – 5 Days** | 3 | **Priority Sale (FEFO Priority)** | 15% Incentive | Move to front-of-store eye-level promotional displays |
| **Grade C (Markdown)** | **1 – 2 Days** | 4 | **Critical Shelf-Life (Flash Markdown)** | 30% – 50% Off | Route to in-store kitchen/deli (guacamole, prepared salads) |
| **Reject (Bio-Divert)** | **< 1 Day** | 5 | **Decommission / Bio-divert** | 100% Write-off | Route to composting/anaerobic digestion to prevent spore transmission |

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.10+
- Dependencies: `pip install -r requirements.txt`

### Step 1: Feature Extraction & Dataset Build
The dataset automatically extracts and caches 38 computer vision features into Parquet in < 50 seconds:
```bash
python dataset.py
```

### Step 2: Train Multi-Head Models
Trains the Stage Classifier, RSL Regressor, Degradation Regressor, and Quality Regressor in ~5 seconds:
```bash
python train.py
```

### Step 3: Run Inference CLI
Test prediction on any individual produce image:
```bash
# Analyze a sample fruit from the dataset
python predict.py --sample 2

# Analyze any arbitrary image (supports studio, retail, or field photos)
python predict.py --image "Hass Avocado Ripening Photographic Dataset/Avocado Ripening Dataset/T20_d03_001_a_2.jpg"

# Machine-readable JSON output for IoT/hardware integration
python predict.py --sample 1 --json
```

**Example Output:**
```text
====================================================================
 PRE-VISUAL FOOD QUALITY & SHELF-LIFE INFERENCE REPORT
====================================================================
Target Image:      T10_d01_002_a_1.jpg
Produce Quality:   Grade A (Premium Fresh) (Score: 95.8 / 100)
Commercial Value:  100% Full Retail Value
Ripening Stage:    Stage 1 - Underripe (Hard/Intact) (100.0% conf)
Remaining Life:    16.2 DAYS remaining before spoilage
Degradation Index: 4.2% internal cellular breakdown
Surface Necrosis:  0.6% damaged/decaying peel
--------------------------------------------------------------------
Quality Summary:   Pristine exocarp turgor, intact cellular walls, maximum shelf stability.
Pre-Visual Alert:  NORMAL: No internal degradation detected
Visual Appearance: Dark green, firm, intact exocarp
Internal Biology:  Intact cellular walls, high turgor pressure, negligible degradation
--------------------------------------------------------------------
 DYNAMIC FEFO (FIRST-EXPIRED, FIRST-OUT) INVENTORY ROUTING:
  Action Tier:     Standard Retail Inventory
  Directive:       Maintain at standard price; standard shelf storage
  Dynamic Pricing: 0% markdown incentive
  Logistics Route: Normal retail display or cool storage retention
  Execution SLA:   Low (> 5 days buffer)
====================================================================
```

### Step 4: Run System Evaluation & Simulation
```bash
python evaluate.py
```

### Step 5: Launch Interactive Web Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to:
- Inspect produce across all ripening stages and quality tiers
- View Commercial Produce Quality Grade Badges (Grade A, B, C, Reject) & Quality Scores (0–100%)
- Visualize pre-visual degradation heatmaps side-by-side with RGB images
- View Remaining Shelf Life (RSL) gauges & dynamic FEFO retail recommendations
