# KalFresh: Predicting the Unseen
### Pre-Visual Food Waste Mitigation, Produce Quality Assessment & Dynamic FEFO Routing

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-OpenCV%20%7C%20LightGBM%20%7C%20Streamlit-orange.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-Hass%20Avocado%20Ripening%20(13.6k%20Images)-success.svg)]()
[![Model](https://img.shields.io/badge/Models-5--Head%20LightGBM%20Bundle-purple.svg)]()
[![Validation](https://img.shields.io/badge/Validation-Zero%20Data%20Leakage%20(Fruit--Level)-brightgreen.svg)]()
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kalfresh-northsouthgit.streamlit.app/)

> **Live Deployment:** [https://kalfresh-northsouthgit.streamlit.app/](https://kalfresh-northsouthgit.streamlit.app/)  
> **IIT Bombay Computer Vision AI Hackathon**  
> An intelligent, biophysically-grounded computer vision platform that detects invisible cellular degradation in fresh produce, forecasts remaining shelf life conditioned on real-world storage microclimates, and triggers automated First-Expired, First-Out (FEFO) retail inventory directives to cut supply-chain shrink by **34.4%**.

---

## Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Scientific & Biophysical Principles](#2-scientific--biophysical-principles)
3. [System Architecture & Pipeline](#3-system-architecture--pipeline)
4. [Computer Vision Biomarker Taxonomy (38 Features)](#4-computer-vision-biomarker-taxonomy-38-features)
5. [Ground Truth Formulation & Multi-Task Targets](#5-ground-truth-formulation--multi-task-targets)
6. [Quantitative Benchmarks & Evaluation Results](#6-quantitative-benchmarks--evaluation-results)
7. [Environmental Storage Conditioning & Biological Kinetics](#7-environmental-storage-conditioning--biological-kinetics)
8. [Dynamic FEFO Inventory Routing Policy](#8-dynamic-fefo-inventory-routing-policy)
9. [Interactive Streamlit Web Dashboard](#9-interactive-streamlit-web-dashboard)
10. [CLI & Machine-Readable IoT/JSON API](#10-cli--machine-readable-iotjson-api)
11. [Repository Structure & Codebase Manifest](#11-repository-structure--codebase-manifest)
12. [Installation & Step-by-Step Quickstart](#12-installation--step-by-step-quickstart)
13. [Scientific References & Citations](#13-scientific-references--citations)

---

## 1. Executive Summary & Problem Statement

Globally, **one-third of all food produced is wasted annually**, with post-harvest fresh fruits and vegetables accounting for the single highest category of economic loss ($>\$400\text{B}$ per year).

Traditional retail grocery and supply-chain logistics manage produce using outdated methods:
- **First-In, First-Out (FIFO):** Dispatches stock based strictly on warehouse arrival date, completely blind to individual specimen ripening kinetics and transit temperature histories.
- **Static "Best-Before" Dates:** Discards perfectly edible produce or permits premature shelf rot due to arbitrary static expiry estimates.
- **Manual Human Visual Inspection:** Human graders and standard threshold RGB cameras only identify spoilage when produce reaches **Stage 5 (Senescence / Overripe Decay)**.

### The Biological "Pre-Visual" Dilemma
Produce breaks down from the inside out. Pectolytic enzyme activity (polygalacturonase, pectin methylesterase) degrades middle lamella cell walls, internal moisture migrates, and chlorophyll pigment converts into pheophytin **days before visible mold, slime, or necrotic browning appears on the exocarp**. By the time a human inspector detects rot, the produce is already unsellable and must be landfilled.

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
|  KalFresh Model: Detects Chlorophyll Loss & Texture                     KalFresh Model: Waste=100% |
+----------------------------------------------------------------------------------------------------+
```

### The KalFresh Solution
**KalFresh** bridges this critical gap by extracting **38 computer-vision biomarkers** (perceptual CIE $L^*a^*b^*$ drift, chlorophyll-to-anthocyanin ratios, GLCM texture energy, and spatial necrosis) combined with a **5-head LightGBM machine learning architecture**. It calculates:
1. **Commercial Produce Quality Grade:** Grade A (Premium), Grade B (Standard), Grade C (Markdown), or Reject.
2. **Produce Quality Score ($0\text{--}100\%$):** Continuous freshness rating.
3. **Internal Degradation Index ($0\text{--}100\%$):** Pre-visual cellular senescence percentage.
4. **Remaining Shelf Life (RSL in Days):** Dynamically conditioned on ambient storage temperature ($^\circ\text{C}$) and relative humidity ($\%$).
5. **Storage Optimization & Days Gained:** Quantifies additional shelf life gained by switching to recommended cold-chain conditions ($10^\circ\text{C}, 85\%\text{ RH}$).
6. **Dynamic FEFO Inventory Directives:** Real-time routing directives (pricing discounts, deli conversions, anaerobic bio-diversion).

---

## 2. Scientific & Biophysical Principles

KalFresh models post-harvest senescence using biophysical and photobiological principles:

### 1. Calibrated CIE $L^*a^*b^*$ Perceptual Drift
In an 8-bit digital color space, standard RGB channels conflate illumination intensity with chrominance. KalFresh converts images into **CIE $L^*a^*b^*$** perceptual space calibrated against an empirical Stage 1 pristine baseline ($L^*_{\text{ref}}=124.4, a^*_{\text{ref}}=116.9, b^*_{\text{ref}}=152.5$):
- **$L^*$ (Lightness):** Monotonically decreases as exocarp gloss fades and internal turgor pressure drops ($120.9$ in Stage 1 $\rightarrow$ $76.6$ in Stage 5).
- **$a^*$ (Chromaticity):** Shifts from green ($117.4$) towards red/brown ($132.4$) as chlorophyll pigment breaks down and anthocyanin accumulates.
- **$\Delta E$ Perceptual Color Drift:** Euclidean spectral distance quantifying total divergence from fresh exocarp tissue:
  $$\Delta E = \sqrt{(L^* - L^*_{\text{ref}})^2 + (a^* - a^*_{\text{ref}})^2 + (b^* - b^*_{\text{ref}})^2}$$

### 2. Spectral Senescence & Chlorophyll Indices
- **Excess Green Index (ExG):** Measures chlorophyll density in outer exocarp cells:
  $$\text{ExG} = \frac{2G - R - B}{R + G + B} \quad (+0.167 \text{ in Stage 1} \rightarrow -0.050 \text{ in Stage 5})$$
- **Excess Red Index (ExR):** Quantifies anthocyanin accumulation and phenolic browning:
  $$\text{ExR} = \frac{1.4R - G}{R + G + B}$$
- **Browning Ratio:** Captures the ratio of enzymatic browning pigments to active chlorophyll:
  $$\text{Browning Ratio} = \frac{\text{ExR}}{\text{ExG} + 0.5}$$
- **Normalized Color Decay Index (CDI):**
  $$\text{CDI} = \frac{a^* - 116.9}{L^* + 10^{-5}}$$

### 3. Micro-Texture & Surface Roughness (GLCM)
As mesocarp cell walls break down, water transpires through exocarp lenticels, causing exocarp nodulation to flatten and lenticel micro-textures to degrade:
- **GLCM Energy (Uniformity):** Measures textural homogeneity across spatial pixel pairs at $0^\circ$ and distance $d=1$. High in Stage 1 ($0.2551$), dropping monotonically as surface discolors.
- **GLCM Contrast & Homogeneity:** Captures local contrast variation of peel nodules.
- **Laplacian Variance:** Computes the second spatial derivative $\nabla^2 I$ to measure high-frequency micro-wrinkling and peel turgidity loss.

### 4. Spatial Necrosis & Decay Heterogeneity
Ripening is rarely isotropic. Localized damage or early fungal micro-lesions create spatial browning before the overall fruit darkens:
- **Surface Necrosis Fraction (`decay_pct`):** Measures percentage of foreground pixels crossing necrotic browning thresholds ($a^* > 132$ and $L^* < 85$).
- **Lightness Percentiles (`p10_l`, `p25_l`, `p90_a`):** Captures localized dark sunken depressions that mean values mask.
- **Heterogeneity Index:** $(L^*_{\text{std}} \times a^*_{\text{std}}) / 100$ flags mottled, uneven ripening patterns.

---

## 3. System Architecture & Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          KALFRESH END-TO-END PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                         [ Produce Image (JPG/PNG) ]
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │       Adaptive Foreground Fruit Segmentation    │
             │   (Otsu Thresholding + Morphological Closing)   │
             └─────────────────────────────────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │    Computer Vision Feature Extraction Engine    │
             │   38 Biomarkers: CIE Lab, HSV, RGB, Texture,    │
             │   Spectral Indices, Spatial Necrosis Fractions  │
             └─────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
   ┌─────────────────────────────────┐ ┌─────────────────────────────────┐
   │    Pre-Visual Heatmap Engine    │ │   Multi-Head LightGBM Bundle    │
   │  Pixel-wise ΔE Baseline Distance│ │  [1] Stage Classifier (1–5)     │
   │  JET Color Mapping & RGB Blend  │ │  [2] Base RSL Regressor         │
   └─────────────────────────────────┘ │  [3] Temp-Conditioned RSL Reg   │
                    │                  │  [4] Degradation Regressor (%)  │
                    │                  │  [5] Quality Score Regressor (%)│
                    │                  └─────────────────────────────────┘
                    │                                   │
                    │                                   ▼
                    │                  ┌─────────────────────────────────┐
                    │                  │  Environmental Microclimate     │
                    │                  │  Conditioning (Temp °C, RH %)   │
                    │                  │  + Biological Q10 Kinetics      │
                    │                  └─────────────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │      Dynamic FEFO Supply Chain Decision Engine  │
             │  * Commercial Quality Grade (A, B, C, Reject)   │
             │  * Storage Recommendation (Target: 10°C, 85% RH)│
             │  * Days Gained Calculation (+Δ Days)            │
             │  * Dynamic Markdown & Logistics Routing Actions │
             └─────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
   ┌─────────────────────────────────┐ ┌─────────────────────────────────┐
   │ Interactive Streamlit Dashboard │ │   Machine-Readable JSON API     │
   │ Dual-Theme (Dark/Light), Metric │ │   For IoT Edge Scanners, Conveyors│
   │ Cards, Heatmap & Biomarkers     │ │   & Warehouse ERP Integration   │
   └─────────────────────────────────┘ └─────────────────────────────────┘
```

---

## 4. Computer Vision Biomarker Taxonomy (38 Features)

All 38 extracted features are cached into Apache Parquet (`data/avocado_features.parquet`) for instantaneous model training and sub-50ms inference:

| Category | Feature Name | Mathematical / Algorithmic Definition | Biophysical Significance |
| :--- | :--- | :--- | :--- |
| **CIE $L^*a^*b^*$ Perceptual** | `l_mean`, `l_std` | Mean & std dev of CIE Lightness channel | Tracks exocarp dulling, moisture loss, and darkening |
| | `a_mean`, `a_std` | Mean & std dev of CIE Chromaticity $a^*$ | Measures chlorophyll loss and red/brown anthocyanin shift |
| | `b_mean`, `b_std` | Mean & std dev of CIE Chromaticity $b^*$ | Measures yellow carotenoid expression during ripening |
| | `chroma` | $\sqrt{(a^*)^2 + (b^*)^2}$ | Total color saturation in perceptual space |
| | `delta_e` | $\sqrt{\Delta L^2 + \Delta a^2 + \Delta b^2}$ vs Stage 1 Baseline | Total perceptual color drift from freshly harvested state |
| **RGB Colorimetry** | `mean_r`, `mean_g`, `mean_b` | Mean intensity in red, green, blue channels | Primary broad-band surface reflectance |
| | `std_r`, `std_g`, `std_b` | Spatial standard deviation across channels | Surface color heterogeneity and mottling |
| | `norm_r`, `norm_g`, `norm_b`| $C / (R + G + B)$ | Normalized chromaticity invariant to lighting intensity |
| | `rg_ratio` | $(R + 10^{-5}) / (G + 10^{-5})$ | Red-to-green reflectance ratio |
| **HSV Color Space** | `h_mean`, `h_std` | Hue angle distribution center and spread | Direct angular color tone shift ($120^\circ \rightarrow 30^\circ$) |
| | `s_mean`, `s_std` | Saturation distribution | Color vividness decline during senescence |
| | `v_mean`, `v_std` | Value (perceived brightness) distribution | Surface reflectivity decline |
| **Spectral Senescence** | `exg` | $(2G - R - B) / (R + G + B)$ | Excess Green Index (outer chlorophyll density) |
| | `exr` | $(1.4R - G) / (R + G + B)$ | Excess Red Index (phenolic browning accumulation) |
| | `ndci` | $(R - G) / (R + G + 10^{-5})$ | Normalized Difference Chlorophyll / Carotenoid Index |
| | `browning_ratio` | $\text{ExR} / (\text{ExG} + 0.5)$ | Senescence browning vs active chlorophyll balance |
| | `color_decay_index` | $(a^* - 116.9) / (L^* + 10^{-5})$ | Normalized senescence progression velocity |
| **Spatial Necrosis** | `decay_pct` | Fraction of pixels with $a^* > 132 \land L^* < 85$ | Percentage of exocarp exhibiting active necrosis/lesions |
| | `p10_l`, `p25_l`, `p90_a` | 10th/25th percentile $L^*$ and 90th percentile $a^*$ | Detects early sunken necrotic spots on green peel |
| | `heterogeneity_index` | $(L^*_{\text{std}} \times a^*_{\text{std}}) / 100$ | Quantifies mottled, patchy discoloration across surface |
| **Micro-Texture** | `glcm_energy` | $\sum_{i,j} P(i,j)^2$ from GLCM at $d=1, \theta=0$ | Measures texture uniformity; drops as lenticels flatten |
| | `glcm_contrast` | $\sum_{i,j} \|i - j\|^2 P(i,j)$ | Micro-surface local contrast between peel nodules |
| | `glcm_homogeneity` | $\sum_{i,j} \frac{P(i,j)}{1 + \|i - j\|}$ | Closeness of element distribution in GLCM to diagonal |
| | `laplacian_var` | $\text{Var}(\nabla^2 I)$ | Exocarp micro-roughness and wrinkling from turgor loss |

---

## 5. Ground Truth Formulation & Multi-Task Targets

KalFresh was developed and validated on the **Hass Avocado Ripening Photographic Dataset**, comprising **13,604 high-resolution images** tracked longitudinally across **410 individual fruit specimens** over 26 days across three controlled storage cohorts:
- `T10`: $10.0^\circ\text{C}$ cold storage (long shelf-life regime)
- `T20`: $20.0^\circ\text{C}$ standard room temperature (medium shelf-life regime)
- `Tam`: Ambient fluctuating room storage ($22\text{--}25^\circ\text{C}$, accelerated senescence)

### 1. Ground Truth Remaining Shelf Life (RSL in Days)
For every individual avocado specimen $i$ on day $t$, ground truth RSL is derived longitudinally from the exact day specimen $i$ reached Stage 5 (decay/spoilage onset):
$$\text{RSL}_{i,t} = \max\left(0, \text{Day}_{\text{Stage 5 Onset}, i} - \text{Day}_{i,t}\right)$$

### 2. Internal Degradation Severity Index ($0\text{--}100\%$)
Quantifies the percentage of cellular lifespan consumed:
$$\text{Degradation Index}_{i,t} = \min\left(100.0, \frac{\text{Day}_{i,t}}{\text{Day}_{\text{Stage 5 Onset}, i}} \times 100\right)$$

### 3. Commercial Produce Quality Score ($0\text{--}100\%$)
Standardized retail rating where $100\%$ is pristine, firm-intact produce and $0\%$ is complete unmarketable breakdown:
$$\text{Quality Score}_{i,t} = \max\left(0.0, \min\left(100.0, 100.0 - \text{Degradation Index}_{i,t}\right)\right)$$

### 4. Ripening Stages & Organoleptic States
- **Stage 1 (Underripe):** Hard exocarp, firmness $> 120\text{ N}$. High chlorophyll, pristine cell turgidity.
- **Stage 2 (Breaking):** Pre-visual softening initiated; firmness $60\text{--}120\text{ N}$. Exocarp visually remains green to human eyes, but internal pectolysis has begun.
- **Stage 3 (Firm-Ripe):** Early edible stage; firmness $30\text{--}60\text{ N}$. Moderate yielding to thumb pressure.
- **Stage 4 (Optimal Ripe):** Ready-to-eat organoleptic peak; firmness $10\text{--}30\text{ N}$. Rich flavor, yielding exocarp.
- **Stage 5 (Overripe / Decay):** Senescent; firmness $< 10\text{ N}$. Vascular browning, rancid off-flavors, necrotic peel lesions.

### 5. Commercial Produce Quality Grades
- **Grade A (Premium Fresh):** Quality $> 70\%$ and $\text{RSL} > 5\text{ days}$. Full retail display value.
- **Grade B (Standard Retail):** Quality $45\%\text{--}70\%$ and $\text{RSL} \in [3, 5]\text{ days}$. Prime consumption window.
- **Grade C (Markdown / Kitchen Repurpose):** Quality $25\%\text{--}45\%$ and $\text{RSL} \in [1, 2]\text{ days}$. High risk of shrink within 48h.
- **Reject (Decommission / Bio-Divert):** Quality $< 25\%$ or $\text{RSL} \le 0.5\text{ days}$ or Surface Necrosis $> 35\%$. Immediate bio-diversion.

---

## 6. Quantitative Benchmarks & Evaluation Results

Evaluated on held-out avocado specimens strictly partitioned at the **individual fruit level** (**Zero Data Leakage**: 10,870 training images across 328 avocados; 2,734 test images across 82 unseen avocados):

### Overall System Benchmark Summary

| Metric | Score | Industry Benchmark / Interpretation |
| :--- | :--- | :--- |
| **Stage Accuracy (±1 Tolerance)** | **97.29%** | Post-harvest industry standard grading tolerance |
| **Exact Stage Accuracy** | **67.70%** | 5-class balanced multiclass classification |
| **Stage Macro F1-Score** | **0.6713** | Uniform performance across all ripening stages |
| **Pre-Visual (Stages 1 & 2) Accuracy** | **79.07% (99.50% tol)** | Near-zero false negatives during the pre-visual window |
| **Base RSL MAE (Image Only)** | **2.46 Days** | High predictive power ($R^2 = 0.7094, \text{RMSE} = 3.33\text{d}$) |
| **Temp-Conditioned RSL MAE** | **1.67 Days** | Significant error drop when incorporating storage temp ($R^2 = 0.8668$) |
| **Produce Quality Score MAE** | **8.52%** | High linear fit across all commercial quality tiers ($R^2 = 0.8739$) |
| **Internal Degradation Index MAE** | **8.52%** | Strong cellular breakdown correlation ($R^2 = 0.8739$) |
| **Average Early Warning Buffer** | **12.5 Days** | Advance warning notice before Stage 5 rot appears |
| **Supply Chain Food Waste Reduction** | **34.4% Mitigated** | Dynamic FEFO vs Static FIFO baseline simulation |

### Per-Class Classification Report (Held-Out Test Set)

```text
               precision    recall  f1-score   support

  1-Underripe      0.853     0.893     0.872       616
   2-Breaking      0.631     0.624     0.628       378
   3-FirmRipe      0.631     0.643     0.637       518
4-OptimalRipe      0.567     0.578     0.572       630
   5-Overripe      0.675     0.622     0.647       592

     accuracy                          0.677      2734
    macro avg      0.671     0.672     0.671      2734
 weighted avg      0.676     0.677     0.676      2734
```

### Multiclass Confusion Matrix
Errors are almost strictly confined to adjacent ripening phases ($\pm 1$ stage), proving continuous natural transition without catastrophic misclassifications:

```text
Actual \ Predicted   1-Underripe  2-Breaking  3-FirmRipe  4-OptimalRipe  5-Overripe
1-Underripe                 550          64           2              0           0
2-Breaking                   95         236          44              3           0
3-FirmRipe                    0          68         333             82          35
4-OptimalRipe                 0           3         121            364         142
5-Overripe                    0           3          28            193         368
```

### Pre-Visual Early Warning Performance
- **Pre-Visual Subset (Stages 1 & 2):** Produce that appears outwardly green and pristine to human inspectors.
- **Exact Accuracy:** $79.07\%$
- **$\pm 1$ Stage Tolerance:** $99.50\%$
- **RSL Error within Window:** $3.52\text{ days}$
- **Advance Warning Notice:** **12.5 Days** average buffer before visible Stage 5 spoilage manifests, providing ample runway for supply chain rerouting.

### FIFO vs FEFO Supply Chain Simulation
Under a simulated commercial retail warehouse lot ($N=2,734$ produce observations):
- **Traditional Static FIFO Shrink:** $21.7\%$ unmitigated waste (produce spoils while newer stock is sold).
- **Dynamic FEFO Residual Shrink:** $14.2\%$ waste.
- **Total Shrink Mitigated:** **34.4% Food Waste Reduction** achieved through automated price markdowns and kitchen conversion before senescence.

---

## 7. Environmental Storage Conditioning & Biological Kinetics

Produce shelf life is not fixed; it is intrinsically linked to environmental microclimates. KalFresh incorporates real-time environmental conditioning:

### 1. Empirical Conditioning Range
The dataset experimentally documents three distinct temperature regimes: $10^\circ\text{C}$ (`T10`), $20^\circ\text{C}$ (`T20`), and $23^\circ\text{C}$ ambient (`Tam`). For temperatures within this nominal range ($10^\circ\text{C}\text{--}23^\circ\text{C}$), the dedicated `temp_rsl_reg` model directly predicts RSL with **1.67 days MAE**.

### 2. Biological $Q_{10}$ Respiration Kinetics
For storage temperatures outside the nominal training range (e.g. elevated tropical retail displays at $28^\circ\text{C}$ or walk-in chillers at $6^\circ\text{C}$), KalFresh applies the biological **Arrhenius $Q_{10}$ temperature coefficient**:
$$\text{Decay Multiplier} = Q_{10}^{\frac{T_{\text{current}} - 23.0}{10.0}} \quad (\text{where } Q_{10} = 2.0)$$
$$\text{RSL}_{\text{extrapolated}} = \frac{\text{RSL}_{\text{ambient}}}{\text{Decay Multiplier}}$$

### 3. Recommended Storage Target & Days Gained
- **Temperature Target:** $10.0^\circ\text{C}$ (validated cold-storage trial baseline).
- **Humidity Target:** $85.0\%\text{ RH}$ (commercial post-harvest guideline).
- **Potential Benefit ($\Delta \text{ Days Gained}$):**
  $$\Delta \text{ Days} = \max\left(0, \text{RSL}_{\text{recommended}} - \text{RSL}_{\text{current}}\right)$$
- **Scientific Disclosure on Humidity:** Relative humidity ($85\%\text{ RH}$) is a recognized post-harvest commercial standard that minimizes fruit transpiration and moisture shriveling. In the dataset metadata, humidity was maintained consistently per temperature regime; thus, humidity serves as a best-practice guideline rather than an independently varied model regression parameter.

---

## 8. Dynamic FEFO Inventory Routing Policy

Rather than static expiry dates, KalFresh translates predictions into actionable supply chain routing directives:

| Quality Grade | Predicted RSL | Stage | Action Tier | Dynamic Pricing | Physical Logistics Directive | Execution SLA |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Grade A (Premium)** | **> 5.0 Days** | 1 – 2 | **Standard Retail Inventory** | 0% Discount | Standard shelf display / cold-storage holding | Low (> 5 days) |
| **Grade B (Standard)** | **3.0 – 5.0 Days** | 3 | **Priority Sale (FEFO Priority)** | 15% Incentive | Move to eye-level front-of-store promotional islands | Moderate (3–5 days) |
| **Grade C (Markdown)** | **1.0 – 2.0 Days** | 4 | **Critical Shelf-Life (Flash Sale)** | 30% – 50% Off | Route immediately to in-store kitchen (guacamole / deli) | Urgent (24–48 hours) |
| **Reject (Bio-Divert)** | **< 1.0 Day** | 5 | **Decommission / Bio-Divert** | 100% Write-Off | Divert to anaerobic composting to prevent fungal transmission | Immediate (< 12 hours) |

---

## 9. Interactive Streamlit Web Dashboard

The web dashboard (`app.py`) provides a clean interface for inventory managers and retail quality inspectors:

### Key UI Features
- **Avocado Freshness Report Card:**
  - $104\times104\text{px}$ rounded produce image thumbnail with smooth aspect ratio fitting.
  - Header with title, subtitle, and green pill badge (*"Illustrative prediction"*).
  - Prominent dual-metric layout: **Freshness (%)** in green and **Days remaining (days)** under current microclimates.
  - "Your current storage conditions" breakdown (Temperature & Relative Humidity).
  - "Recommended storage conditions" target cards ($10^\circ\text{C}$ and $85\%\text{ RH}$) with custom SVG icons.
  - "Potential benefit of switching storage conditions" card highlighting $+\Delta\text{ days}$ additional shelf life gained.
- **Instant Reactive Dual-Theme Engine:**
  - Instant, one-click toggle between **🌙 Dark Mode** and **☀️ Light Mode** using `st.rerun` without browser reload.
  - Deep dark theme with pure `#000000` canvas and `#0B0D11` panel cards.
  - Complete CSS overrides for number inputs, dropdowns, and file dropzones preventing white-box visual glitches.
  - Built with `st.html` for zero code-block leaks.
- **Flexible Produce Input:**
  - Choose sample avocados from the dataset across all 5 ripening stages.
  - Upload arbitrary custom photos (JPG/PNG).
  - Temperature and humidity parameter controls.
- **Scientific Deep-Dive Drawer (Collapsible Expander):**
  - **Tab 1 (Spectral Heatmap):** Side-by-side comparison of original RGB surface vs OpenCV JET colormap pre-visual degradation heatmap highlighting internal enzymatic breakdown zones.
  - **Tab 2 (Biomarkers Table):** Interactive data table displaying 7 key CV biomarkers (CIE $L^*$, $a^*$, $\Delta E$, ExG, ExR, GLCM Energy, Laplacian Roughness).
  - **Tab 3 (Dynamic FEFO Directive):** Real-time actionable card detailing action tier, discount percentage, logistics route, and execution SLA.

---

## 10. CLI & Machine-Readable IoT/JSON API

KalFresh supports fully automated command-line execution for integration into conveyor belt cameras, smart cold-room IoT microcontrollers, and retail ERP systems.

### CLI Usage Examples
```bash
# Analyze a sample fruit with custom storage microclimate
python predict.py --sample 2 --temp 28.0 --humidity 65.0

# Analyze an arbitrary image file
python predict.py --image "Hass Avocado Ripening Photographic Dataset/Avocado Ripening Dataset/T20_d03_001_a_2.jpg" --temp 24.0

# Export structured JSON for IoT / ERP integration
python predict.py --sample 1 --temp 22.0 --humidity 60.0 --json
```

### Formatted Terminal Output
```text
====================================================================
 PRE-VISUAL FOOD QUALITY & SHELF-LIFE INFERENCE REPORT
====================================================================
Target Image:      T10_d01_002_a_1.jpg
Produce Quality:   Grade A (Premium Fresh) (Score: 95.8 / 100)
Commercial Value:  100% Full Retail Value
Ripening Stage:    Stage 1 - Underripe (Hard/Intact) (100.0% conf)
Remaining Life:    5.6 DAYS (at 28.0°C, 65.0% RH)
Degradation Index: 4.2% internal cellular breakdown
Surface Necrosis:  0.6% damaged/decaying peel
--------------------------------------------------------------------
 STORAGE CONDITION & SHELF LIFE RECOMMENDATION:
  Current Storage:     28.0°C, 65.0% RH -> 5.6 Days Remaining
  Recommended Storage: 10.0°C, 85.0% RH -> 20.4 Days Remaining
  Potential Benefit:   +14.8 Additional Days of Shelf Life
  Directive:           Transferring fruit from current 28.0°C to recommended 10.0°C cold storage slows cellular respiration, extending estimated shelf life by +14.8 days.
  Humidity Note:       Relative humidity (85% RH) is a recognized post-harvest commercial guideline to minimize fruit water loss and shriveling, rather than an independently model-optimized value (the dataset metadata does not independently vary RH).
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
--------------------------------------------------------------------
 KEY CV BIOMARKERS:
  * cie_lightness_L               : 140.4
  * cie_chromaticity_a            : 114.4
  * delta_e_color_drift           : 16.2
  * excess_green_chlorophyll      : 0.178
  * excess_red_browning           : 0.099
  * browning_ratio                : 0.146
  * surface_necrosis_decay_pct    : 0.6
  * texture_energy                : 0.2551
  * surface_roughness_laplacian   : 1092.6
====================================================================
```

### Machine-Readable JSON Schema (`--json`)
```json
{
  "image_path": ".../T10_d01_002_a_1.jpg",
  "predicted_stage": 1,
  "stage_name": "Underripe (Hard/Intact)",
  "confidence_pct": 100.0,
  "quality_grade": "Grade A (Premium Fresh)",
  "quality_score_pct": 95.8,
  "remaining_shelf_life_days": 5.6,
  "internal_degradation_index_pct": 4.2,
  "surface_decay_pct": 0.6,
  "storage_analysis": {
    "current_conditions": {
      "temperature_c": 28.0,
      "humidity_pct": 65.0,
      "estimated_rsl_days": 5.6,
      "is_out_of_dataset_range": true
    },
    "recommended_conditions": {
      "temperature_c": 10.0,
      "humidity_pct": 85.0,
      "estimated_rsl_days": 20.4,
      "additional_days_gained": 14.8
    }
  },
  "fefo_recommendation": {
    "tier": "Standard Retail Inventory",
    "action": "Maintain at standard price; standard shelf storage",
    "discount_percent": 0,
    "routing": "Normal retail display or cool storage retention",
    "urgency": "Low (> 5 days buffer)"
  }
}
```

---

## 11. Repository Structure & Codebase Manifest

The codebase is organized into clean, modular components:

```
KalFresh_NorthSouth/
├── dataset.py            # Dataset ingestion, longitudinal RSL calculation, multithreaded Parquet feature caching
├── features.py           # 38 CV biomarker extraction (CIE Lab, HSV, RGB, GLCM texture, spatial decay, heatmap)
├── train.py              # 5-head LightGBM training (Stage Classifier, Base RSL, Temp RSL, Degradation, Quality)
├── predict.py            # End-to-end inference engine with storage conditioning, biological kinetics & FEFO routing
├── evaluate.py           # Comprehensive evaluation suite, confusion matrix & FIFO vs FEFO simulation
├── app.py                # Interactive Streamlit dashboard with Avocado Freshness Report & Dark/Light mode
├── requirements.txt      # Dependency specification
├── models/
│   └── model_bundle.joblib  # Serialized 5-model bundle and validation metrics
├── data/
│   └── avocado_features.parquet # Cached 38-feature dataset (13,604 rows)
└── Hass Avocado Ripening Photographic Dataset/ # Longitudinal ripening imagery and metadata
```

---

## 12. Installation & Step-by-Step Quickstart

### Prerequisites
- Python 3.10+
- Git

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/Tannuuuu/KalFresh_NorthSouth.git
cd KalFresh_NorthSouth

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Extract Features & Cache Dataset
Extracts all 38 computer-vision biomarkers across 13,604 images using multi-threaded CPU workers and caches to Apache Parquet:
```bash
python dataset.py
```

### Step 4: Train 5-Head Machine Learning Models
Trains the Ripening Stage Classifier, Base RSL Regressor, Temperature-Conditioned RSL Regressor, Degradation Regressor, and Quality Regressor, saving the bundle to `models/model_bundle.joblib`:
```bash
python train.py
```

### Step 5: Run Evaluation & Supply Chain Simulation
Generates the quantitative evaluation report, per-class classification metrics, confusion matrix, and FIFO vs FEFO food waste mitigation impact:
```bash
python evaluate.py
```

### Step 6: Test CLI Inference
```bash
python predict.py --sample 2 --temp 28.0 --humidity 65.0
```

### Step 7: Launch Interactive Web Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) or visit the live deployment at [https://kalfresh-northsouthgit.streamlit.app/](https://kalfresh-northsouthgit.streamlit.app/) in your browser to inspect produce, test storage microclimates, view degradation heatmaps, and toggle between Dark and Light mode.

---

## 13. Scientific References & Citations

1. **Hass Avocado Ripening Photographic Dataset:** Longitudinal post-harvest image series of Hass avocados across controlled temperature regimes ($10^\circ\text{C}, 20^\circ\text{C}, \text{Ambient}$).
2. **Arrhenius & $Q_{10}$ Respiration Kinetics:** Kays, S. J., & Paull, R. E. (2004). *Postharvest Biology*. Exbury Press.
3. **CIE $L^*a^*b^*$ Perceptual Colorimetry:** International Commission on Illumination (CIE). *Colorimetry — Part 4: 1976 $L^*a^*b^*$ Colour Spaces*.
4. **Gray-Level Co-occurrence Matrix (GLCM) Texture Analysis:** Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). *Textural Features for Image Classification*. IEEE Transactions on Systems, Man, and Cybernetics.
5. **FEFO Inventory Routing Principles:** Tromp, S. O., et al. (2012). *The impact of First-Expired, First-Out on food waste and availability in retail*. International Journal of Production Economics.

---

<p align="center">
  <b>KalFresh · Built for IIT Bombay Computer Vision AI Hackathon</b><br>
  <i>Empowering resilient, data-driven food supply chains with non-invasive computer vision.</i>
</p>
