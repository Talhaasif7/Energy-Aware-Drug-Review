# Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https.mit-license.org)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![CodeCarbon](https://img.shields.io/badge/Energy%20Tracking-CodeCarbon-green.svg)](https://codecarbon.io)
[![Gating Status](https://img.shields.io/badge/Smoke%20Tests%20ST1--ST7-PASSED-brightgreen.svg)]()

This repository contains the complete experimental framework and empirical codebase for **"Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals"**.

The project introduces the **ECC-MS (Energy–Calibration Constrained Model Selection)** framework for clinical NLP. It evaluates the multi-objective Pareto front balancing **predictive accuracy (F1-score)**, **probability calibration (Expected Calibration Error / ECE, Brier Score, NLL)**, and **computational energy consumption (Joules, kWh)** in Adverse Drug Reaction (ADR) detection.

---

## 📋 Core Research Questions (RQs)

* **RQ1 (Predictive-Energy Pareto Front):** How do classical CPU model arms (Linear, GBDT) compare to Transformer arms (Efficient, Biomedical) in trade-offs between ADR F1-score and energy consumption (Joules/kWh)?
* **RQ2 (Calibration & Post-Hoc Recalibration):** Can near-zero energy post-hoc recalibration (Temperature Scaling, Isotonic Regression) effectively mitigate overconfidence and reduce ECE without degrading predictive discrimination?
* **RQ3 (Cross-Corpus Transfer & Distribution Shift):** How well do source-fitted recalibrators transfer out-of-domain under covariate shift (PsyTAR $\rightarrow$ CADEC zero-shot transfer)?
* **RQ4 (Subgroup Fairness & Reliability):** How does calibration quality vary across patient drug subgroups, and what are minimum sample size bounds ($N \ge 50$) for statistically reliable 10-bin ECE estimation?

---

## 📁 Repository Architecture

```text
e:\AI Green\
├── configs/                                # Experimental hyperparameter configs
├── data/                                   # Datasets (Harmonised & Raw)
│   ├── 01_primary_adr_detection/
│   │   ├── dev_psytar/                     # PsyTAR Development Corpus
│   │   │   ├── PsyTAR_dataset.xlsx         # Raw Excel spreadsheet
│   │   │   └── psytar_harmonised.csv       # Harmonised dataset (6,003 rows)
│   │   └── external_val_cadec/             # CADEC External Validation Corpus
│   │       ├── cadec/                      # Raw text & Brat annotations
│   │       └── cadec_harmonised.csv        # Harmonised dataset (7,681 rows)
│   └── 02_secondary_sentiment_scaling/     # Secondary Task Datasets
│       ├── dev_uci_drug_review/            # UCI Drug Review dataset (3,076 rows)
│       └── external_val_webmd/             # WebMD dataset (320,096 rows)
├── reports/                                # Generated figures & visual artifacts
│   └── st4_reliability_diagrams.png        # Calibration Reliability Diagram plot
├── results/                                # Output tables and metric CSVs
├── scripts/                                # Executable Python benchmark scripts
│   ├── harmonise_st1.py                    # ST1: Data Load & Label Harmonisation
│   ├── energy_sanity_st2.py                # ST2: Energy Tracking Sanity
│   ├── minimal_pipeline_st3.py             # ST3: Minimal End-to-End CPU Pipeline
│   ├── calibration_mechanics_st4.py        # ST4: Calibration & Recalibration Mechanics
│   ├── cross_corpus_plumbing_st5.py        # ST5: Cross-Corpus Out-of-Domain Transfer
│   └── budget_and_subgroup_st6_st7.py      # ST6/ST7: Budget Extrapolation & Subgroup Audit
├── .gitignore                              # Git exclusion rules
├── README.md                               # Project documentation & report
└── requirements.txt                        # Python dependencies
```

---

## 🧪 Smoke Test Gating Milestones & Empirical Results (ST1–ST7)

All seven preliminary gating tests (Smoke Tests ST1 through ST7) have been executed and validated:

### 1. ST1: Data Load & Label Harmonisation
* **Objective:** Harmonise raw PsyTAR (Excel) and CADEC (Brat text/annotations) corpora into standardized binary sentence-level schema `['text', 'label']`.
* **PsyTAR (Dev):** Extracted 6,003 valid sentences (`Sentence_Labeling` sheet).
* **CADEC (External Val):** Parsed 7,409 ADR character spans across 1,250 text posts using NLTK `PunktSentenceTokenizer.span_tokenize` for exact character boundary mapping (7,681 total sentences).

### 2. ST2: Energy Measurement Sanity
* **Objective:** Verify CodeCarbon energy tracking engine and measure baseline idle draw.
* **Environment:** Windows 11 AMD64 (Intel64 4 physical / 8 logical cores, 19.82 GB RAM).
* **Baseline Idle Power:** **0.0930 Watts** (5.9439 Joules over 60s idle sleep).
* **Workload Repeatability:** 3x GBDT training repeats recorded mean energy of 0.0885 J (CV = 14.02%).

### 3. ST3: Minimal End-to-End CPU Pipeline
* **Objective:** Benchmark TF-IDF (1,000 features) on a 2,000 unit stratified PsyTAR subset (1,600 train / 400 test).
* **Results Table:**

| Model | Macro F1 | ADR F1 (Class 1) | ECE (10-bin) | Brier Score | Train Time (s) | Train Energy (J) | Inf Energy/1k (J) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Linear)** | 0.7847 | 0.7040 | 0.1204 | 0.1394 | 6.041 s | 3.2038 J | 0.0228 J |
| **LightGBM (GBDT)** | 0.7503 | 0.6641 | 0.0450 | 0.1545 | 2.975 s | 4.9648 J | 0.0161 J |

* **Validation:** Verified probabilities are non-NaN, strictly bounded in $[0, 1]$, and sum to $1.0$.

### 4. ST4: Calibration & Post-Hoc Recalibration Mechanics
* **Objective:** Evaluate Temperature Scaling ($T$) and Isotonic Regression on a stratified 3-way split (1,200 Train / 400 Calib / 400 Test).
* **Fitted Parameters:** $T_{\text{Logistic Regression}} = 0.6251$, $T_{\text{LightGBM}} = 1.2418$.
* **Results Table:**

| Model | Method | Macro F1 | ADR F1 | ECE (10-bin) | Brier Score | NLL | Fit Time (ms) | Fit Energy (J) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Uncalibrated | 0.7325 | 0.6167 | 0.1451 | 0.1499 | 0.4709 | 0.00 ms | 0.000000 J |
| **Logistic Regression** | Temperature Scaled | 0.7325 | 0.6167 | **0.0802** | **0.1398** | **0.4345** | 7.06 ms | 0.000005 J |
| **Logistic Regression** | Isotonic Regression | **0.7400** | **0.6288** | **0.0555** | **0.1357** | 0.5865 | 2.52 ms | 0.007148 J |
| **LightGBM (GBDT)** | Uncalibrated | 0.6933 | 0.5837 | 0.0600 | 0.1747 | 0.5241 | 0.00 ms | 0.000000 J |
| **LightGBM (GBDT)** | Temperature Scaled | 0.6933 | 0.5837 | 0.0634 | **0.1734** | **0.5195** | 2.78 ms | 0.005347 J |
| **LightGBM (GBDT)** | Isotonic Regression | 0.6278 | 0.4528 | **0.0348** | **0.1730** | **0.5148** | 2.70 ms | 0.004242 J |

* **Artifact:** Generated 1x2 Reliability Diagram saved to [`reports/st4_reliability_diagrams.png`](file:///e:/AI%20Green/reports/st4_reliability_diagrams.png).

### 5. ST5: Cross-Corpus Out-of-Domain Transfer
* **Objective:** Train & fit recalibrators on PsyTAR (Source), evaluate zero-shot on CADEC (Target, $N=1,500$).
* **Vocabulary Alignment:** PsyTAR TF-IDF achieved 73.1% feature coverage (731 overlapping tokens) on CADEC with clean OOV handling.
* **Results Table (CADEC Zero-Shot Target):**

| Model | Method | CADEC ADR F1 | CADEC Macro F1 | CADEC ECE | CADEC Brier | CADEC NLL |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Uncalibrated | 0.5282 | 0.6734 | 0.0939 | 0.1765 | 0.5316 |
| **Logistic Regression** | Temperature Scaled (Transfer) | 0.5282 | 0.6734 | **0.0816** | **0.1746** | **0.5234** |
| **Logistic Regression** | Isotonic Regression (Transfer) | **0.6000** | **0.7143** | **0.0666** | **0.1689** | 0.6027 |
| **LightGBM (GBDT)** | Uncalibrated | 0.5613 | 0.6821 | 0.0244 | 0.1852 | 0.5518 |
| **LightGBM (GBDT)** | Temperature Scaled (Transfer) | 0.5613 | 0.6821 | 0.0260 | **0.1849** | **0.5504** |

### 6. ST6: Compute & Energy Budget Extrapolation
* **Objective:** Extrapolate compute time and energy across 5 random seeds for full experimental matrix.
* **Extrapolation Table:**

| Model Tier | Hardware | Train Time (5 seeds) | Inf Time (5 seeds) | Total Time (h) | Total Energy (kWh) | Feasibility Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Linear** | CPU | 1.89 mins | 0.42 mins | 0.04 h | 0.0000 kWh | **PASSED (Negligible)** |
| **Classical GBDT** | CPU | 0.93 mins | 0.30 mins | 0.02 h | 0.0000 kWh | **PASSED (Negligible)** |
| **Efficient Transformer** | Colab T4 GPU | 3.33 h | 0.44 h | 3.77 h | 0.2639 kWh | **PASSED (< 12h Free Limit)** |
| **Biomedical Transformer** | Colab T4 GPU | 4.29 h | 0.58 h | 4.87 h | 0.3407 kWh | **PASSED (< 12h Free Limit)** |

### 7. ST7: Subgroup Feasibility Audit
* **Objective:** Inspect sample count $N$ per drug class subgroup to ensure statistical reliability of 10-bin ECE estimation ($N \ge 50$).
* **Reliable Subgroups ($N \ge 50$):**
  * PsyTAR: `SNRI` (3,254), `SSRI` (2,749), `Cymbalta` (1,705), `EffexorXR` (1,549), `Lexapro` (1,491), `Zoloft` (1,258).
  * CADEC: `LIPITOR` (~6,000 sents), `ARTHROTEC` (~870 sents), `VOLTAREN` (~276 sents), `VOLTAREN-XR` (~132 sents), `CATAFLAM` (~60 sents).
* **Underpowered Subgroups ($N < 50$):** `Diclofenac-Sodium` (~42 sents), `Zipsor` (~30 sents), `Cambia` (~24 sents), `Pennsaid` (~24 sents), `Diclofenac-Potassium` (~18 sents), `Solaraze` (~18 sents), `Flector` (~6 sents).
* **Rule:** Aggregate underpowered rare drug classes into macro-categories to maintain 10-bin ECE calculation integrity.

---

## ⚙️ Reproduction & Execution Instructions

### 1. Installation & Environment Setup
Clone the repository and install requirements:
```bash
git clone https://github.com/Talhaasif7/Energy-Aware-Drug-Review.git
cd Energy-Aware-Drug-Review
pip install -r requirements.txt
```

### 2. Running Smoke Test Benchmark Scripts
Execute each benchmark script directly from the project root:

```bash
# ST1: Data Load & Label Harmonisation
python scripts/harmonise_st1.py

# ST2: Energy Tracking Sanity Check
python scripts/energy_sanity_st2.py

# ST3: Minimal End-to-End CPU Pipeline
python scripts/minimal_pipeline_st3.py

# ST4: Calibration & Recalibration Mechanics
python scripts/calibration_mechanics_st4.py

# ST5: Cross-Corpus Out-of-Domain Transfer
python scripts/cross_corpus_plumbing_st5.py

# ST6 & ST7: Compute Budget Extrapolation & Subgroup Audit
python scripts/budget_and_subgroup_st6_st7.py
```

### 3. Google Colab GPU Execution Guidelines (Transformers)
For GPU fine-tuning of DistilBERT/PubMedBERT:
1. Open a Google Colab instance and set Hardware Accelerator to **T4 GPU** (`Runtime > Change runtime type`).
2. Install dependencies:
   ```bash
   !pip install codecarbon transformers datasets torch
   ```
3. Wrap model training in CodeCarbon `EmissionsTracker(save_to_file=False)` to record GPU Wh energy per epoch.

---

## 📜 License & Citation

This project is licensed under the MIT License.

```bibtex
@article{eccms2026energy,
  title={Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals},
  author={Asif, Talha et al.},
  journal={Clinical NLP & Energy-Aware Machine Learning},
  year={2026}
}
```
