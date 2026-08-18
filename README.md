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
├── configs/                                # Experimental hyperparameter configs
│   └── default_config.json                 # Benchmark configuration parameters
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
│   ├── st1_st5_review_closure.md           # ST1 & ST5 Review Closure & Verification Report
│   ├── st2_energy_sanity_report.md         # ST2 Linux Intel RAPL Energy Sanity Report
│   └── st4_reliability_diagrams.png        # Calibration Reliability Diagram plot
├── results/                                # Output tables and metric CSVs
│   └── colab_transformer_gpu_results.json  # Empirical Colab GPU results JSON
├── scripts/                                # Executable Python benchmark scripts
│   ├── colab_gpu_transformer_primary_adr.py# Colab T4 GPU Transformer Fine-Tuning & Energy Tracking
│   ├── verify_st1_gaps.py                  # ST1 & ST5 Review Closure Verification Script
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
* **PsyTAR (Dev):** Extracted 6,003 valid sentences (2,168 positive ADR / 36.12%, 3,835 negative / 63.88%).
* **CADEC (External Val):** Parsed 7,409 ADR character spans across 1,250 text posts using NLTK `PunktSentenceTokenizer.span_tokenize` for exact character boundary mapping (7,681 total sentences; 2,854 positive ADR / 37.16%, 4,827 negative / 62.84%).

### 2. ST2: Energy Measurement Sanity
* **Objective:** Verify CodeCarbon energy tracking engine and measure baseline idle draw using direct Intel RAPL sysfs access.
* **Environment:** Linux 7.0.0-29-generic (x86_64, 6 physical / 6 logical cores, 14.9 GB RAM).
* **Energy Interface:** Linux Intel RAPL (`/sys/class/powercap/intel-rapl`) via CodeCarbon Engine.
* **Baseline Idle Power:** **6.7340 Watts** (437.8717 Joules over 65.02s sleep).
* **Workload Repeatability:** 3x LightGBM training repeats on 10,000 synthetic rows recorded mean energy of **6.5846 Joules** (Std Dev = 0.2881 J, CV = **4.38%** — **PASSED** $< 10\%$ threshold).
* **Verification Outcome:** Direct Intel RAPL sysfs access resolved coarse Windows TDP estimates, achieving high repeatability (CV = 4.38%). Saved to [`reports/st2_energy_sanity_report.md`](file:///e:/AI%20Green/reports/st2_energy_sanity_report.md).

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
| **LightGBM (GBDT)** | Isotonic Regression (Transfer) | 0.5598 | 0.6839 | 0.0535 | 0.1903 | 0.6082 |

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

### 8. Empirical Google Colab T4 GPU Transformer Benchmark (ST3b & ST6 Real Validation)
* **Objective:** Real empirical fine-tuning, probability calibration, and CodeCarbon GPU energy tracking executed on a Google Colab T4 GPU instance.
* **Corpora Used:**
  - **Source Dev Corpus (PsyTAR):** Harmonised sentence dataset (6,003 rows; 2,000 unit stratified subset for ST3b gating).
  - **Target External Val Corpus (CADEC):** Harmonised zero-shot evaluation target (7,681 rows).
* **Hardware & Acceleration:** Google Colab NVIDIA Tesla T4 GPU with PyTorch Mixed Precision (`fp16`).
* **Empirical Results Table (`colab_transformer_gpu_results.json`):**

| Model Tier | Method | PsyTAR ADR F1 | PsyTAR ECE-U | PsyTAR NLL | CADEC ADR F1 | CADEC ECE-U | CADEC NLL | Train Time (s) | Train Energy (J) | Inf Throughput | Inf Energy/1k |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Efficient Transformer** (`distilbert-base-uncased`) | Uncalibrated | 0.7762 | 0.0532 | 0.3567 | 0.7976 | 0.0590 | 0.3514 | 13.13 s | 126.99 J | 1,034.0 s/s | 10.20 J |
| **Efficient Transformer** (`distilbert-base-uncased`) | Temperature Scaled | 0.7762 | 0.0702 | 0.3613 | 0.7976 | 0.0729 | 0.3581 | 13.13 s | 126.99 J | 1,034.0 s/s | 10.20 J |
| **Efficient Transformer** (`distilbert-base-uncased`) | Isotonic Regression | 0.7063 | 0.0584 | 0.5234 | 0.7831 | 0.0506 | 0.4205 | 13.13 s | 126.99 J | 1,034.0 s/s | 10.20 J |
| **Biomedical Transformer** (`PubMedBERT-base-uncased`) | Uncalibrated | **0.8140** | **0.0366** | **0.3364** | **0.8008** | **0.0367** | **0.3384** | 13.60 s | 150.39 J | 585.2 s/s | 18.50 J |
| **Biomedical Transformer** (`PubMedBERT-base-uncased`) | Temperature Scaled | **0.8140** | 0.0528 | 0.3461 | **0.8008** | 0.0745 | 0.3556 | 13.60 s | 150.39 J | 585.2 s/s | 18.50 J |
| **Biomedical Transformer** (`PubMedBERT-base-uncased`) | Isotonic Regression | 0.7177 | **0.0369** | 0.5263 | 0.7711 | **0.0352** | 0.3495 | 13.60 s | 150.39 J | 585.2 s/s | 18.50 J |

* **Empirical Takeaways:**
  - **Predictive Superiority:** `PubMedBERT-base-uncased` achieved highest discrimination across both PsyTAR (ADR F1 = **0.8140**) and zero-shot CADEC target (ADR F1 = **0.8008**) while maintaining intrinsic low calibration error (ECE = **0.0366**).
  - **Energy-Throughput Trade-off:** `DistilBERT` delivered **1,034.0 sentences/sec** inference throughput (1.77x faster than PubMedBERT) with only **10.20 J / 1,000 sentences** (44.9% energy reduction), demonstrating high utility for large-scale real-time screening.

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

### 3. Google Colab T4 GPU Execution Guidelines (Transformers)
For GPU fine-tuning, energy tracking, calibration, and zero-shot transfer of DistilBERT/PubMedBERT:
1. Open [Google Colab](https://colab.research.google.com/) and set Hardware Accelerator to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).
2. Run the automated script execution commands:
   ```bash
   !pip install codecarbon transformers datasets accelerate evaluate torch pandas numpy scikit-learn scipy
   !git clone https://github.com/Talhaasif7/Energy-Aware-Drug-Review.git
   %cd Energy-Aware-Drug-Review
   !python scripts/colab_gpu_transformer_primary_adr.py
   ```
3. The script automatically runs fine-tuning, CodeCarbon energy tracking, recalibration, evaluation on PsyTAR & CADEC, prints Markdown result tables, and exports `colab_transformer_gpu_results.json`.

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
