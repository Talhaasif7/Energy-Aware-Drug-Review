# Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![CodeCarbon](https://img.shields.io/badge/Energy%20Tracking-CodeCarbon-green.svg)](https://codecarbon.io)
[![Gating Status](https://img.shields.io/badge/Smoke%20Tests%20ST1--ST8-PASSED-brightgreen.svg)]()

This repository contains the complete experimental framework and empirical codebase for **"Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals"**.

The project introduces the **ECC-MS (Energy–Calibration Constrained Model Selection)** framework for clinical NLP. It evaluates the multi-objective Pareto front balancing **predictive discrimination (AUROC, AUPRC, threshold-tuned F1)**, **probability calibration (Adaptive Equal-Mass ECE with Bootstrap CIs, Brier Score, NLL)**, and **computational energy consumption (Joules, kWh, load Wattage)** in Adverse Drug Reaction (ADR) detection.

---

## 📋 Core Research Questions (RQs)

* **RQ1 (Predictive-Energy Pareto Front):** How do classical CPU model arms (Linear, GBDT) compare to Transformer arms (Efficient, Biomedical) in trade-offs between ADR discrimination (AUROC/AUPRC) and energy consumption (Joules/kWh)?
* **RQ2 (Calibration & Post-Hoc Recalibration):** Can near-zero energy post-hoc recalibration (Temperature Scaling, Isotonic Regression) effectively mitigate overconfidence and reduce ECE without degrading predictive discrimination?
* **RQ3 (Cross-Corpus Transfer & Distribution Shift):** How well do source-fitted recalibrators transfer out-of-domain under covariate shift (PsyTAR $\rightarrow$ CADEC zero-shot transfer)?
* **RQ4 (Subgroup Fairness & Reliability):** How does calibration quality vary across patient drug subgroups, and what are minimum sample size bounds ($N \ge 200$) for statistically reliable adaptive ECE estimation?
* **RQ5 (ECC-MS Framework Selection):** Under what inference volume and energy budget constraints ($E$) does the framework transition selection between lightweight classical models and high-capacity transformer models?

---

## 📁 Repository Architecture

```text
├── configs/                                # Experimental hyperparameter configs
│   ├── default_config.json                 # Benchmark configuration parameters
│   └── gpu_energy_protocol.md              # GPU energy tracking & manual nvidia-smi protocol
├── data/                                   # Datasets (Harmonised & Raw)
│   ├── 01_primary_adr_detection/
│   │   ├── dev_psytar/                     # PsyTAR Development Corpus
│   │   │   ├── PsyTAR_dataset.xlsx         # Raw Excel spreadsheet
│   │   │   └── psytar_harmonised.csv       # Harmonised dataset (6,003 rows)
│   │   └── external_val_cadec/             # CADEC External Validation Corpus
│   │       ├── cadec/                      # Raw text & Brat annotations
│   │       └── cadec_harmonised.csv        # Harmonised dataset (7,823 sentences, pre-split on \n)
│   └── 02_secondary_sentiment_scaling/     # Secondary Task Datasets
│       ├── dev_uci_drug_review/            # UCI DrugLib dataset (4,107 rows)
│       │   └── uci_druglib_harmonised.csv  # 3-class ordinal harmonised
│       └── external_val_webmd/             # WebMD dataset (320,093 rows)
│           └── webmd_harmonised.csv        # 3-class ordinal harmonised
├── reports/                                # Generated figures & visual artifacts
│   ├── st4_reliability_diagrams.png        # Calibration Reliability Diagram plot
│   └── st8_regime_map.png                  # ECC-MS Regime Map & Break-Even curve
├── results/                                # Output tables and metric CSVs
│   └── colab_transformer_gpu_results.json  # Empirical Colab GPU results JSON
├── scripts/                                # Executable Python benchmark scripts
│   ├── metrics_utils.py                    # Shared metrics module (AUROC, AUPRC, Adaptive ECE, Bootstrap CIs)
│   ├── harmonise_st1.py                    # ST1: Primary Data Load & Label Harmonisation (PsyTAR + CADEC)
│   ├── harmonise_secondary_st1b.py         # ST1b: Secondary Task Label Harmonisation (UCI DrugLib + WebMD)
│   ├── energy_sanity_st2.py                # ST2: Energy Tracking Sanity & Repeatability
│   ├── minimal_pipeline_st3.py             # ST3: Minimal End-to-End CPU Pipeline (100x Amortised Energy)
│   ├── calibration_mechanics_st4.py        # ST4: Calibration & Recalibration Mechanics
│   ├── cross_corpus_plumbing_st5.py        # ST5: Cross-Corpus Out-of-Domain Transfer (Frozen Full CADEC Split)
│   ├── budget_and_subgroup_st6_st7.py      # ST6/ST7: Compute Budget Extrapolation & Subgroup Audit (N>=200)
│   └── eccms_regime_st8.py                 # ST8: ECC-MS Regime Sweep & Break-Even Inference Volume Analysis
├── .gitignore                              # Git exclusion rules
├── README.md                               # Project documentation & report
└── requirements.txt                        # Python dependencies
```

---

## 🧪 Smoke Test Gating Milestones & Corrected Empirical Results (ST1–ST8)

All eight preliminary gating tests (Smoke Tests ST1 through ST8) have been executed and validated under strict statistical and measurement standards:

### 1. ST1 & ST1b: Data Load & Label Harmonisation
* **PsyTAR (Dev):** 6,003 valid sentences (2,168 positive ADR / 36.12%, 3,835 negative / 63.88%).
* **CADEC (External Val):** 7,823 sentences pre-split on bare newlines (`\n`) before Punkt tokenization to handle forum post bullet points cleanly. Parsed 7,409 ADR character spans (2,865 positive ADR / 36.62%, 4,958 negative / 63.38%).
* **CADEC 50-Unit Audit:** Verified sentence length distributions (PsyTAR median=65 chars, CADEC median=62 chars) with 0 embedded newlines.
* **Secondary Task Harmonisation (ST1b):** Harmonised UCI DrugLib (4,107 rows) and WebMD (320,093 rows) into a 3-class ordinal target (`0=Negative`, `1=Neutral`, `2=Positive`) based on effectiveness. Mapping locked before inspection.

### 2. ST2: Energy Measurement Sanity
* **Baseline Idle Power:** **6.7340 Watts** (437.87 J over 65.02s).
* **Workload Repeatability:** 3x LightGBM training repeats recorded mean energy of **6.5846 Joules** (Std Dev = 0.2881 J, CV = **4.38%** — **PASSED** $< 10\%$ threshold).

### 3. ST3: Minimal End-to-End CPU Pipeline (Amortised Inference)
* **Objective:** Benchmark linear and GBDT models on 2,000 PsyTAR sentences (1,600 train / 400 test) with **100x inference amortisation** to exceed hardware sensor polling resolution.

| Model | AUROC | AUPRC | ADR F1@0.5 | ECE (Adaptive) | ECE 95% CI | Train Energy (J) | Inf Energy/1k (J) | Inf Load (W) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Linear)** | **0.8904** | **0.8550** | 0.7040 | 0.1173 | [0.0983, 0.1568] | 2.12 J | **0.0201 J** | 0.34 W |
| **LightGBM (GBDT)** | 0.8295 | 0.7756 | 0.6641 | **0.0477** | [0.0387, 0.0960] | 8.36 J | 0.2394 J | 3.21 W |

### 4. ST4: Calibration & Recalibration Mechanics (Threshold-Tuned F1)
* **Objective:** Evaluate Temperature Scaling ($T$) and Isotonic Regression on 3-way split (1,200 Train / 400 Calib / 400 Test). Threshold tuned on calibration split ($t^*$).

| Model | Method | AUROC | AUPRC | F1@t* | t* | F1@0.5 | ECE (Adaptive) | ECE 95% CI | Fit (ms) | Fit (J) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Uncalibrated | 0.8835 | 0.8327 | 0.7515 | 0.33 | 0.6167 | 0.1365 | [0.1130, 0.1762] | 0.00 ms | 0.0000 J |
| **Logistic Regression** | Temp Scaled | **0.8835** | **0.8327** | **0.7547** | 0.27 | 0.6167 | **0.0815** | [0.0669, 0.1280] | 4.38 ms | 0.0000 J |
| **Logistic Regression** | Isotonic | 0.8809 | 0.8013 | 0.7500 | 0.34 | 0.6288 | **0.0704** | [0.0467, 0.0996] | 2.38 ms | 0.0108 J |
| **LightGBM (GBDT)** | Uncalibrated | 0.7942 | 0.6902 | 0.6766 | 0.32 | 0.5837 | 0.0595 | [0.0474, 0.1148] | 0.00 ms | 0.0000 J |
| **LightGBM (GBDT)** | Temp Scaled | **0.7942** | **0.6902** | **0.6766** | 0.35 | 0.5837 | **0.0543** | [0.0446, 0.1088] | 2.29 ms | 0.0074 J |
| **LightGBM (GBDT)** | Isotonic | 0.7920 | 0.6630 | 0.6766 | 0.31 | 0.4528 | **0.0548** | [0.0337, 0.0913] | 2.49 ms | 0.0054 J |

* **Key Finding:** Temperature scaling preserves AUROC identically (0.8835), while threshold tuning ($t^*$) recovers F1 discrimination.

### 5. ST5: Cross-Corpus Out-of-Domain Transfer (Frozen Full CADEC Split)
* **Objective:** Zero-shot transfer from PsyTAR to full CADEC dataset ($N=7,823$).

| Model | Method | AUROC | AUPRC | F1@t* | t* | F1@0.5 | ECE (Adaptive) | ECE 95% CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Uncalibrated | 0.8200 | 0.7212 | 0.6518 | 0.42 | 0.5262 | 0.1013 | [0.0929, 0.1115] |
| **Logistic Regression** | Temp Scaled (Transfer) | **0.8200** | **0.7212** | **0.6603** | 0.36 | 0.5262 | **0.0771** | [0.0695, 0.0865] |
| **Logistic Regression** | Isotonic (Transfer) | 0.8061 | 0.6720 | 0.6486 | 0.31 | 0.5893 | **0.0519** | [0.0437, 0.0610] |
| **LightGBM (GBDT)** | Uncalibrated | 0.7499 | 0.6530 | 0.6043 | 0.22 | 0.5522 | 0.0328 | [0.0291, 0.0448] |
| **LightGBM (GBDT)** | Temp Scaled (Transfer) | **0.7499** | **0.6530** | **0.6041** | 0.23 | 0.5522 | **0.0272** | [0.0239, 0.0401] |

### 6. ST6: Compute & Energy Budget Extrapolation
* **Parameters:** PsyTAR (6,003), CADEC (7,681), Secondary CPU (324,204), Secondary Transformer cap (30,000), 5 seeds.

| Model Tier | Hardware | Train Time (5 seeds) | Inf Time (5 seeds) | Total Time (h) | Total Energy (J) | Total Energy (kWh) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Linear** | CPU | 1.89 min | 0.01 min | 0.03 h | 60.5 J | 0.0000 kWh | **PASSED** |
| **Classical GBDT** | CPU | 0.93 min | 0.00 min | 0.02 h | 93.4 J | 0.0000 kWh | **PASSED** |
| **Efficient Transformer** | Colab T4 | 3.33 h | 0.44 h | 3.77 h | 949,973 J | 0.2639 kWh | **PASSED (<12h)** |
| **Biomedical Transformer** | Colab T4 | 4.29 h | 0.58 h | 4.87 h | 1,226,627 J | 0.3407 kWh | **PASSED (<12h)** |

### 7. ST7: Subgroup Feasibility Audit ($N \ge 200$)
* **Declared Hierarchy:** Level 1 = Drug Class (SNRI, SSRI), Level 2 = Individual Drug (Cymbalta, EffexorXR, Lexapro, Zoloft). All PsyTAR subgroups clear $N \ge 200$.
* **CADEC Composition:** Excluded from subgroup calibration because 78% of sentences belong to `LIPITOR`, making subgroup analysis uninformative.

### 8. ST8: ECC-MS Regime Sweep & Break-Even Analysis
* **Selection Map over $(\tau, E)$ Grid:**
  - When energy budget $E \le 10\text{ J/1k}$, **LR + Isotonic / TempScale** is selected.
  - When $E > 10\text{ J/1k}$ and $\tau \ge 0.03$, **PubMedBERT** is selected.
* **Energy Ratio:** Transformer inference costs **1,156x more energy per 1k sentences** than classical linear model ($18.5\text{ J/1k}$ vs $0.016\text{ J/1k}$).
* **Break-Even Volume (1,000 J/day budget):** Transformer serves up to ~54k sentences/day; Classical model serves up to ~62.5M sentences/day.

---

## ⚙️ Reproduction & Execution Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Execute smoke test benchmark suite (ST1 through ST8)
python scripts/harmonise_st1.py
python scripts/harmonise_secondary_st1b.py
python scripts/energy_sanity_st2.py
python scripts/minimal_pipeline_st3.py
python scripts/calibration_mechanics_st4.py
python scripts/cross_corpus_plumbing_st5.py
python scripts/budget_and_subgroup_st6_st7.py
python scripts/eccms_regime_st8.py
```

---

## 📜 License & Citation

Licensed under the MIT License.

```bibtex
@article{eccms2026energy,
  title={Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals},
  author={Asif, Talha et al.},
  journal={Clinical NLP & Energy-Aware Machine Learning},
  year={2026}
}
```
