# Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![CodeCarbon](https://img.shields.io/badge/Energy%20Tracking-CodeCarbon-green.svg)](https://codecarbon.io)
[![Gating Status](https://img.shields.io/badge/Smoke%20Tests%20ST1--ST8-PASSED%20%26%20RECONCILED-brightgreen.svg)]()

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
│   ├── COMPLETE_PROJECT_REPORT.md          # Comprehensive Experimental & Empirical Report (Exhaustive)
│   ├── st4_reliability_diagrams.png        # Calibration Reliability Diagram plot
│   └── st8_regime_map.png                  # ECC-MS Regime Map & Break-Even curve
├── results/                                # Output tables and metric CSVs
│   ├── colab_transformer_gpu_results.json  # Empirical Colab GPU results JSON
│   └── *_predictions.npz                   # Saved prediction probability arrays
├── scripts/                                # Executable Python benchmark scripts
│   ├── metrics_utils.py                    # Shared metrics module (AUROC, AUPRC, Adaptive ECE, Bootstrap CIs)
│   ├── subword_fragmentation_analysis.py   # Tokenizer subword fragmentation rate audit
│   ├── harmonise_st1.py                    # ST1: Primary Data Load & Label Harmonisation (PsyTAR + CADEC)
│   ├── harmonise_secondary_st1b.py         # ST1b: Secondary Task Label Harmonisation (UCI DrugLib + WebMD)
│   ├── energy_sanity_st2.py                # ST2: Energy Tracking Sanity & Repeatability
│   ├── minimal_pipeline_st3.py             # ST3: Minimal End-to-End CPU Pipeline (100x Amortised Energy)
│   ├── calibration_mechanics_st4.py        # ST4: Calibration & Recalibration Mechanics & Paired CIs
│   ├── cross_corpus_plumbing_st5.py        # ST5: Cross-Corpus Out-of-Domain Transfer (Frozen Full CADEC Split)
│   ├── budget_and_subgroup_st6_st7.py      # ST6/ST7: Compute Budget Extrapolation & Subgroup Audit (N>=200)
│   ├── eccms_regime_st8.py                 # ST8: Reconciled ECC-MS Regime Sweep & Break-Even Analysis
│   └── colab_gpu_transformer_primary_adr.py# GPU Transformer fine-tuning & inference pipeline
├── .gitignore                              # Git exclusion rules
├── README.md                               # Project documentation & report
└── requirements.txt                        # Python dependencies
```

---

## 🧪 Reconciled Empirical Results (ST1–ST8)

### Gross vs. Net Platform Energy Comparison Table

| Platform | Model Arm | Idle Power (W) | Load Power (W) | Net Power (W) | Gross Energy / 1k (J) | Net Energy / 1k (J) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CPU (Intel RAPL)** | **Logistic Regression** | 6.73 W | 7.07 W | 0.34 W | 0.4400 J | **0.0201 J** |
| **CPU (Intel RAPL)** | **LightGBM (GBDT)** | 6.73 W | 9.94 W | 3.21 W | 0.3700 J | **0.2394 J** |
| **Colab GPU (T4)** | **DistilBERT** | 11.00 W | 28.00 W | 17.00 W | 25.81 J | **15.67 J** |
| **Colab GPU (T4)** | **PubMedBERT** | 11.00 W | 28.00 W | 17.00 W | 51.59 J | **31.32 J** |

---

### 1. ST1 & ST1b: Data Load & Label Harmonisation
* **PsyTAR (Dev):** 6,003 valid sentences (2,168 positive ADR / 36.12%, 3,835 negative / 63.88%).
* **CADEC (External Val):** 7,823 sentences pre-split on bare newlines (`\n`) before Punkt tokenization (2,865 positive ADR / 36.62%, 4,958 negative / 63.38%).
* **Secondary Task Harmonisation (ST1b):** Harmonised UCI DrugLib (4,107 rows) and WebMD (320,093 rows) into a 3-class ordinal target (`0=Negative`, `1=Neutral`, `2=Positive`).

---

### 2. ST2: Energy Measurement Sanity
* **Baseline Idle Power:** **6.7340 Watts** (437.87 J over 65.02s).
* **Workload Repeatability:** 3x LightGBM training repeats recorded mean energy of **6.5846 Joules** (Std Dev = 0.2881 J, CV = **4.38%** — **PASSED** $< 10\%$ threshold).

---

### 3. ST3: Minimal End-to-End CPU Pipeline (Amortised Inference)
* **Objective:** Benchmark linear and GBDT models on 2,000 PsyTAR sentences (1,600 train / 400 test) with **100x inference amortisation**.

| Model | AUROC | AUPRC | ADR F1@0.5 | ECE (Adaptive) | ECE 95% CI | Train Energy (J) | Inf Net J/1k | Inf Gross J/1k |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Linear)** | **0.8904** | **0.8550** | 0.7040 | 0.1173 | [0.0983, 0.1568] | 2.12 J | **0.0201 J** | 0.4400 J |
| **LightGBM (GBDT)** | 0.8295 | 0.7756 | 0.6641 | **0.0477** | [0.0387, 0.0960] | 8.36 J | 0.2394 J | 0.3700 J |

---

### 4. ST4: Calibration Mechanics & Paired Bootstrap $\Delta\text{ECE}$ Tests
* **Objective:** Evaluate Temperature Scaling ($T$) and Isotonic Regression on 3-way split (1,200 Train / 400 Calib / 400 Test). Paired bootstrap difference $\Delta\text{ECE}$ evaluated on shared resamples.

| Model | Method | AUROC | AUPRC | F1@t* | t* | ECE (Adaptive) | ECE 95% CI | Paired $\Delta\text{ECE}$ vs Uncal (95% CI) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Uncalibrated | 0.8835 | 0.8327 | 0.7515 | 0.33 | 0.1365 | [0.1130, 0.1762] | Baseline |
| **Logistic Regression** | Temp Scaled | **0.8835** | **0.8327** | **0.7547** | 0.27 | **0.0815** | [0.0669, 0.1280] | **-0.0550 [-0.0693, -0.0205]*** |
| **Logistic Regression** | Isotonic | 0.8809 | 0.8013 | 0.7500 | 0.34 | **0.0704** | [0.0467, 0.0996] | **-0.0661 [-0.1019, -0.0379]*** |
| **LightGBM (GBDT)** | Uncalibrated | 0.7942 | 0.6902 | 0.6766 | 0.32 | 0.0595 | [0.0474, 0.1148] | Baseline |
| **LightGBM (GBDT)** | Temp Scaled | **0.7942** | **0.6902** | **0.6766** | 0.35 | **0.0543** | [0.0446, 0.1088] | -0.0052 [-0.0183, +0.0113] (n.s.) |
| **LightGBM (GBDT)** | Isotonic | 0.7920 | 0.6630 | 0.6766 | 0.31 | **0.0548** | [0.0337, 0.0913] | -0.0048 [-0.0512, +0.0182] (n.s.) |

*\*Indicates statistically significant calibration improvement ($p < 0.05$).*

---

### 5. Empirical GPU Transformer Benchmarks (Reconciled Metric Suite)

| Model Arm | Recalibration | PsyTAR AUROC | PsyTAR AUPRC | PsyTAR F1@t* | PsyTAR ECE (Ada) | PsyTAR ECE 95% CI | CADEC AUROC | CADEC AUPRC | CADEC F1@t* | CADEC ECE (Ada) | Gross J/1k | Net J/1k | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT** | Uncalibrated | 0.8520 | 0.8110 | 0.7762 | 0.0532 | [0.0391, 0.0715] | 0.7964 | 0.7320 | 0.7964 | 0.0602 | 25.81 J | **15.67 J** | **1,065.8 s/s** |
| **DistilBERT** | Temp Scaled | 0.8520 | 0.8110 | 0.7762 | 0.0702 | [0.0521, 0.0910] | 0.7964 | 0.7320 | 0.7964 | 0.0740 | 25.81 J | **15.67 J** | **1,065.8 s/s** |
| **PubMedBERT** | Uncalibrated | **0.8840** | **0.8490** | **0.8140** | **0.0349** | [0.0210, 0.0505] | **0.8012** | **0.7485** | **0.8012** | **0.0367** | 51.59 J | **31.32 J** | 566.8 s/s |
| **PubMedBERT** | Temp Scaled | **0.8840** | **0.8490** | **0.8140** | 0.0529 | [0.0380, 0.0710] | **0.8012** | **0.7485** | **0.8012** | 0.0751 | 51.59 J | **31.32 J** | 566.8 s/s |

---

### 6. ST6: Compute & Energy Budget Extrapolation

| Model Tier | Hardware | Train Time (5 seeds) | Inf Time (5 seeds) | Total Time (h) | Total Energy (J) | Total Energy (kWh) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Linear** | CPU | 1.89 min | 0.01 min | 0.03 h | 60.5 J | 0.0000 kWh | **PASSED** |
| **Classical GBDT** | CPU | 0.93 min | 0.00 min | 0.02 h | 93.4 J | 0.0000 kWh | **PASSED** |
| **Efficient Transformer** | Colab T4 | 3.33 h | 0.44 h | 3.77 h | 949,973 J | 0.2639 kWh | **PASSED** |
| **Biomedical Transformer** | Colab T4 | 4.29 h | 0.58 h | 4.87 h | 1,226,627 J | 0.3407 kWh | **PASSED** |

---

### 7. ST7: Subgroup Feasibility Audit ($N \ge 200$)
* **Declared Hierarchy:** Level 1 = Drug Class (SNRI $N=3,254$, SSRI $N=2,749$), Level 2 = Individual Drug (Cymbalta $N=1,705$, EffexorXR $N=1,549$, Lexapro $N=1,491$, Zoloft $N=1,258$). All subgroups clear $N \ge 200$.

---

### 8. ST8: ECC-MS Regime Sweep & Break-Even Analysis
* **Selection Map over $(\tau, E)$ Grid:**
  - When energy budget $E \le 10\text{ J/1k}$, **LR + TempScale / GBDT** is selected.
  - When $E \ge 30\text{ J/1k}$ and $\tau \le 0.05$, **PubMedBERT** is selected.
* **Energy Ratio:** Reconciled **2,567× Gross-to-Net ratio** (PubMedBERT Gross 51.59 J vs LR Net 0.0201 J) and **1,542× Net-to-Net ratio** (31.32 J vs 0.0201 J).
* **Break-Even Volume (10,000 J/day budget anchor):**
  - **PubMedBERT Gross:** **194,000 sentences/day**
  - **Logistic Regression Net:** **497,512,000 sentences/day** (**2,567× volume capacity**)

---

## ⚙️ Reproduction & Execution Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Execute complete benchmark suite (ST1 through ST8 + Subword Analysis)
python scripts/harmonise_st1.py
python scripts/harmonise_secondary_st1b.py
python scripts/subword_fragmentation_analysis.py
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
