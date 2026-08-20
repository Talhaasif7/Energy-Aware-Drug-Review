# Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![CodeCarbon](https://img.shields.io/badge/Energy%20Tracking-CodeCarbon-green.svg)](https://codecarbon.io)
[![Status](https://img.shields.io/badge/Empirical%20Suite-100%25%20VERIFIED-brightgreen.svg)]()

This repository contains the complete experimental framework and empirical codebase for **"Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals"**.

The project introduces the **ECC-MS (Energy–Calibration Constrained Model Selection)** framework with a **statistical-tie rule**: when candidate models' AUROCs fall within a bootstrap confidence interval margin of error ($\Delta\text{AUROC} \le 0.02$), ECC-MS declares a statistical tie and selects the **lowest-energy model** clearing the calibration constraint $\tau$ — avoiding fifth-decimal over-allocation to models that are 100× more energy-intensive.

---

## 📋 Core Research Questions (RQs)

* **RQ1 (Predictive-Energy Pareto Front):** How do classical CPU model arms (Linear, GBDT) compare to Transformer arms (Efficient, Biomedical) in trade-offs between ADR discrimination (AUROC, AUPRC) and energy consumption (Joules/kWh)?
* **RQ2 (Calibration & Post-Hoc Recalibration):** Can near-zero energy post-hoc recalibration (Temperature Scaling, Isotonic Regression) effectively mitigate overconfidence and reduce ECE without degrading predictive discrimination?
* **RQ3 (Cross-Corpus Transfer & Covariate Shift):** How well do source-fitted recalibrators transfer out-of-domain under distribution shift (PsyTAR $\rightarrow$ CADEC zero-shot transfer)?
* **RQ4 (Transferability of Model Selection):** Does the model configuration selected by ECC-MS on the development corpus (PsyTAR) generalize out-of-domain to sustain top accuracy and calibration on an unseen external target (CADEC)?
* **RQ5 (ECC-MS Framework Selection):** Under what inference volume and energy budget constraints ($E$) does the framework transition selection between lightweight classical models and high-capacity transformer models?

---

## 📁 Repository Architecture

```text
├── configs/                                # Experimental hyperparameter configs
├── data/                                   # Datasets (Harmonised & Raw)
│   ├── 01_primary_adr_detection/
│   │   ├── dev_psytar/                     # PsyTAR Development Corpus (6,003 rows)
│   │   └── external_val_cadec/             # CADEC External Validation Corpus (7,823 sentences)
│   └── 02_secondary_sentiment_scaling/
│       ├── dev_uci_drug_review/            # UCI DrugLib dataset (4,107 rows)
│       └── external_val_webmd/             # WebMD dataset (320,093 rows)
├── reports/                                # Generated figures & visual artifacts
│   ├── COMPLETE_PROJECT_REPORT.md          # Comprehensive Empirical Report
│   ├── st4_reliability_diagrams.png        # Calibration Reliability Diagram plot
│   └── st8_regime_map.png                  # ECC-MS Regime Map & Break-Even curve
├── results/                                # Output tables, JSONs, and prediction artifacts
│   ├── colab_transformer_gpu_results.json  # Colab T4 execution JSON
│   ├── gpu_metrics_recomputed.json         # Recomputed CPU metrics JSON
│   └── *_predictions.npz                   # Saved prediction arrays (logits & probs)
├── scripts/                                # Executable Python benchmark scripts
│   ├── metrics_utils.py                    # Shared metrics module (AUROC, AUPRC, Adaptive ECE, Bootstrap CIs)
│   ├── subword_fragmentation_analysis.py   # Tokenizer subword fragmentation rate audit
│   ├── harmonise_st1.py                    # ST1: Primary Data Load & Label Harmonisation (PsyTAR + CADEC)
│   ├── harmonise_secondary_st1b.py         # ST1b: Secondary Task Label Harmonisation + Cutoff Sensitivity
│   ├── energy_sanity_st2.py                # ST2: Energy Tracking Sanity & Repeatability
│   ├── minimal_pipeline_st3.py             # ST3: Minimal End-to-End CPU Pipeline (100x Amortised Energy)
│   ├── calibration_mechanics_st4.py        # ST4: Calibration Mechanics & Paired ΔECE Bootstrap CIs
│   ├── cross_corpus_plumbing_st5.py        # ST5: Cross-Corpus Out-of-Domain Transfer (Frozen Full CADEC Split)
│   ├── budget_and_subgroup_st6_st7.py      # ST6/ST7: Compute Budget Extrapolation & Subgroup Audit (N>=200)
│   ├── eccms_regime_st8.py                 # ST8: Reconciled ECC-MS Regime Sweep & Statistical-Tie Analysis
│   ├── colab_gpu_transformer_primary_adr.py# GPU Transformer fine-tuning & inference pipeline
│   └── recompute_gpu_metrics.py            # CPU-side metric recomputation from .npz artifacts
├── .gitignore                              # Git exclusion rules
├── README.md                               # Project documentation & report
└── requirements.txt                        # Python dependencies
```

---

## ⚡ Unified Platform Power & Energy Table

*Power measurements reported to 3 decimal places to reconcile load, idle, and net energy per 1k inferences.*

| Platform | Model Arm | Idle Power (W) | Load Power (W) | Net Power (W) | Gross Energy / 1k (J) | Net Energy / 1k (J) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CPU (Intel RAPL)** | **Logistic Regression** | 6.734 W | 7.072 W | **0.338 W** | 0.4400 J | **0.0210 J** |
| **CPU (Intel RAPL)** | **LightGBM (GBDT)** | 6.734 W | 9.940 W | **3.206 W** | 0.7412 J | **0.2391 J** |
| **Colab GPU (T4)** | **DistilBERT** | 10.220 W | 63.670 W | **53.450 W** | 25.8100 J | **21.6600 J** |
| **Colab GPU (T4)** | **PubMedBERT** | 10.220 W | 65.810 W | **55.590 W** | 51.5900 J | **43.5700 J** |

---

## 🧪 Comprehensive Empirical Benchmark Results (ST1–ST8)

### 1. Primary Discrimination & Calibration (CPU Classical Arms)

| Model Arm | Recalibration Method | AUROC | AUPRC | F1@t* | t* | F1@0.5 | ECE (Adaptive) | ECE 95% CI | Paired ΔECE vs Uncal (95% CI) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Uncalibrated | 0.8835 | 0.8327 | 0.7515 | 0.33 | 0.6167 | 0.1365 | [0.1130, 0.1762] | Baseline |
| **Logistic Regression** | Temp Scaled | **0.8835** | **0.8327** | **0.7547** | 0.27 | 0.6167 | **0.0815** | [0.0669, 0.1280] | **-0.0550 [-0.0693, -0.0205]*** |
| **Logistic Regression** | Isotonic | 0.8809 | 0.8013 | 0.7500 | 0.34 | 0.6288 | **0.0704** | [0.0467, 0.0996] | **-0.0661 [-0.1019, -0.0379]*** |
| **LightGBM (GBDT)** | Uncalibrated | 0.7942 | 0.6902 | 0.6766 | 0.32 | 0.5837 | 0.0595 | [0.0474, 0.1148] | Baseline |
| **LightGBM (GBDT)** | Temp Scaled | **0.7942** | **0.6902** | **0.6766** | 0.35 | 0.5837 | **0.0543** | [0.0446, 0.1088] | -0.0052 [-0.0183, +0.0113] (n.s.) |
| **LightGBM (GBDT)** | Isotonic | 0.7920 | 0.6630 | 0.6766 | 0.31 | 0.4528 | **0.0548** | [0.0337, 0.0913] | -0.0048 [-0.0512, +0.0182] (n.s.) |

*\*Indicates statistically significant calibration improvement ($p < 0.05$).*

---

### 2. Empirical GPU Transformer Benchmarks (Google Colab T4 FP16)

| Model Arm | Recalibration Method | PsyTAR AUROC | PsyTAR AUPRC | PsyTAR F1@t* | PsyTAR F1@0.5 | PsyTAR ECE (Ada) | PsyTAR ECE 95% CI | CADEC AUROC | CADEC AUPRC | CADEC F1@t* | CADEC ECE (Ada) | Gross J/1k | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT** | Uncalibrated | 0.9059 | 0.8422 | 0.7579 | 0.7733 | 0.0666 | [0.0486, 0.1009] | 0.9258 | 0.8982 | 0.7987 | 0.0927 | 25.81 J | **1,065.8 s/s** |
| **DistilBERT** | Temp Scaled | 0.9059 | 0.8433 | 0.7606 | 0.7733 | 0.0675 | [0.0495, 0.1019] | 0.9258 | 0.8983 | 0.7972 | 0.0941 | 25.81 J | **1,065.8 s/s** |
| **DistilBERT** | Isotonic | 0.8952 | 0.8044 | 0.6939 | 0.6939 | 0.0577 | [0.0363, 0.0928] | 0.9200 | 0.8597 | 0.7655 | 0.0635 | 25.81 J | **1,065.8 s/s** |
| **PubMedBERT** | Uncalibrated | **0.9138** | **0.8530** | **0.7675** | **0.7907** | **0.0442** | [0.0371, 0.0892] | **0.9336** | **0.9112** | **0.8221** | **0.0793** | 51.59 J | 566.8 s/s |
| **PubMedBERT** | Temp Scaled | **0.9138** | **0.8527** | 0.7529 | **0.7907** | 0.0677 | [0.0559, 0.1107] | **0.9336** | **0.9113** | 0.8126 | 0.1154 | 51.59 J | 566.8 s/s |
| **PubMedBERT** | Isotonic | 0.9036 | 0.8162 | 0.6888 | 0.7704 | 0.0717 | [0.0486, 0.1068] | 0.9249 | 0.8724 | 0.7644 | 0.0650 | 51.59 J | 566.8 s/s |

---

### 3. ST6: Full Compute & Energy Budget Extrapolation Table

| Model Tier | Hardware | Train Time (5 seeds) | Inf Time (5 seeds) | Total Time (h) | Total Energy (J) | Total Energy (kWh) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Linear (LR)** | CPU | 1.89 min | 0.01 min | 0.03 h | 60.5 J | 0.0000 kWh | **PASSED** |
| **Classical GBDT (LightGBM)** | CPU | 0.93 min | 0.00 min | 0.02 h | 93.4 J | 0.0000 kWh | **PASSED** |
| **Efficient Transformer (DistilBERT)** | Colab T4 | 0.59 h | 0.05 h | 0.64 h | 64,017.1 J | 0.0178 kWh | **PASSED** |
| **Biomedical Transformer (PubMedBERT)** | Colab T4 | 0.97 h | 0.09 h | 1.07 h | 107,497.2 J | 0.0299 kWh | **PASSED** |

---

### 4. ST8: Detailed ECC-MS Model Selection Table

*Selection outcome over $(\tau, E)$ grid under the Statistical-Tie Rule ($\Delta\text{AUROC} \le 0.02$):*

| $\tau$ (ECE) | $E$ Budget (Gross J/1k) | Selected Model (Argmax Rule) | Selected Model (Statistical-Tie Rule) | Selected AUROC | Gross Energy / 1k (J) | Feasible Arms |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| **0.05** | 60.0 J | PubMedBERT + Uncalibrated | **PubMedBERT + Uncalibrated** | 0.9138 | 51.59 J | 1 |
| **0.07** | 10.0 J | LightGBM + Uncalibrated | **LightGBM + Uncalibrated** | 0.7942 | 0.74 J | 3 |
| **0.07** | 60.0 J | PubMedBERT + Uncalibrated | **DistilBERT + Uncalibrated** | 0.9059 | **25.81 J** | 7 |
| **0.10** | 0.5 J | LR + TempScale | **LR + TempScale** | 0.8835 | **0.44 J** | 2 |
| **0.10** | 10.0 J | LR + TempScale | **LR + TempScale** | 0.8835 | **0.44 J** | 5 |
| **0.10** | 60.0 J | PubMedBERT + Uncalibrated | **DistilBERT + Uncalibrated** | 0.9059 | **25.81 J** | 9 |

---

## 💡 Absolute Energy Scale & Deployment Framing Warning

At $51.59\text{ J/1k}$, screening **1 million sentences/day** on PubMedBERT consumes **14.3 Wh/day** — roughly equivalent to a single smartphone charge.

While energy asymmetry ratios are large (**117×** Gross PubMedBERT vs LR Gross, **2,075×** Net-to-Net), absolute inference energy remains modest at realistic pharmacovigilance volumes. The framework's core contribution lies in **deployment feasibility under constraint** (on-premise clinical edge hardware, procurement limits, and throughput per watt), rather than grandiose environmental claims.

---

## ⚙️ Reproduction & Execution Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Execute CPU benchmark suite (ST1 through ST8 + Subword Analysis)
python scripts/harmonise_st1.py
python scripts/harmonise_secondary_st1b.py
python scripts/subword_fragmentation_analysis.py
python scripts/energy_sanity_st2.py
python scripts/minimal_pipeline_st3.py
python scripts/calibration_mechanics_st4.py
python scripts/cross_corpus_plumbing_st5.py
python scripts/budget_and_subgroup_st6_st7.py
python scripts/eccms_regime_st8.py

# Execute GPU benchmark (Colab T4 FP16) & CPU-side prediction recomputation
python scripts/colab_gpu_transformer_primary_adr.py
python scripts/recompute_gpu_metrics.py
```

---

## 📜 Citation

```bibtex
@article{asif2026green,
  title={Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals},
  author={Asif, Talha and others},
  journal={arXiv preprint arXiv:2608.XXXXX},
  year={2026}
}
```
