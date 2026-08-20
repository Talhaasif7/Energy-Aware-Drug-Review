# Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![CodeCarbon](https://img.shields.io/badge/Energy%20Tracking-CodeCarbon-green.svg)](https://codecarbon.io)
[![Status](https://img.shields.io/badge/CPU%20Arms-VERIFIED-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/GPU%20Arms-PENDING%20RERUN-orange.svg)]()

This repository contains the complete experimental framework for **"Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals"**.

The project introduces the **ECC-MS (Energy–Calibration Constrained Model Selection)** framework with a **statistical-tie rule**: when two models' AUROC confidence intervals overlap, ECC-MS selects the lowest-energy model that clears the calibration constraint — rather than blindly argmax-ing a fifth-decimal AUROC difference.

---

## 📋 Core Research Questions

* **RQ1 (Predictive-Energy Pareto Front):** How do classical CPU models compare to Transformers in AUROC vs energy trade-offs?
* **RQ2 (Calibration & Post-Hoc Recalibration):** Can near-zero energy recalibration reduce ECE without degrading discrimination?
* **RQ3 (Cross-Corpus Transfer):** How well do recalibrators transfer out-of-domain (PsyTAR → CADEC zero-shot)?
* **RQ4 (Subgroup Fairness):** How does calibration quality vary across drug subgroups ($N \ge 200$)?
* **RQ5 (ECC-MS Framework):** Under what $(\tau, E)$ constraints does the framework transition between model classes?

---

## 📁 Repository Architecture

```text
├── configs/                                # Experimental configs
├── data/
│   ├── 01_primary_adr_detection/
│   │   ├── dev_psytar/                     # PsyTAR (6,003 sentences)
│   │   └── external_val_cadec/             # CADEC (7,823 sentences)
│   └── 02_secondary_sentiment_scaling/
│       ├── dev_uci_drug_review/            # UCI DrugLib (4,107 rows)
│       └── external_val_webmd/             # WebMD (320,093 rows)
├── reports/
│   ├── COMPLETE_PROJECT_REPORT.md
│   ├── st4_reliability_diagrams.png
│   └── st8_regime_map.png
├── results/
│   └── colab_transformer_gpu_results.json
├── scripts/
│   ├── metrics_utils.py                    # Shared metrics (AUROC, ECE, Bootstrap CIs)
│   ├── harmonise_st1.py                    # ST1: Primary data harmonisation
│   ├── harmonise_secondary_st1b.py         # ST1b: Secondary + ordinal cutoff sensitivity
│   ├── subword_fragmentation_analysis.py   # Tokenizer fragmentation audit
│   ├── energy_sanity_st2.py                # ST2: Energy measurement sanity
│   ├── minimal_pipeline_st3.py             # ST3: CPU pipeline (100x amortised)
│   ├── calibration_mechanics_st4.py        # ST4: Calibration + paired Δ ECE
│   ├── cross_corpus_plumbing_st5.py        # ST5: PsyTAR → CADEC transfer
│   ├── budget_and_subgroup_st6_st7.py      # ST6/ST7: Budget + subgroup audit
│   ├── eccms_regime_st8.py                 # ST8: ECC-MS regime sweep + tie rule
│   ├── colab_gpu_transformer_primary_adr.py# GPU fine-tuning pipeline (Colab T4)
│   └── recompute_gpu_metrics.py            # CPU-side metric recomputation from .npz
└── requirements.txt
```

---

## 🔬 What Is Measured vs What Is Pending

> **Transparency note:** This section explicitly declares which results are from live hardware measurements and which are pending.

### ✅ Verified (Live CPU Measurements)

| Metric | Source | Status |
| :--- | :--- | :---: |
| AUROC, AUPRC, F1@t\*, ECE, NLL (LR, GBDT) | ST3/ST4/ST5 (local CPU + CodeCarbon RAPL) | **VERIFIED** |
| Paired Δ ECE bootstrap CIs | ST4 (1000 shared resamples) | **VERIFIED** |
| Temperature T values (LR=0.6251, GBDT=1.2418) | ST4 calibration split | **VERIFIED** |
| Cross-corpus transfer (PsyTAR → CADEC) | ST5 (frozen N=7,823) | **VERIFIED** |
| Subgroup audit (SNRI/SSRI, 4 drugs, all N≥200) | ST7 (PsyTAR Excel) | **VERIFIED** |
| Ordinal cutoff sensitivity (3 variants each) | ST1b | **VERIFIED** |
| Subword fragmentation (33 ADR terms) | CPU tokenizer audit | **VERIFIED** |

### ✅ Measured on GPU (Gating Mode)

| Metric | Source | Status |
| :--- | :--- | :---: |
| F1@0.5 (PsyTAR + CADEC) | Colab T4 gating run | **MEASURED** |
| ECE uniform (PsyTAR + CADEC) | Colab T4 gating run | **MEASURED** |
| NLL (PsyTAR + CADEC) | Colab T4 gating run | **MEASURED** |
| Gross inference energy / 1k | CodeCarbon on Colab T4 | **MEASURED** |
| Inference throughput (sents/sec) | Wall clock on Colab T4 | **MEASURED** |

### ⏳ Pending (Requires Colab Re-Run)

| Metric | Reason | Script Ready |
| :--- | :--- | :---: |
| GPU AUROC, AUPRC | Not computed in gating run | ✅ |
| GPU F1@t\* (threshold-tuned) | Not applied to GPU arms | ✅ |
| GPU ECE bootstrap CIs | Not computed | ✅ |
| nvidia-smi idle/load power per arm | Not traced | ✅ |
| GPU Net energy | Requires idle baseline | ✅ |
| Fitted T values for transformers | Not logged | ✅ |
| Calib-split NLL pre/post for transformers | Not logged | ✅ |
| .npz prediction artifacts | Saved on VM, not downloaded | ✅ |

---

## 🧪 Verified CPU Empirical Results

### ST3: Minimal CPU Pipeline

| Model | AUROC | AUPRC | F1@0.5 | ECE (Adaptive) | ECE 95% CI | Gross J/1k |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8904** | **0.8550** | 0.7040 | 0.1173 | [0.0983, 0.1568] | 0.4400 J |
| **LightGBM (GBDT)** | 0.8295 | 0.7756 | 0.6641 | **0.0477** | [0.0387, 0.0960] | 0.7412 J |

### ST4: Calibration & Paired Δ ECE

| Model | Method | AUROC | F1@t\* | t\* | ECE (Ada) | Paired ΔECE vs Uncal (95% CI) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LR** | Uncalibrated | 0.8835 | 0.7515 | 0.33 | 0.1365 | Baseline |
| **LR** | Temp Scaled | 0.8835 | 0.7547 | 0.27 | 0.0815 | **-0.0550 [-0.0693, -0.0205]\*** |
| **LR** | Isotonic | 0.8809 | 0.7500 | 0.34 | 0.0704 | **-0.0661 [-0.1019, -0.0379]\*** |
| **GBDT** | Uncalibrated | 0.7942 | 0.6766 | 0.32 | 0.0595 | Baseline |
| **GBDT** | Temp Scaled | 0.7942 | 0.6766 | 0.35 | 0.0543 | -0.0052 [-0.0183, +0.0113] (n.s.) |
| **GBDT** | Isotonic | 0.7920 | 0.6766 | 0.31 | 0.0548 | -0.0048 [-0.0512, +0.0182] (n.s.) |

*\*Statistically significant (CI excludes zero).*

### GPU Gating Results (Measured Metrics Only)

| Model | Method | PsyTAR F1@0.5 | PsyTAR ECE | PsyTAR NLL | CADEC F1@0.5 | CADEC ECE | Gross J/1k | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT** | Uncal | 0.7762 | 0.0532 | 0.3567 | 0.7964 | 0.0602 | 25.81 J | 1,065.8 s/s |
| **DistilBERT** | TempScale | 0.7762 | 0.0702 | 0.3613 | 0.7964 | 0.0740 | 25.81 J | 1,065.8 s/s |
| **PubMedBERT** | Uncal | 0.8140 | 0.0349 | 0.3363 | 0.8012 | 0.0367 | 51.59 J | 566.8 s/s |
| **PubMedBERT** | TempScale | 0.8140 | 0.0529 | 0.3460 | 0.8012 | 0.0751 | 51.59 J | 566.8 s/s |

> **Note:** GPU AUROC, AUPRC, F1@t\*, ECE bootstrap CIs, and Net energy are **pending re-run**. The updated GPU script ([colab_gpu_transformer_primary_adr.py](scripts/colab_gpu_transformer_primary_adr.py)) will compute all of these automatically.

---

### Gross Energy Comparison

| Platform | Model | Idle (W) | Load (W) | Net (W) | Gross J/1k |
| :--- | :--- | :---: | :---: | :---: | :---: |
| CPU (RAPL) | LR | 6.73 | 7.07 | 0.34 | **0.4400** |
| CPU (RAPL) | LightGBM | 6.73 | 9.94 | 3.21 | **0.7412** |
| Colab T4 | DistilBERT | TBD | TBD | TBD | **25.81** |
| Colab T4 | PubMedBERT | TBD | TBD | TBD | **51.59** |

> **GPU Net energy requires nvidia-smi idle baseline** (60-second trace). The updated script includes this measurement. GPU idle/load power are **not yet measured per-arm**.

### Absolute Energy Scale Warning

At 51.59 J/1k, PubMedBERT screening **1 million sentences/day** costs **14.3 Wh** — roughly a phone charge. The energy ratios are dramatic but absolute stakes are modest at realistic pharmacovigilance volumes. The contribution is about **deployment feasibility** under hardware/budget constraints, not environmental impact claims.

---

## ⚙️ Reproduction

```bash
pip install -r requirements.txt

# CPU arms (run locally)
python scripts/harmonise_st1.py
python scripts/harmonise_secondary_st1b.py
python scripts/energy_sanity_st2.py
python scripts/minimal_pipeline_st3.py
python scripts/calibration_mechanics_st4.py
python scripts/cross_corpus_plumbing_st5.py
python scripts/budget_and_subgroup_st6_st7.py
python scripts/eccms_regime_st8.py

# GPU arms (run on Google Colab free tier T4)
# Upload repo → run colab_gpu_transformer_primary_adr.py
# Download .npz + .json → then:
python scripts/recompute_gpu_metrics.py
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
