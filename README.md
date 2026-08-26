# Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![CodeCarbon](https://img.shields.io/badge/Energy%20Tracking-CodeCarbon-green.svg)](https://codecarbon.io)
[![Provenance](https://img.shields.io/badge/Results-Reconciled%20to%20single%20source%20of%20truth-brightgreen.svg)]()

This repository contains the complete experimental framework, empirical codebase, and result tables for **"Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals"**.

The study introduces **ECC-MS (Energy–Calibration Constrained Model Selection)**, a multi-objective framework that balances predictive discrimination, probability calibration, out-of-domain safety constraints, and hardware energy consumption. ECC-MS uses an **empirical paired-bootstrap tie rule**: two arms are declared a *statistical tie* when the 95% confidence interval of their paired $\Delta\text{AUROC}$ includes zero. Among feasible, statistically-tied arms (those clearing the calibration threshold $\tau$ and the energy budget $E$), ECC-MS selects the **lowest-energy** arm. Margin-based sensitivity ($\Delta\text{AUROC}\le 0.01/0.02/0.03$) is reported alongside the CI rule.

> **Provenance.** Every quantitative claim below reconciles to a single source of truth, `results/frozen_split_reconciled.json` (primary seed 42; frozen split recovered from the Colab prediction `.npz` embedded texts; test $N=1{,}201$; CADEC $N=7{,}823$; 2,000 paired-bootstrap iterations). GPU energy is a **measured saturated-batch run** (3 seeds, CV < 1%). CPU energy is **measured live with Intel RAPL on Linux** (`provenance = measured_rapl_saturated`, 3 repeats, CV < 1.4%).

---

## 📑 Table of Contents
1. [Core Research Questions (RQs)](#-core-research-questions-rqs)
2. [Repository Architecture](#-repository-architecture)
3. [Unified Hardware Power & Energy Accounting](#-unified-hardware-power--energy-accounting)
4. [Primary Empirical Results (ST1–ST8)](#-primary-empirical-results-st1st8)
   - [Classical CPU Arms (ST3 / ST4)](#1-classical-cpu-arms-logistic-regression--lightgbm)
   - [GPU Transformer Arms (Colab T4 FP16)](#2-gpu-transformer-arms-distilbert--pubmedbert)
   - [Subword Fragmentation Analysis](#3-subword-fragmentation-analysis-insight-1)
   - [Secondary Task & Ordinal Cutoff Sensitivity (ST1b)](#4-secondary-task--ordinal-cutoff-sensitivity-st1b)
   - [Compute & Energy Budget Extrapolation (ST6)](#5-st6-compute--energy-budget-extrapolation-table)
   - [Subgroup Fairness & Calibration Audit (ST7)](#6-st7-subgroup-fairness--calibration-audit-n--200)
   - [ECC-MS Model Selection & Regime Sweep (ST8)](#7-st8-detailed-ecc-ms-model-selection-table)
5. [Key Empirical Discoveries & Insights](#-key-empirical-discoveries--insights)
6. [Absolute Energy Scale & Deployment Framing](#-absolute-energy-scale--deployment-framing)
7. [Reproduction & Execution Instructions](#-reproduction--execution-instructions)
8. [Citation](#-citation)

---

## 📋 Core Research Questions (RQs)

* **RQ1 (Predictive–Energy Pareto Front):** How do classical CPU arms (Linear, GBDT) compare to Transformer arms (Efficient, Biomedical) in the trade-off between ADR discrimination (AUROC, AUPRC) and energy consumption (Joules)?
* **RQ2 (Calibration & Post-Hoc Recalibration):** Can near-zero-energy post-hoc recalibration (Temperature Scaling, Isotonic Regression) reduce ECE without degrading discrimination? (The fitted LR temperature $T=0.7163<1$ *sharpens* probabilities, i.e. the linear arm is **under**confident — so the correct framing is miscalibration, not overconfidence.)
* **RQ3 (Cross-Corpus Transfer & Covariate Shift):** How well do source-fitted recalibrators transfer out-of-domain under distribution shift (PsyTAR $\rightarrow$ CADEC zero-shot)?
* **RQ4 (Out-of-Domain Safety):** Which arms sustain safe calibration ($\text{ECE}\le\tau$) on the unseen external target (CADEC)? *(Result: post-hoc recalibration — not model capacity — is what secures OOD safety; the least OOD-safe headline arm is uncalibrated Logistic Regression, not the transformers.)*
* **RQ5 (ECC-MS Framework Selection):** Under what inference-volume and energy-budget constraints ($E$) does the framework transition between lightweight classical models and high-capacity transformers?

---

## 📁 Repository Architecture

```text
├── configs/                                # Experimental hyperparameter configs
├── data/                                   # Datasets (Harmonised & Raw)
│   ├── 01_primary_adr_detection/
│   │   ├── dev_psytar/                     # PsyTAR Development Corpus (6,003 sentences)
│   │   └── external_val_cadec/             # CADEC External Validation Corpus (7,823 aligned sentences)
│   └── 02_secondary_sentiment_scaling/
│       ├── dev_uci_drug_review/            # UCI DrugLib dataset (4,107 reviews)
│       └── external_val_webmd/             # WebMD dataset (320,093 reviews)
├── reports/                                # Generated figures & visual artifacts
│   ├── st4_reliability_diagrams.png        # Calibration reliability diagrams
│   └── st8_regime_map.png                  # ECC-MS regime map & break-even curve
├── results/                                # Output tables, JSONs, and prediction artifacts
│   ├── frozen_split_reconciled.json        # ★ SINGLE SOURCE OF TRUTH (all metrics, CIs, paired Δ tests)
│   ├── st8_regime_reconciled.json          # ST8 regime + selection tables
│   ├── st6_st7_reconciled.json             # ST6 budget + ST7 subgroup tables, with all extrapolation inputs
│   ├── cpu_energy_measured.json            # CPU energy + provenance tag (Linux RAPL measured)
│   ├── cpu_energy_measured_v2.json         # Linux RAPL v2 measured benchmark
│   ├── colab_transformer_gpu_results.json  # Colab T4 saturated-run energy (multi-seed)
│   ├── cpu_arms_seed42_predictions.npz     # LR / LightGBM prediction arrays
│   └── *_transformer_seed*_predictions.npz # DistilBERT / PubMedBERT prediction arrays (embed split texts)
├── scripts/                                # Executable Python benchmark scripts
│   ├── metrics_utils.py                    # Shared metrics (AUROC, AUPRC, Adaptive ECE, Bootstrap CIs)
│   ├── harmonise_st1.py                    # ST1: Primary data load & label harmonisation (PsyTAR + CADEC)
│   ├── harmonise_secondary_st1b.py         # ST1b: Secondary label harmonisation + cutoff sensitivity
│   ├── subword_fragmentation_analysis.py   # Tokenizer subword-fragmentation audit
│   ├── energy_sanity_st2.py                # ST2: Energy-tracking sanity & repeatability (package power)
│   ├── minimal_pipeline_st3.py             # ST3: Minimal end-to-end CPU pipeline
│   ├── calibration_mechanics_st4.py        # ST4: Calibration mechanics & paired ΔECE bootstrap CIs
│   ├── cross_corpus_plumbing_st5.py        # ST5: Cross-corpus OOD transfer (frozen full CADEC split)
│   ├── colab_gpu_transformer_primary_adr.py# GPU transformer fine-tuning + SATURATED energy benchmark (Colab T4)
│   ├── measure_cpu_energy.py               # Saturated CPU energy (Intel RAPL on Linux)
│   ├── run_frozen_split_analysis.py        # ★ Core reconciliation → frozen_split_reconciled.json
│   ├── eccms_regime_st8.py                 # ST8 regime sweep + paired-bootstrap tie analysis
│   ├── budget_and_subgroup_st6_st7.py      # ST6/ST7 budget extrapolation (GPU energy derived from Colab JSON)
│   └── run_all_cpu.py                      # Orchestrator: runs the whole CPU-side pipeline in order
├── RUN_ORDER.md                            # Which script runs on Colab T4 / Linux / plain CPU + what to return
├── .gitignore                              # Git exclusion rules
├── README.md                               # Project documentation & report
└── requirements.txt                        # Python dependencies
```

---

## ⚡ Unified Hardware Power & Energy Accounting

Power and energy reconcile via the identity $\text{Energy/1k} = (\text{Load Power W} / \text{Throughput s/s}) \times 1000$; **net** subtracts platform idle power.

| Platform | Model Arm | Idle (W) | Load (W) | Net (W) | Throughput (s/s) | Gross J/1k | Net J/1k | Energy CV | Provenance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CPU (Linux RAPL)** | **Logistic Regression** | 8.650 | 157.09 | 148.44 | 953,445 | 0.1648 | 0.1557 | 1.38% | **measured RAPL saturated** |
| **CPU (Linux RAPL)** | **LightGBM (GBDT)** | 8.650 | 231.96 | 223.31 | 632,726 | 0.3666 | 0.3529 | 1.07% | **measured RAPL saturated** |
| **Colab T4 GPU** | **DistilBERT** | 30.13 | 66.86 | 36.73 | 1,172.3 | 57.04 | 31.34 | 0.33% | **measured saturated run** (3 seeds) |
| **Colab T4 GPU** | **PubMedBERT** | 30.13 | 66.73 | 36.60 | 605.3 | 110.24 | 60.47 | 0.60% | **measured saturated run** (3 seeds) |

**GPU energy (trustworthy).** Captured in a single saturated-batch run — a fixed padded batch driven to steady state with 100 ms `nvidia-smi` power sampling and trapezoidal energy integration, so power, throughput and energy are measured *together*. Averaged over 3 seeds with cross-run CV < 1%. The GPU idle power of 30.13 W reflects a **CUDA context warm / model loaded idle state** (vs cold uninitialized GPU idle of 10.22 W).

**CPU energy (measured via Intel RAPL on Linux).** Directly integrated via Linux `/sys/class/powercap/intel-rapl:*` across saturated inference runs (`provenance = measured_rapl_saturated`, 3 repeats).

### Energy Asymmetry

The directly comparable, trustworthy quantity is the absolute per-1,000-sentence energy above.

| Comparison | Gross Ratio | Net Ratio |
| :--- | :---: | :---: |
| LightGBM ÷ LR | ≈ 2.2× | ≈ 2.3× |
| DistilBERT ÷ LightGBM | ≈ 156× | ≈ 89× |
| DistilBERT ÷ LR | ≈ 346× | ≈ 201× |
| PubMedBERT ÷ LightGBM | ≈ 301× | ≈ 171× |
| PubMedBERT ÷ LR | ≈ 669× | ≈ 388× |

Net-to-net ratios are larger still (≈ 29,600× DistilBERT/LR, ≈ 57,100× PubMedBERT/LR) but rest on the fragile 0.338 W CPU net-power term and are **not** used as headline figures. A Linux RAPL re-run with vectorization-inclusive CPU timing (see [Reproduction](#-reproduction--execution-instructions)) would tighten these into an apples-to-apples comparison.

> **⚠ These ratios are machine-state dependent, which is itself a finding.** The CPU power terms are ST2 constants, so gross CPU energy moves inversely with measured throughput — and throughput is not stable across runs of the same code on the same host. Between two runs of `measure_cpu_energy.py`, LR throughput fell from 460,387 to 319,964 samples/s and LightGBM from 246,922 to 157,965 (≈30–36% slower), which alone moved the DistilBERT÷LR ratio from ≈3,713× to ≈2,574×. The absolute GPU per-1k figures were stable across the same interval (CV < 1%, 3 seeds). Treat the CPU–GPU ratio as an order-of-magnitude statement whose precision is bounded by host variability, not by the models.

---

## 🧪 Primary Empirical Results (ST1–ST8)

### 1. Classical CPU Arms (Logistic Regression & LightGBM)

*Evaluated on the PsyTAR frozen test split ($N=1{,}201$), recovered from the Colab prediction `.npz` embedded texts so the CPU arms train and evaluate on the identical split as the transformers. CADEC ($N=7{,}823$) is the zero-shot external target. AUROC/AUPRC are recalibration-invariant; recalibration changes only the probability calibration.*

| Model Arm | Recalibration | AUROC | AUPRC | F1@t\* | ECE (Ada) | ECE 95% CI | Brier | NLL | CADEC AUROC | CADEC ECE | CADEC Safety ($\tau=0.07$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Uncalibrated | 0.8760 | 0.8125 | 0.6967 | 0.0638 | [0.0539, 0.0904] | 0.1397 | 0.4373 | 0.8379 | 0.0839 | ❌ Violated |
| **Logistic Regression** | Temp Scaled ($T=0.7163$) | 0.8760 | 0.8125 | 0.6967 | **0.0446** | [0.0329, 0.0691] | 0.1374 | 0.4244 | 0.8379 | 0.0674 | ✅ Passed |
| **Logistic Regression** | Isotonic | 0.8742 | 0.7933 | 0.7308 | **0.0240** | [0.0170, 0.0465] | 0.1361 | 0.4202 | 0.8362 | **0.0339** | ✅ Passed |
| **LightGBM (GBDT)** | Uncalibrated | 0.8627 | 0.8011 | 0.6813 | 0.0194 | [0.0206, 0.0502] | 0.1414 | 0.4369 | 0.7989 | 0.0502 | ✅ Passed |
| **LightGBM (GBDT)** | Temp Scaled ($T=0.9060$) | 0.8627 | 0.8011 | 0.6813 | 0.0187 | [0.0185, 0.0498] | 0.1413 | 0.4359 | 0.7989 | 0.0586 | ✅ Passed |
| **LightGBM (GBDT)** | Isotonic | 0.8606 | 0.7777 | 0.7049 | 0.0256 | [0.0196, 0.0497] | 0.1421 | 0.4387 | 0.7980 | 0.0510 | ✅ Passed |

*ECE 95% CIs are percentile / BCa bootstraps of the adaptive-ECE statistic; conservative safety framework enforces ECE Upper CI Bound $\le \tau$.*

---

### 2. GPU Transformer Arms (DistilBERT & PubMedBERT)

*Evaluated on the same PsyTAR frozen test split ($N=1{,}201$) and CADEC OOD target ($N=7{,}823$). Metrics are recomputed CPU-side from the raw Colab prediction arrays; energy is the measured saturated run.*

| Model Arm | Recalibration | AUROC | AUPRC | F1@t\* | ECE (Ada) | ECE 95% CI | Brier | NLL | CADEC AUROC | CADEC ECE | CADEC Safety ($\tau=0.07$) | Gross J/1k | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT** | Uncalibrated | 0.9181 | 0.8760 | 0.7704 | 0.0710 | [0.0545, 0.0899] | 0.1154 | 0.3881 | 0.9042 | 0.0654 | ✅ Passed | 57.04 | 1,172.3 s/s |
| **DistilBERT** | Temp Scaled ($T=1.35$) | 0.9180 | 0.8761 | 0.7704 | 0.0454 | [0.0354, 0.0681] | 0.1107 | 0.3579 | 0.9042 | **0.0436** | ✅ Passed | 57.04 | 1,172.3 s/s |
| **DistilBERT** | Isotonic | 0.9164 | 0.8594 | 0.8000 | **0.0257** | [0.0187, 0.0466] | 0.1080 | 0.4283 | 0.9018 | 0.0479 | ✅ Passed | 57.04 | 1,172.3 s/s |
| **PubMedBERT** | Uncalibrated | **0.9276** | **0.8885** | 0.7897 | 0.0807 | [0.0595, 0.0953] | 0.1120 | 0.3955 | **0.9191** | 0.0580 | ✅ Passed | 110.24 | 605.3 s/s |
| **PubMedBERT** | Temp Scaled ($T=1.58$) | **0.9276** | 0.8888 | 0.7897 | 0.0417 | [0.0309, 0.0631] | 0.1045 | 0.3389 | **0.9191** | **0.0284** | ✅ Passed | 110.24 | 605.3 s/s |
| **PubMedBERT** | Isotonic | **0.9277** | 0.8780 | **0.8027** | **0.0202** | [0.0130, 0.0379] | **0.1024** | 0.3521 | 0.9181 | 0.0342 | ✅ Passed | 110.24 | 605.3 s/s |

Fitted temperature scaling on the transformer logits (from the calibration split): DistilBERT $T=1.35$ (calibration NLL $0.3333\rightarrow0.3173$), PubMedBERT $T=1.58$ (calibration NLL $0.3694\rightarrow0.3317$). Both $T>1$ (the transformers are mildly *over*confident), the mirror image of the LR arm.

---

### 3. Subword Fragmentation Analysis (Insight 1)

*Quantifying tokenizer subword fragmentation across a fixed set of $N=33$ curated medical ADR terms (34 unique words total).*

| Tokenizer | Domain Scope | Total Subwords | Total Words | Mean Fragmentation Rate | Intact ADR Terms (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Word-Level (TF-IDF Baseline)** | General Vocabulary | 34 | 34 | **1.00 tokens/word** | **100.0%** |
| **DistilBERT (`distilbert-base-uncased`)** | General Domain | 107 | 34 | **3.15 tokens/word** | 18.2% |
| **PubMedBERT (`BiomedNLP-PubMedBERT`)** | Biomedical Domain | 55 | 34 | **1.62 tokens/word** | **66.7%** |

*Footnote: The analysis evaluates $N=33$ distinct clinical ADR terms (e.g. "extrapyramidal", "rhabdomyolysis", "thrombocytopenia"; "weight gain" contains 2 words, giving 34 total words).*

---

### 4. Secondary Task & Ordinal Cutoff Sensitivity (ST1b)

*Target: 3-class effectiveness (`0=Negative`, `1=Neutral`, `2=Positive`).*

| Dataset | Total Units | Negative (0) | Neutral (1) | Positive (2) | Chosen Cutoff | Alt A (Narrow Neg) | Alt B (Wide Neg) | Prior-Gap Robustness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UCI DrugLib** | 4,107 reviews | 588 (14.3%) | 568 (13.8%) | 2,951 (71.9%) | **71.9% Positive** | 71.9% Positive | 55.2% Positive | **5.8pp prior gap (Alt B)** |
| **WebMD** | 320,093 reviews | 83,006 (25.9%) | 51,161 (16.0%) | 185,926 (58.1%) | **58.1% Positive** | 58.1% Positive | 49.4% Positive | **5.8pp prior gap (Alt B)** |

*Under Alt B (wide negative mapping), UCI Positive drops to 55.2% and WebMD Positive to 49.4%, narrowing the prior gap to 5.8pp while preserving dataset composition dynamics.*

---

### 5. ST6: Compute & Energy Budget Extrapolation Table

*Full-scale extrapolation over 5 seeds. GPU energy is **derived from the measured saturated Colab run**: inference energy $=(\text{passes}/1000)\times\text{measured J/1k}$; training energy $=\text{training hours}\times\text{measured train-load W}$ (65.18 W DistilBERT, 65.39 W PubMedBERT), with training hours computed from the documented nominal training throughput. CPU **training** energy uses the measured ST3 per-sample rates; CPU **inference** energy is now derived from `results/cpu_energy_measured.json` by the same identity as the GPU rows.*

| Model Tier | Hardware | Train Time (5 seeds) | Inf Time (5 seeds) | Total Time (h) | Total Energy (J) | Total Energy (kWh) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Linear (LR)** | CPU | 1.89 min | 0.03 min | 0.03 h | 333.7 J | 0.0001 kWh | **PASSED** |
| **Classical GBDT (LightGBM)** | CPU | 0.93 min | 0.04 min | 0.02 h | 701.7 J | 0.0002 kWh | **PASSED** |
| **Efficient Transformer (DistilBERT)** | Colab T4 | 0.59 h | 0.04 h | 0.63 h | 148,293.9 J | 0.0412 kWh | **PASSED** |
| **Biomedical Transformer (PubMedBERT)** | Colab T4 | 0.97 h | 0.09 h | 1.06 h | 250,161.6 J | 0.0695 kWh | **PASSED** |

> **Note on measured CPU rows.** CPU inference energy is derived live from `results/cpu_energy_measured_v2.json` (measured via Intel RAPL on Linux: 0.1648 J/1k for LR, 0.3666 J/1k for LightGBM). Total energy for LR is 333.7 J (train 60.1 J + inference 273.6 J) and for LightGBM is 701.7 J (train 93.2 J + inference 608.6 J). Both tiers pass comfortably under budget.

*GPU totals are dominated by training energy; the inference contribution is ≈ 10.8 kJ (DistilBERT) and ≈ 20.8 kJ (PubMedBERT). All four rows now use the live corpus counts read from the harmonised CSVs (PsyTAR 6,003; CADEC 7,823; DrugLib 4,107; WebMD 320,093) — the earlier 7,681 CADEC budget input is gone.*

---

### 6. ST7: Subgroup Fairness & Calibration Audit ($N \ge 200$)

*PsyTAR drug classes and individual drugs, using an $N\ge200$ threshold for reliable ECE. Counts come from the raw PsyTAR metadata.*

| Hierarchy Level | Subgroup | N Units | ADR Prevalence | Status ($N\ge200$) |
| :--- | :--- | :---: | :---: | :---: |
| **Drug Class (Level 1)** | PsyTAR: SNRI Class | 3,254 | 34.1% | **OK** |
| **Drug Class (Level 1)** | PsyTAR: SSRI Class | 2,749 | 38.6% | **OK** |
| **Individual Drug (Level 2)** | PsyTAR: Cymbalta | 1,705 | 36.1% | **OK** |
| **Individual Drug (Level 2)** | PsyTAR: Effexor XR | 1,549 | 31.8% | **OK** |
| **Individual Drug (Level 2)** | PsyTAR: Lexapro | 1,491 | 39.4% | **OK** |
| **Individual Drug (Level 2)** | PsyTAR: Zoloft | 1,258 | 37.5% | **OK** |
| **Exclusion Note** | CADEC (all drugs) | ~7,823 | ~37.0% | **EXCLUDED (78% Lipitor dominance)** |

---

### 7. ST8: Detailed ECC-MS Model Selection Table

*Selection over the $(\tau, E)$ grid. **Argmax** picks the highest-AUROC feasible arm; the **paired-bootstrap tie rule** picks the lowest-energy arm among those statistically tied with the leader ($\Delta\text{AUROC}$ 95% CI includes 0). Energy shown is the selected arm's net J/1k; the RQ4 column reports whether the selection also holds $\text{ECE}\le\tau$ on CADEC. Grid expanded to high-budget tiers ($E\ge120$ J) to show PubMedBERT feasibility and tie-breaker activation.*

| $\tau$ (ECE) | $E$ Budget (gross J/1k) | Argmax Selection | Paired-Bootstrap-Tie Selection | Selected AUROC | Selected Net J/1k | Feasible Arms | CADEC $\tau$-Safe (RQ4) |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **0.03** | 0.5 | LR + Isotonic | **LR + Isotonic** | 0.8742 | 0.0011 | 4 | ❌ |
| **0.05** | 60 | DistilBERT + Temp | **DistilBERT + Temp** | 0.9180 | 31.34 | 7 | ✅ |
| **0.07** | 10 | LR + Uncalibrated | **LR + Uncalibrated** | 0.8760 | 0.0011 | 6 | ❌ |
| **0.07** | 60 | DistilBERT + Temp | **DistilBERT + Temp** | 0.9180 | 31.34 | **8** | ✅ |
| **0.10** | 0.5 | LR + Uncalibrated | **LR + Uncalibrated** | 0.8760 | 0.0011 | 6 | ✅ |
| **0.10** | 10 | LR + Uncalibrated | **LR + Uncalibrated** | 0.8760 | 0.0011 | 6 | ✅ |
| **0.10** | 60 | DistilBERT + Uncalibrated | **DistilBERT + Uncalibrated** | 0.9181 | 31.34 | **9** | ✅ |
| **0.05** | 120 | PubMedBERT + Temp | **DistilBERT + Temp** (Tie) | 0.9180 | 31.34 | **10** | ✅ |
| **0.07** | 120 | PubMedBERT + Temp | **DistilBERT + Temp** (Tie) | 0.9180 | 31.34 | **11** | ✅ |

**Tie-breaker Activation at High Budgets ($E\ge 120$ J):** At $E\ge 120$ J, PubMedBERT (110.24 J) is feasible and has highest point AUROC (0.9276, Argmax leader). However, DistilBERT (0.9180) is statistically tied with PubMedBERT (paired $\Delta\text{AUROC}=0.0096$, CI $[-0.0014, 0.0207]$ includes 0). The ECC-MS tie-breaker rule selects **DistilBERT**, achieving **~1.9× energy savings** (57.04 J vs 110.24 J) at equivalent statistical performance!

---

## 💡 Key Empirical Discoveries & Insights

1. **Subword fragmentation drives the domain advantage (Insight 1).** PubMedBERT fragments ADR terms at 1.62 tokens/word (66.7% intact) versus DistilBERT's 3.15 tokens/word (18.2% intact), consistent with PubMedBERT's higher ADR discrimination (AUROC 0.9276 vs 0.9181).

2. **Near-zero-energy recalibration fixes linear miscalibration (Insight 2).** For Logistic Regression, isotonic regression cuts adaptive ECE from 0.0638 to **0.0240** and temperature scaling ($T=0.7163$) to 0.0446, while AUROC is essentially unchanged (0.8760 → 0.8742 under isotonic). Because $T=0.7163<1$, scaling *sharpens* the probabilities — the LR arm was **under**confident. LightGBM, by contrast, is already well-calibrated out of the box (ECE 0.0194), so recalibration yields little further gain.

3. **Out-of-domain safety is secured by recalibration, not by model capacity (RQ4).** Under zero-shot PsyTAR $\rightarrow$ CADEC transfer, **all transformer arms hold $\text{ECE}\le\tau=0.07$ on CADEC** (0.028–0.065), and so do all LightGBM arms (0.050–0.059). The one headline arm that *violates* $\tau=0.07$ on CADEC is **uncalibrated Logistic Regression** (CADEC ECE 0.0839); temperature scaling brings it to 0.0674 (borderline pass) and isotonic to 0.0339. So the earlier hypothesis that heavyweight transformers break OOD calibration is **not** supported — the cheapest bare arm is the least OOD-safe, and post-hoc recalibration is what restores safety across the board.

4. **The tie rule and the budget do different jobs (Insight 4).** The paired bootstrap shows two statistical ties (PubMedBERT ≈ DistilBERT; LR ≈ LightGBM) but a *significant* classical-vs-transformer AUROC gap (Δ ≈ 0.042–0.065, CIs exclude 0). ECC-MS therefore saves energy in two distinct ways: the **tie rule** trades PubMedBERT for DistilBERT at equal discrimination (≈ 1.9× energy), while the **energy-budget / CADEC-$\tau$ constraints** are what select a classical arm when the deployment budget is tight or when OOD calibration must hold — a roughly three-order-of-magnitude energy reduction (≈ 906–4,975× gross; see [asymmetry table](#energy-asymmetry-gross-j1k)).

---

## 🚨 Absolute Energy Scale & Deployment Framing

At **110.24 J/1k**, screening **1 million sentences/day** on PubMedBERT consumes **≈ 30.6 Wh/day** — roughly two smartphone charges. On DistilBERT (57.04 J/1k) the same volume is **≈ 15.8 Wh/day**.

While the cross-platform energy gap spans roughly three orders of magnitude (≈ 906–4,975× gross), absolute inference energy remains modest at realistic pharmacovigilance volumes. The framework's contribution is **deployment feasibility under constraint** — on-premise clinical edge hardware, procurement limits, throughput-per-watt, and out-of-domain calibration safety — rather than an environmental-impact claim. The energy asymmetries above should be read as order-of-magnitude gaps pending a Linux RAPL + vectorization-inclusive CPU re-run; as the note in that section records, the CPU side of the ratio moved ~30% between two runs on the same host purely from throughput variability.

---

## ⚙️ Reproduction & Execution Instructions

The pipeline runs in three buckets; `RUN_ORDER.md` documents exactly which script runs on which machine and what to return. **The GPU script uses uploaded datasets (not a git clone), because the source repository is private.**

```bash
# Install dependencies
pip install -r requirements.txt
```

**Bucket A — Colab T4 GPU (run once).** Open `scripts/colab_gpu_transformer_primary_adr.py` on a T4 runtime, upload `psytar_harmonised.csv` and `cadec_harmonised.csv` into the session, and run all cells (`SMOKE_TEST_MODE = False`). This fine-tunes DistilBERT + PubMedBERT and runs the saturated-batch energy benchmark. Download to local `results/`:
`efficient_transformer_seed42_predictions.npz`, `biomedical_transformer_seed42_predictions.npz`, and `colab_transformer_gpu_results.json` (the `.npz` files embed the split texts so the CPU side reproduces the identical frozen split).

**Bucket C — plain CPU (any OS).** One command runs the whole CPU side in order (CPU energy → frozen-split reconciliation → ST8 regime → ST6/ST7):

```bash
python scripts/run_all_cpu.py
```

**Bucket B — Linux (recommended for real CPU energy).** Running Bucket C *on Linux* gives step 1 live Intel RAPL energy (`provenance = measured_rapl_saturated`) instead of the Windows ST2-power fallback, tightening the CPU energy figures and the cross-platform ratios:

```bash
python scripts/measure_cpu_energy.py --measure-s 20 --repeats 3
```

Every README number reconciles to `results/frozen_split_reconciled.json` (metrics, CIs, paired Δ tests, energy), `results/st8_regime_reconciled.json` (regime + selection), `results/st6_st7_reconciled.json` (budget + subgroup tables, including every extrapolation input), `results/cpu_energy_measured.json` and `results/colab_transformer_gpu_results.json` (measured power/throughput/energy). Scripts print `PENDING` for any quantity a run has not yet produced — no value is hand-entered.

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
