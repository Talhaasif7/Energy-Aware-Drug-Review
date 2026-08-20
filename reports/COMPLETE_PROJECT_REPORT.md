# Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals
**Comprehensive Experimental Benchmark & Empirical Report**

---

## 1. Executive Summary & Core Research Questions

This document presents the complete empirical benchmark for **"Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals"**.

The study introduces **ECC-MS (Energy–Calibration Constrained Model Selection)**, a multi-objective framework designed to balance:
1. **Predictive Discrimination:** AUROC, AUPRC, threshold-tuned $F1@t^*$.
2. **Probability Calibration:** Adaptive equal-mass ECE with 95% bootstrap CIs, Brier score, NLL.
3. **Energy Consumption:** Joules per 1k inferences, load Wattage, and kWh.

---

### Core Research Questions (RQs)

* **RQ1 (Predictive-Energy Pareto Front):** How do classical CPU model arms (Linear, GBDT) compare to Transformer arms (Efficient, Biomedical) in trade-offs between ADR discrimination (AUROC, AUPRC) and energy consumption (Joules/kWh)?
* **RQ2 (Calibration & Post-Hoc Recalibration):** Can near-zero energy post-hoc recalibration (Temperature Scaling, Isotonic Regression) effectively mitigate overconfidence and reduce ECE without degrading predictive discrimination?
* **RQ3 (Cross-Corpus Transfer & Covariate Shift):** How well do source-fitted recalibrators transfer out-of-domain under distribution shift (PsyTAR $\rightarrow$ CADEC zero-shot transfer)?
* **RQ4 (Transferability of Model Selection):** Does the model configuration selected by ECC-MS on the development corpus (PsyTAR) generalize out-of-domain to sustain top accuracy and calibration on an unseen external target (CADEC)?
* **RQ5 (ECC-MS Framework Selection):** Under what inference volume and energy budget constraints ($E$) does the framework transition selection between lightweight classical models and high-capacity transformer models?

---

## 2. Unified Platform Power & Energy Accounting Table

*Power measurements reported to 3 decimal places to reconcile load, idle, and net energy per 1k inferences.*

| Platform | Model Arm | Idle Power (W) | Load Power (W) | Net Power (W) | Gross Energy / 1k (J) | Net Energy / 1k (J) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CPU (Intel RAPL)** | **Logistic Regression** | 6.734 W | 7.072 W | **0.338 W** | 0.4400 J | **0.0210 J** |
| **CPU (Intel RAPL)** | **LightGBM (GBDT)** | 6.734 W | 9.940 W | **3.206 W** | 0.7412 J | **0.2391 J** |
| **Colab GPU (T4)** | **DistilBERT** | 10.220 W | 63.670 W | **53.450 W** | 25.8100 J | **21.6600 J** |
| **Colab GPU (T4)** | **PubMedBERT** | 10.220 W | 65.810 W | **55.590 W** | 51.5900 J | **43.5700 J** |

---

## 3. Primary ADR Detection Results (PsyTAR & CADEC Zero-Shot)

### Classical CPU Arms (Logistic Regression & LightGBM)

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

### GPU Transformer Arms (DistilBERT & PubMedBERT)

| Model Arm | Recalibration Method | PsyTAR AUROC | PsyTAR AUPRC | PsyTAR F1@t* | PsyTAR F1@0.5 | PsyTAR ECE (Ada) | PsyTAR ECE 95% CI | CADEC AUROC | CADEC AUPRC | CADEC F1@t* | CADEC ECE (Ada) | Gross J/1k | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT** | Uncalibrated | 0.9059 | 0.8422 | 0.7579 | 0.7733 | 0.0666 | [0.0486, 0.1009] | 0.9258 | 0.8982 | 0.7987 | 0.0927 | 25.81 J | **1,065.8 s/s** |
| **DistilBERT** | Temp Scaled | 0.9059 | 0.8433 | 0.7606 | 0.7733 | 0.0675 | [0.0495, 0.1019] | 0.9258 | 0.8983 | 0.7972 | 0.0941 | 25.81 J | **1,065.8 s/s** |
| **DistilBERT** | Isotonic | 0.8952 | 0.8044 | 0.6939 | 0.6939 | 0.0577 | [0.0363, 0.0928] | 0.9200 | 0.8597 | 0.7655 | 0.0635 | 25.81 J | **1,065.8 s/s** |
| **PubMedBERT** | Uncalibrated | **0.9138** | **0.8530** | **0.7675** | **0.7907** | **0.0442** | [0.0371, 0.0892] | **0.9336** | **0.9112** | **0.8221** | **0.0793** | 51.59 J | 566.8 s/s |
| **PubMedBERT** | Temp Scaled | **0.9138** | **0.8527** | 0.7529 | **0.7907** | 0.0677 | [0.0559, 0.1107] | **0.9336** | **0.9113** | 0.8126 | 0.1154 | 51.59 J | 566.8 s/s |
| **PubMedBERT** | Isotonic | 0.9036 | 0.8162 | 0.6888 | 0.7704 | 0.0717 | [0.0486, 0.1068] | 0.9249 | 0.8724 | 0.7644 | 0.0650 | 51.59 J | 566.8 s/s |

---

## 4. ST6: Compute & Energy Budget Extrapolation Table

| Model Tier | Hardware | Train Time (5 seeds) | Inf Time (5 seeds) | Total Time (h) | Total Energy (J) | Total Energy (kWh) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Linear (LR)** | CPU | 1.89 min | 0.01 min | 0.03 h | 60.5 J | 0.0000 kWh | **PASSED** |
| **Classical GBDT (LightGBM)** | CPU | 0.93 min | 0.00 min | 0.02 h | 93.4 J | 0.0000 kWh | **PASSED** |
| **Efficient Transformer (DistilBERT)** | Colab T4 | 0.59 h | 0.05 h | 0.64 h | 64,017.1 J | 0.0178 kWh | **PASSED** |
| **Biomedical Transformer (PubMedBERT)** | Colab T4 | 0.97 h | 0.09 h | 1.07 h | 107,497.2 J | 0.0299 kWh | **PASSED** |

---

## 5. ST8: Detailed ECC-MS Model Selection & Regime Sweep

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

## 6. Subword Fragmentation Analysis (Insight 1 Empirical Proof)

| Tokenizer | Domain Scope | Total Subwords | Total Words | Mean Fragmentation Rate | Intact ADR Terms (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Word-Level (TF-IDF Baseline)** | General Vocabulary | 34 | 34 | **1.00 tokens/word** | **100.0%** |
| **DistilBERT (`distilbert-base-uncased`)** | General Domain | 107 | 34 | **3.15 tokens/word** | 18.2% |
| **PubMedBERT (`BiomedNLP-PubMedBERT`)** | Biomedical Domain | 55 | 34 | **1.62 tokens/word** | **66.7%** |

---

## 7. Secondary Task Dataset Scope & Ordinal Cutoffs

The secondary task sentiment-scaling benchmarks utilize:
- **UCI DrugLib (`drugLibTrain` + `drugLibTest`):** 4,107 reviews, mapped via effectiveness to a 3-class target (`0=Negative`, `1=Neutral`, `2=Positive`).
- **WebMD Corpus:** 320,093 reviews, mapped identically via effectiveness (`1,2=Neg`, `3=Neu`, `4,5=Pos`).

Cutoff sensitivity analysis proves the 13.8pp prior gap between UCI DrugLib (71.9% Pos) and WebMD (58.1% Pos) is robust across boundary definitions and reflects true corpus composition differences.

---

## 8. Absolute Energy Scale & Framing Statement

At $51.59\text{ J/1k}$, screening **1 million sentences/day** on PubMedBERT consumes **14.3 Wh/day** — approximately equivalent to a single smartphone charge.

While energy asymmetry ratios are large (**117×** Gross PubMedBERT vs LR Gross, **2,075×** Net-to-Net), absolute inference energy remains modest at realistic pharmacovigilance volumes. The framework's core contribution lies in **deployment feasibility under constraint** (on-premise clinical edge hardware, procurement limits, and throughput per watt), rather than environmental impact claims.
