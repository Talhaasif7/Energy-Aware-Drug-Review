# Green and Trustworthy: Energy-Aware NLP for Patient Drug-Review Safety Signals

[![Status](https://img.shields.io/badge/Status-Active_Research-emerald.svg)](#)
[![Framework](https://img.shields.io/badge/Framework-ECC--MS-blue.svg)](#)
[![Focus](https://img.shields.io/badge/Focus-Clinical_NLP_%7C_Green_AI_%7C_Calibration-green.svg)](#)

> **Core Research Question**: When we make patient-review NLP cheaper, what happens to its calibration, and can we restore calibration cheaply enough that a small model remains the right deployment choice?

---

## Executive Summary & Rationale

Patient drug reviews carry critical clinical safety signals (such as Adverse Drug Reactions and drug effectiveness). For privacy, regulatory compliance, and cost efficiency, processing these reviews increasingly occurs on-premise or at the edge in resource-constrained healthcare settings.

While lighter NLP models reduce energy footprints, an efficient model whose probability estimates are poorly calibrated poses a severe risk in clinical environments. A confident false negative (missing an adverse event) or false positive carries serious real-world consequences. This project introduces **ECC-MS (Energy–Calibration Constrained Model Selection)**—a framework that treats **uncertainty calibration as a first-class, measurable property of trustworthy deployment**.

---

## Core Contribution: The ECC-MS Framework

**ECC-MS** selects the optimal model, representation, and recalibration configuration on a development corpus by solving a constrained optimization problem:

$$\max_{\theta \in \Theta} \text{Accuracy}(\theta) \quad \text{subject to} \quad \text{ECE}(\theta) \le \tau \quad \text{and} \quad \text{Energy}(\theta) \le E$$

### Key Methodological Innovations:
1. **Measured Physical Energy**: The feasible set $\Theta$ is populated using **direct hardware measurements** (RAPL for CPU, CodeCarbon for GPU), rather than theoretical FLOP estimates or proxies.
2. **Low-Cost Recalibration as First-Class Option**: Post-hoc recalibration (Temperature Scaling, Isotonic Regression) is evaluated as a near-zero-energy intervention that reshapes the feasible space (allowing a lightweight model + recalibration to enter the optimal region previously held by heavy models).
3. **Validated Cross-Corpus Transfer**: Demonstrates that the configuration selected by ECC-MS on a development corpus remains near-optimal on an **unseen external corpus** under distribution shift.

---

## Research Questions

* **RQ1 (Accuracy–Calibration–Energy Frontier)**: Across an efficient model ladder, how do accuracy, calibration error (ECE), and measured energy trade off on patient-review safety detection?
* **RQ2 (Recalibration per Joule)**: How much of the calibration gap does post-hoc recalibration (temperature scaling / isotonic regression) close, and at what marginal energy cost?
* **RQ3 (Frontier Transferability)**: Do the accuracy-calibration-energy frontier and recalibration effects survive cross-corpus distribution shifts?
* **RQ4 (ECC-MS Optimality)**: Does ECC-MS select a configuration on a development corpus that remains near-optimal on an unseen external validation corpus?

---

## Task Architecture & Dataset Taxonomy

```text
energy-aware-drug-review-nlp/
├── data/
│   ├── 01_primary_adr_detection/                  # Task 1: ADR Presence Detection (Safety Anchor)
│   │   ├── dev_psytar/                            # Development Corpus
│   │   │   └── PsyTAR_dataset.xlsx                # ~6,009 annotated sentences from 891 reviews
│   │   └── external_val_cadec/                    # OOD External-Validation Corpus
│   │       ├── cadec/                             # 1,253 clinical forum posts (MedDRA/SNOMED spans)
│   │       └── metadata/                          # CSIRO license & Dublin Core XML
│   │
│   └── 02_secondary_sentiment_scaling/            # Task 2: Effectiveness / Sentiment at Scale
│       ├── dev_uci_drug_review/                   # Development Corpus (Train/Test splits)
│       │   ├── drugLibTrain_cleaned.csv / .tsv     # 3,076 cleaned reviews
│       │   └── drugLibTest_cleaned.csv / .tsv      # 1,032 cleaned reviews
│       └── external_val_webmd/                    # OOD External-Validation Corpus
│           └── webmd.csv                          # 320,096 web-scale reviews
│
├── scripts/                                       # Data Processing & Refinement Pipeline
│   └── refine_datasets.py                         # Automated cleaning and standardization script
│
└── README.md                                      # Project Architecture & Research Specifications
```

### Partition Rationale:
1. **Primary Task: ADR Presence Detection (Safety Anchor)**
   - **Development**: **PsyTAR** (891 reviews $\rightarrow$ 6,009 sentences with sentence-level annotations).
   - **External Validation**: **CADEC v2** (1,253 posts with span-level annotations).
   - *Clinical Role*: Safety anchor where missing an ADR carries high clinical cost and calibration is vital.
2. **Secondary Task: Effectiveness / Sentiment at Scale (Generalization Dimension)**
   - **Development**: **UCI ML Drug Review** (3,076 train / 1,032 test reviews).
   - **External Validation**: **WebMD Drug Reviews** (320,096 reviews).
   - *Role*: Evaluates scaling behavior across large patient cohorts and diverse medical conditions.

---

## Label Harmonization Protocol (ST1)

To ensure valid cross-corpus transfer between PsyTAR and CADEC, target labels are pre-harmonized to a unified target at a fixed sentence unit: **Binary ADR-Present vs. ADR-Absent**.

* **PsyTAR Harmonization**: A sentence unit is **ADR-Present** ($\mathbf{1}$) iff it carries an `ADR` label.
* **CADEC Harmonization**: A sentence unit is **ADR-Present** ($\mathbf{1}$) iff it contains an `ADR` entity span (after splitting forum posts into sentences).

---

## Efficient Model Ladder

| Tier | Models | Hardware Execution | Primary Characteristics |
| :--- | :--- | :--- | :--- |
| **Classical** | TF-IDF + Logistic Regression, LightGBM, CatBoost | Metered CPU (RAPL) | Ultra-fast inference, low baseline energy, reproducible |
| **Efficient Transformers** | DistilBERT, TinyBERT | Metered GPU (CodeCarbon) | Compact neural architectures, fast sequence representation |
| **Biomedical Transformers** | PubMedBERT, BioClinicalBERT | Metered GPU (CodeCarbon) | Domain-specific pretrained weights, higher capacity |

*Note: Oversized generative LLMs are intentionally excluded to focus strictly on resource-efficient, edge-deployable, well-calibrated NLP.*

---

## Evaluation Battery

* **Accuracy Metrics**: Macro-F1 & ADR-Class F1 (Safety Task), Macro-F1 & MAE (Effectiveness Task); 5 random seeds with 95% bootstrap confidence intervals.
* **Calibration Metrics**: Adaptive Expected Calibration Error (Adaptive ECE), Brier Score & decomposition, reliability diagrams before and after Temperature Scaling and Isotonic Regression.
* **Energy Accounting**: Intel RAPL (sysfs idle-subtracted energy in Joules) for CPU arms; CodeCarbon / `nvidia-smi` for GPU arms. Reported as Joules per 1,000 inferences.
* **Subgroup Analysis**: Per-drug-class (PsyTAR) and per-condition (UCI ML) calibration error to expose subgroup miscalibration.

---

## Gating Smoke Tests (ST1 – ST7)

Before launching full-scale experimental matrix runs, all seven gating checks must pass:

1. **ST1 — Data Load & Label Harmonization**: Count tables, class balance check, and 10 audited harmonized samples per corpus.
2. **ST2 — Energy Measurement Sanity**: 60s idle W reading, RAPL reproducibility on CPU (3x repeats, CV < 5%), CodeCarbon GPU test.
3. **ST3 — Minimal End-to-End Pipeline**: Fast pilot on 2k ADR units (TF-IDF+LR, GBDT, DistilBERT 1 epoch) verifying F1, ECE, Brier, and energy logging.
4. **ST4 — Calibration & Recalibration Mechanics**: Verification that Temperature Scaling and Isotonic Regression run cleanly and reduce ECE.
5. **ST5 — Cross-Corpus Plumbing**: Pilot inference of PsyTAR-trained model on CADEC using harmonized targets.
6. **ST6 — Budget Extrapolation**: Extrapolated wall-clock time and total Joules/kWh across 5 seeds for all planned arms.
7. **ST7 — Subgroup Feasibility**: Sample size audit per drug class and medical condition to ensure statistical power for group ECE.

---

## Key Literature & References

1. **Energy vs. Accuracy in Text NLP**: *Comparing energy consumption and accuracy in text classification inference* (Nature Scientific Reports, 2026; arXiv:2508.14170).
2. **Clinical Calibration**: *CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction* (ACL 2026; arXiv:2604.14651).
3. **Biomedical Text Processing for ADRs**: *A computationally efficient biomedical text processing framework for pharmacovigilance: integrating LoRA and interpretable AI for ADR detection* (PMC12950037, 2025).
4. **PsyTAR Corpus**: Zolnoori et al. (2019), *A corpus of psychiatric drug reviews for pharmacovigilance*.
5. **CADEC Corpus**: Karimi et al. (2015), *CADEC: A corpus of adverse drug events and medical concepts*.
