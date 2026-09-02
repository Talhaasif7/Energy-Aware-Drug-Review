# CADEC Label Harmonisation & Span-to-Sentence Mapping Sensitivity Audit

## 1. Corpus-Level Mapping Audit

| Metric | Value | Methodological Explanation |
| :--- | :---: | :--- |
| **Total CADEC Posts on Disk** | 1,250 | Total `.txt` files in official CADEC corpus |
| **Empty Placeholder Files (0-byte)** | 2 | `LIPITOR.40.txt`, `VOLTAREN-XR.9.txt` (excluded) |
| **Total Evaluated Non-Empty Posts** | 1,248 | Individual patient forum posts with text content |
| **Total Derived Sentence Units** | 7,823 | Pre-split on newlines + Punkt sentence tokenization |
| **Total Gold Brat ADR Spans** | 7,409 | Character-offset annotations in `.ann` files |
| **Missing Annotation Files** | 0 | 100% complete gold clinical annotations |
| **Rule A (Overlap) Positives** | 2,865 (36.62%) | Primary protocol: sentence positive if $\ge 1$ ADR span overlaps |
| **Rule B (Strict Contained) Positives** | 2,863 (36.60%) | Sensitivity rule: sentence positive iff entire ADR span $\subseteq$ sentence |
| **Difference (Rule A vs Rule B)** | **Only 2 sentences (0.02%)** | Sentence-level ground truth is virtually identical |
| **Boundary-Crossing Span Events** | 10 | Spans crossing sentence boundaries (5 left, 5 right) |
| **Sentences Affected by Crossing** | 6 (0.08%) | **Mapping ambiguity rate is < 0.1% across the entire corpus** |

## 2. Primary Harmonisation Robustness: Sentence-Level Sensitivity (Rule A vs Rule B)

| Model Arm | Rule A (Overlap) AUROC | Rule B (Contained) AUROC | ΔAUROC (B - A) | Rule A ECE | Rule B ECE | Discrimination Ranking |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Uncalibrated)** | 0.8309 | 0.8307 | -0.0002 | 0.0924 | 0.0921 | **Strictly Invariant** |
| **Logistic Regression (TempScaled)** | 0.8309 | 0.8307 | -0.0002 | 0.0859 | 0.0857 | **Strictly Invariant** |
| **Logistic Regression (Isotonic)** | 0.8266 | 0.8264 | -0.0002 | 0.0239 | 0.0237 | **Strictly Invariant** |
| **LightGBM (Uncalibrated)** | 0.7801 | 0.7799 | -0.0002 | 0.0681 | 0.0679 | **Strictly Invariant** |
| **LightGBM (TempScaled)** | 0.7801 | 0.7799 | -0.0002 | 0.0650 | 0.0648 | **Strictly Invariant** |
| **LightGBM (Isotonic)** | 0.7775 | 0.7773 | -0.0002 | 0.0563 | 0.0563 | **Strictly Invariant** |
| **DistilBERT (Uncalibrated)** | 0.9170 | 0.9170 | +0.0000 | 0.0559 | 0.0556 | **Strictly Invariant** |
| **DistilBERT (TempScaled)** | 0.9170 | 0.9170 | +0.0000 | 0.0391 | 0.0389 | **Strictly Invariant** |
| **DistilBERT (Isotonic)** | 0.9153 | 0.9153 | +0.0001 | 0.0230 | 0.0230 | **Strictly Invariant** |
| **PubMedBERT (Uncalibrated)** | 0.9258 | 0.9258 | +0.0000 | 0.0606 | 0.0606 | **Strictly Invariant** |
| **PubMedBERT (TempScaled)** | 0.9258 | 0.9258 | +0.0000 | 0.0477 | 0.0477 | **Strictly Invariant** |
| **PubMedBERT (Isotonic)** | 0.9247 | 0.9247 | +0.0000 | 0.0265 | 0.0267 | **Strictly Invariant** |

## 3. Complementary Post-Level Validation (Rule C Max-Pooling Aggregation)

> **Note on Unit of Analysis:** Rule C changes the unit of analysis from individual sentences to entire patient posts ($N=1{,}248$, empirical ADR post prevalence $=88.70\%$) using max-pooling probability aggregation. It is interpreted as a complementary post-level clinical triage validation rather than a direct alternative sentence-labeling rule.

| Model Arm | Post-Level AUROC | Post-Level AUPRC | Post-Level ECE | Post-Level Brier | Transformer Dominance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Uncalibrated)** | 0.8115 | 0.9673 | 0.3107 | 0.1844 | **Preserved** |
| **Logistic Regression (TempScaled)** | 0.8115 | 0.9673 | 0.2928 | 0.1793 | **Preserved** |
| **Logistic Regression (Isotonic)** | 0.8111 | 0.9613 | 0.1691 | 0.1178 | **Preserved** |
| **LightGBM (Uncalibrated)** | 0.8280 | 0.9674 | 0.2802 | 0.1736 | **Preserved** |
| **LightGBM (TempScaled)** | 0.8280 | 0.9674 | 0.2819 | 0.1739 | **Preserved** |
| **LightGBM (Isotonic)** | 0.8271 | 0.9643 | 0.2113 | 0.1364 | **Preserved** |
| **DistilBERT (Uncalibrated)** | 0.9422 | 0.9911 | 0.0516 | 0.0557 | **Preserved** |
| **DistilBERT (TempScaled)** | 0.9423 | 0.9911 | 0.0903 | 0.0609 | **Preserved** |
| **DistilBERT (Isotonic)** | 0.9426 | 0.9896 | 0.0813 | 0.0610 | **Preserved** |
| **PubMedBERT (Uncalibrated)** | 0.9589 | 0.9937 | 0.0213 | 0.0445 | **Preserved** |
| **PubMedBERT (TempScaled)** | 0.9591 | 0.9938 | 0.0514 | 0.0467 | **Preserved** |
| **PubMedBERT (Isotonic)** | 0.9545 | 0.9911 | 0.0541 | 0.0484 | **Preserved** |

## 4. Formal Methodological Defense for Manuscripts

> **Standard Text for Peer-Review Defense:**
> "CADEC sentence-level labels were derived from gold Brat character-level ADR spans using a deterministic sentence-span overlap rule (Rule A). To ensure findings are not artifacts of this transformation, we performed sensitivity analysis under strict span containment (Rule B), where a sentence is labeled positive only if the entire ADR entity span falls within its character boundaries. Across all 7,823 derived sentence units, boundary-crossing entity spans occurred in only 6 sentences (0.08%), altering the positive class count by just 2 sentences (2,865 vs 2,863; 36.62% vs 36.60%). Model discrimination hierarchies ($\text{PubMedBERT} > \text{DistilBERT} > \text{Logistic Regression} > \text{LightGBM}$) and calibration dynamics are strictly identical (maximum $\Delta\text{AUROC} = \pm 0.0002$, maximum $\Delta\text{ECE} = \pm 0.0003$).
>
> In a complementary post-level validation (Rule C), sentence probabilities were aggregated to entire patient forum posts ($N=1{,}248$, 88.7% ADR prevalence) via max-pooling. Transformer superiority remained decisive (PubMedBERT AUROC 0.9589 vs DistilBERT 0.9422 vs Classical $\le 0.8280$), confirming that the observed cross-corpus transfer dynamics reflect genuine model representations rather than sentence-tokenization conventions."
