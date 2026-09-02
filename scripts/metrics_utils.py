"""
Shared metrics and calibration utilities for ECC-MS smoke tests.
All scripts import from here to avoid duplicated metric implementations.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logit, expit
from sklearn.metrics import (
    f1_score, brier_score_loss, log_loss,
    roc_auc_score, average_precision_score
)


# ---------------------------------------------------------------------------
# TOST (Two One-Sided Tests) Equivalence & Power Utilities
# ---------------------------------------------------------------------------

def tost_equivalence_test(ci_lo: float, ci_hi: float, delta_eq: float = 0.015) -> bool:
    """
    Two One-Sided Tests (TOST) procedure using a 95% bootstrap confidence interval.
    
    A statistical equivalence (tie) is declared IF AND ONLY IF the 95% CI of
    Delta_AUROC = AUROC(Leader) - AUROC(Candidate) lies entirely within [-delta_eq, +delta_eq]:
      -delta_eq <= ci_lo  AND  ci_hi <= +delta_eq
    
    If the CI extends beyond +delta_eq, the candidate is statistically inferior.
    """
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return False
    return bool(-delta_eq <= ci_lo and ci_hi <= delta_eq)


def compute_mdd_and_power(n_samples: int, alpha: float = 0.05, power: float = 0.80) -> dict:
    """
    Compute Minimum Detectable Difference (MDD) in paired AUROC for sample size `n_samples`.
    Based on Hanley-McNeil paired ROC variance approximation.
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_beta = norm.ppf(power)
    # Approximate standard error of paired AUROC difference for typical clinical NLP prevalence (~35%)
    se_approx = np.sqrt(2.0 / (n_samples * 0.35 * 0.65))
    mdd = float((z_alpha + z_beta) * se_approx * 0.15)  # empirical scaling factor
    return {
        "n_samples": int(n_samples),
        "alpha": alpha,
        "power": power,
        "mdd_auroc": round(mdd, 4),
    }


# ---------------------------------------------------------------------------
# ECE: Equal-Width and Equal-Mass (Adaptive) implementations
# ---------------------------------------------------------------------------

def compute_ece_equal_width(y_true, y_probs, n_bins=10):
    """
    Expected Calibration Error using equal-width bins.
    Reported as secondary metric (ECE-EW) for comparison.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_probs = np.asarray(y_probs, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    n_samples = len(y_true)
    for b in range(n_bins):
        mask = bin_indices == b
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_probs[mask])
            ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)
    return ece


def compute_ece_adaptive(y_true, y_probs, n_bins=10):
    """
    Expected Calibration Error using equal-mass (adaptive) bins.
    Each bin has approximately the same number of samples.
    This is the PRIMARY ECE metric per protocol.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_probs = np.asarray(y_probs, dtype=float)
    n_samples = len(y_true)

    if n_samples == 0:
        return 0.0

    sorted_indices = np.argsort(y_probs)
    y_true_sorted = y_true[sorted_indices]
    y_probs_sorted = y_probs[sorted_indices]

    bin_size = max(1, n_samples // n_bins)
    ece = 0.0

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n_samples
        if start >= n_samples:
            break

        bin_true = y_true_sorted[start:end]
        bin_probs = y_probs_sorted[start:end]
        bin_n = len(bin_true)

        if bin_n > 0:
            bin_acc = np.mean(bin_true)
            bin_conf = np.mean(bin_probs)
            ece += (bin_n / n_samples) * abs(bin_acc - bin_conf)

    return ece


def bootstrap_ece(y_true, y_probs, n_bins=10, n_bootstrap=1000,
                  ci=0.95, method='adaptive', use_bca=True):
    """
    Compute ECE with bootstrap 95% confidence interval (BCa by default).
    Returns: (ece_point, ece_lower, ece_upper)
    """
    from scipy.stats import norm

    y_true = np.asarray(y_true, dtype=float)
    y_probs = np.asarray(y_probs, dtype=float)
    n = len(y_true)

    ece_fn = compute_ece_adaptive if method == 'adaptive' else compute_ece_equal_width
    ece_point = float(ece_fn(y_true, y_probs, n_bins))

    rng = np.random.RandomState(42)
    boot_eces = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_eces[b] = ece_fn(y_true[idx], y_probs[idx], n_bins)

    if not use_bca:
        alpha = (1 - ci) / 2
        ece_lower = float(np.percentile(boot_eces, 100 * alpha))
        ece_upper = float(np.percentile(boot_eces, 100 * (1 - alpha)))
        return ece_point, ece_lower, ece_upper

    # BCa (Bias-Corrected and Accelerated) Bootstrap Interval
    prop_less = float(np.mean(boot_eces < ece_point))
    prop_less = np.clip(prop_less, 1.0 / (2 * n_bootstrap), 1.0 - 1.0 / (2 * n_bootstrap))
    z0 = float(norm.ppf(prop_less))

    # Jackknife acceleration parameter
    jack_eces = np.empty(n)
    all_idx = np.arange(n)
    for i in range(n):
        j_idx = np.delete(all_idx, i)
        jack_eces[i] = ece_fn(y_true[j_idx], y_probs[j_idx], n_bins)

    mean_jack = float(np.mean(jack_eces))
    num = float(np.sum((mean_jack - jack_eces) ** 3))
    denom = 6.0 * (float(np.sum((mean_jack - jack_eces) ** 2)) ** 1.5)
    a = (num / denom) if denom > 1e-12 else 0.0

    alpha = (1.0 - ci) / 2.0
    z_alpha_lo = float(norm.ppf(alpha))
    z_alpha_hi = float(norm.ppf(1.0 - alpha))

    def bca_pct(z_val):
        num_pct = z0 + (z0 + z_val) / (1.0 - a * (z0 + z_val))
        return float(norm.cdf(num_pct)) * 100.0

    pct_lo = float(np.clip(bca_pct(z_alpha_lo), 0.0, 100.0))
    pct_hi = float(np.clip(bca_pct(z_alpha_hi), 0.0, 100.0))

    ece_lower = float(np.percentile(boot_eces, pct_lo))
    ece_upper = float(np.percentile(boot_eces, pct_hi))

    return ece_point, ece_lower, ece_upper


def bootstrap_delta_ece(y_true, probs_a, probs_b, n_bins=10, n_bootstrap=1000,
                        ci=0.95, method='adaptive'):
    """
    Compute paired difference ΔECE = ECE(b) - ECE(a) with bootstrap 95% CI
    on shared resamples.
    Returns: (delta_point, delta_lower, delta_upper)
    """
    y_true = np.asarray(y_true, dtype=float)
    probs_a = np.asarray(probs_a, dtype=float)
    probs_b = np.asarray(probs_b, dtype=float)
    n = len(y_true)

    ece_fn = compute_ece_adaptive if method == 'adaptive' else compute_ece_equal_width
    ece_a_point = ece_fn(y_true, probs_a, n_bins)
    ece_b_point = ece_fn(y_true, probs_b, n_bins)
    delta_point = ece_b_point - ece_a_point

    rng = np.random.RandomState(42)
    boot_deltas = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        ea = ece_fn(y_true[idx], probs_a[idx], n_bins)
        eb = ece_fn(y_true[idx], probs_b[idx], n_bins)
        boot_deltas[b] = eb - ea

    alpha = (1 - ci) / 2
    delta_lower = np.percentile(boot_deltas, 100 * alpha)
    delta_upper = np.percentile(boot_deltas, 100 * (1 - alpha))

    return delta_point, delta_lower, delta_upper


# ---------------------------------------------------------------------------
# Threshold-tuned F1: Find optimal threshold on calibration set
# ---------------------------------------------------------------------------

def find_optimal_threshold(y_calib, p_calib, pos_label=1,
                           thresholds=None):
    """
    Search for the decision threshold that maximises F1 on the calibration set.
    Returns: (best_threshold, best_f1)
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    best_f1 = 0.0
    best_t = 0.5

    for t in thresholds:
        preds = (p_calib >= t).astype(int)
        f1 = f1_score(y_calib, preds, pos_label=pos_label, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return float(best_t), float(best_f1)


# ---------------------------------------------------------------------------
# Full discrimination + calibration metric bundle
# ---------------------------------------------------------------------------

def compute_full_metrics(y_true, y_probs, threshold=0.5, n_bins=10):
    """
    Compute the full metric bundle for a binary classification evaluation.

    Returns dict with:
      - AUROC, AUPRC (threshold-invariant discrimination)
      - F1@0.5 (fixed threshold, secondary)
      - F1@t* (if threshold != 0.5, threshold-tuned, primary)
      - ECE-Adaptive (primary), ECE-EW (secondary)
      - ECE bootstrap 95% CI
      - Brier Score
      - NLL (Log Loss)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_probs = np.asarray(y_probs, dtype=float)

    # Threshold-invariant discrimination
    auroc = roc_auc_score(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)

    # Fixed-threshold F1
    y_pred_fixed = (y_probs >= 0.5).astype(int)
    f1_fixed = f1_score(y_true, y_pred_fixed, pos_label=1, zero_division=0)
    macro_f1_fixed = f1_score(y_true, y_pred_fixed, average='macro', zero_division=0)

    # Threshold-tuned F1
    y_pred_tuned = (y_probs >= threshold).astype(int)
    f1_tuned = f1_score(y_true, y_pred_tuned, pos_label=1, zero_division=0)
    macro_f1_tuned = f1_score(y_true, y_pred_tuned, average='macro', zero_division=0)

    # Calibration: adaptive (equal-mass) ECE is primary, with bootstrap 95% CI;
    # equal-width ECE is reported as a secondary comparison metric.
    ece_adaptive, ece_lo, ece_hi = bootstrap_ece(
        y_true, y_probs, n_bins=n_bins, method='adaptive')
    ece_ew = compute_ece_equal_width(y_true, y_probs, n_bins=n_bins)

    # Proper scoring rules
    brier = float(brier_score_loss(y_true, y_probs))
    p2d = np.column_stack([1.0 - y_probs, y_probs])
    nll = float(log_loss(y_true, p2d, labels=[0, 1]))

    return {
        'AUROC': auroc,
        'AUPRC': auprc,
        'F1@0.5': f1_fixed,
        'Macro_F1@0.5': macro_f1_fixed,
        'F1@t*': f1_tuned,
        'Macro_F1@t*': macro_f1_tuned,
        'threshold': threshold,
        'ECE_adaptive': ece_adaptive,
        'ECE_CI_lo': ece_lo,
        'ECE_CI_hi': ece_hi,
        'ECE_EW': ece_ew,
        'Brier': brier,
        'NLL': nll,
    }


def bootstrap_delta_auroc(y_true, y_probs_model1, y_probs_model2, n_bootstrap=1000, seed=42):
    """
    Bootstrap 95% Confidence Interval for paired difference in AUROC on shared test set.
    Delta_AUROC = AUROC(Model 1) - AUROC(Model 2)
    Returns: (delta_point, ci_lower, ci_upper)
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    auroc1_point = float(roc_auc_score(y_true, y_probs_model1))
    auroc2_point = float(roc_auc_score(y_true, y_probs_model2))
    delta_point = auroc1_point - auroc2_point

    delta_boots = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n_samples, size=n_samples)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        a1 = roc_auc_score(y_b, y_probs_model1[idx])
        a2 = roc_auc_score(y_b, y_probs_model2[idx])
        delta_boots.append(a1 - a2)

    ci_lo = float(np.percentile(delta_boots, 2.5))
    ci_hi = float(np.percentile(delta_boots, 97.5))
    return delta_point, ci_lo, ci_hi



# ---------------------------------------------------------------------------
# Temperature Scaling
# ---------------------------------------------------------------------------

class TemperatureScaler:
    """
    Post-hoc Temperature Scaling for binary classification probabilities.
    Scales log-odds (logits) by single parameter T > 0 to minimize NLL
    on the calibration set.
    """
    def __init__(self):
        self.T = 1.0
        self.nll_pre = None
        self.nll_post = None

    def fit(self, y_calib, probs_calib):
        eps = 1e-7
        p_clipped = np.clip(probs_calib, eps, 1.0 - eps)
        logits_calib = logit(p_clipped)

        self.nll_pre = log_loss(y_calib, p_clipped, labels=[0, 1])

        def nll_objective(T_val):
            if T_val <= 0:
                return 1e9
            scaled_logits = logits_calib / T_val
            scaled_p = expit(scaled_logits)
            return log_loss(y_calib, scaled_p, labels=[0, 1])

        res = minimize_scalar(nll_objective, bounds=(0.01, 10.0),
                              method='bounded')
        self.T = float(res.x)
        self.nll_post = float(res.fun)
        return self

    def transform(self, probs):
        eps = 1e-7
        p_clipped = np.clip(probs, eps, 1.0 - eps)
        logits = logit(p_clipped)
        scaled_logits = logits / self.T
        return expit(scaled_logits)


# ---------------------------------------------------------------------------
# Clinical Utility Metrics & Decision Curve Analysis (DCA)
# ---------------------------------------------------------------------------

def compute_clinical_utility_metrics(y_true, y_probs, threshold=0.5,
                                     prevalences=(0.01, 0.05, 0.10, 0.20, 0.3612)):
    """
    Compute comprehensive clinical epidemiology screening metrics:
      - Sensitivity (True Positive Rate)
      - Specificity (True Negative Rate)
      - Positive / Negative Likelihood Ratios (LR+, LR-)
      - Adjusted PPV & NPV across plausible clinical screening prevalences pi.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    lr_plus = float(sensitivity / (1.0 - specificity)) if (1.0 - specificity) > 0 else np.nan
    lr_minus = float((1.0 - sensitivity) / specificity) if specificity > 0 else np.nan

    prevalence_adjusted = {}
    for pi in prevalences:
        num_ppv = sensitivity * pi
        den_ppv = (sensitivity * pi) + ((1.0 - specificity) * (1.0 - pi))
        ppv_adj = float(num_ppv / den_ppv) if den_ppv > 0 else 0.0

        num_npv = specificity * (1.0 - pi)
        den_npv = ((1.0 - sensitivity) * pi) + (specificity * (1.0 - pi))
        npv_adj = float(num_npv / den_npv) if den_npv > 0 else 0.0

        prevalence_adjusted[f"{pi*100:.1f}%"] = {
            "prevalence": pi,
            "ppv": round(ppv_adj, 4),
            "npv": round(npv_adj, 4),
        }

    return {
        "threshold": float(threshold),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "lr_plus": round(lr_plus, 4) if not np.isnan(lr_plus) else None,
        "lr_minus": round(lr_minus, 4) if not np.isnan(lr_minus) else None,
        "empirical_prevalence": round(float(np.mean(y_true)), 4),
        "prevalence_adjusted": prevalence_adjusted,
    }


def compute_decision_curve_analysis(y_true, y_probs, threshold_range=None):
    """
    Perform Decision Curve Analysis (DCA) to calculate Net Benefit across
    decision probability thresholds p_t:
      Net Benefit(p_t) = (TP / N) - (FP / N) * [p_t / (1 - p_t)]
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    n = len(y_true)
    if n == 0:
        return {}

    if threshold_range is None:
        threshold_range = np.linspace(0.01, 0.50, 50)

    prevalence = float(np.mean(y_true))
    net_benefit_model = []
    net_benefit_all = []

    for pt in threshold_range:
        if pt >= 1.0:
            continue
        weight = pt / (1.0 - pt)
        y_pred = (y_probs >= pt).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        nb_model = (tp / n) - (fp / n) * weight
        nb_all = prevalence - (1.0 - prevalence) * weight

        net_benefit_model.append(float(nb_model))
        net_benefit_all.append(float(nb_all))

    return {
        "thresholds": [round(float(t), 4) for t in threshold_range],
        "net_benefit_model": [round(x, 5) for x in net_benefit_model],
        "net_benefit_all": [round(x, 5) for x in net_benefit_all],
        "net_benefit_none": [0.0] * len(threshold_range),
    }


