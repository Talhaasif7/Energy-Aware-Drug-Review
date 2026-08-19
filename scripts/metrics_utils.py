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
                  ci=0.95, method='adaptive'):
    """
    Compute ECE with bootstrap 95% confidence interval.
    Returns: (ece_point, ece_lower, ece_upper)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_probs = np.asarray(y_probs, dtype=float)
    n = len(y_true)

    ece_fn = compute_ece_adaptive if method == 'adaptive' else compute_ece_equal_width
    ece_point = ece_fn(y_true, y_probs, n_bins)

    rng = np.random.RandomState(42)
    boot_eces = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_eces[b] = ece_fn(y_true[idx], y_probs[idx], n_bins)

    alpha = (1 - ci) / 2
    ece_lower = np.percentile(boot_eces, 100 * alpha)
    ece_upper = np.percentile(boot_eces, 100 * (1 - alpha))

    return ece_point, ece_lower, ece_upper


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

    # Calibration
    ece_adaptive, ece_lo, ece_hi = bootstrap_ece(
        y_true, y_probs, n_bins=n_bins, method='adaptive'
    )
    ece_ew = compute_ece_equal_width(y_true, y_probs, n_bins=n_bins)
    brier = brier_score_loss(y_true, y_probs)
    nll = log_loss(y_true, y_probs, labels=[0, 1])

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

    def fit(self, y_calib, probs_calib):
        eps = 1e-7
        p_clipped = np.clip(probs_calib, eps, 1.0 - eps)
        logits_calib = logit(p_clipped)

        def nll_objective(T_val):
            if T_val <= 0:
                return 1e9
            scaled_logits = logits_calib / T_val
            scaled_p = expit(scaled_logits)
            return log_loss(y_calib, scaled_p, labels=[0, 1])

        res = minimize_scalar(nll_objective, bounds=(0.01, 10.0),
                              method='bounded')
        self.T = float(res.x)
        return self

    def transform(self, probs):
        eps = 1e-7
        p_clipped = np.clip(probs, eps, 1.0 - eps)
        logits = logit(p_clipped)
        scaled_logits = logits / self.T
        return expit(scaled_logits)
