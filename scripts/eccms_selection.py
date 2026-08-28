"""
ECC-MS selection logic (shared by the CPU runner and ST8).

This module centralises the Energy-Calibration Constrained Model Selection
rules so the runner and the regime sweep cannot drift apart.

Key design point (Round 5 fix)
------------------------------
The statistical-tie rule is a REAL paired bootstrap of the AUROC difference on
*shared* resamples of a single frozen test set, NOT a hardcoded margin.

  - Among the feasible arms (ECE <= tau AND energy <= E), find the arm with the
    highest point AUROC ("leader").
  - For every other feasible arm, bootstrap Delta_AUROC = AUROC(leader) - AUROC(arm)
    on the SAME resampled indices for both arms. If the 95% CI of Delta_AUROC
    includes 0, the arm is *statistically tied* with the leader.
  - Among {leader} U {tied arms}, select the LOWEST-energy arm.

AUROC is invariant under monotone recalibration (temperature / isotonic), so the
tie test is computed at *model* granularity (the base model's test probabilities)
and then applied to every recalibration variant of that model.

A pre-registered practical-equivalence margin (fixed Delta_AUROC threshold) is
also provided for a sensitivity strip at 0.01 / 0.02 / 0.03, reported alongside
the bootstrap rule rather than as a substitute for it.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Energy / AUROC accessors
# ---------------------------------------------------------------------------

def get_energy(config, use_gross=False):
    """Energy per 1k inferences for a config, preferring net unless use_gross."""
    if use_gross:
        return config.get('inf_j_gross', float('inf'))
    net = config.get('inf_j_net')
    if net is not None:
        return net
    return config.get('inf_j_gross', float('inf'))


def get_auroc(config):
    return float(config.get('auroc', 0.0))


# ---------------------------------------------------------------------------
# Paired bootstrap of Delta_AUROC on shared resamples
# ---------------------------------------------------------------------------

_BOOTSTRAP_CACHE = {}

def _fast_auroc(y_true, p):
    desc_indices = np.argsort(-p)
    y_sorted = y_true[desc_indices]
    n_pos = np.sum(y_sorted)
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Rank calculation for non-tied/lightly-tied scores
    ranks = np.arange(len(y_sorted), 0, -1)
    pos_rank_sum = np.sum(ranks[y_sorted == 1])
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def paired_delta_auroc(y_true, p_leader, p_other, n_bootstrap=2000, seed=42):
    """
    Paired bootstrap of Delta_AUROC = AUROC(leader) - AUROC(other) on a single
    shared test set. Both models are scored on the SAME resampled indices.

    Returns (delta_point, ci_lo, ci_hi).
    """
    y_true = np.asarray(y_true, dtype=float)
    p_leader = np.ascontiguousarray(p_leader, dtype=np.float64)
    p_other = np.ascontiguousarray(p_other, dtype=np.float64)

    cache_key = (len(y_true), p_leader.tobytes(), p_other.tobytes(), n_bootstrap, seed)
    if cache_key in _BOOTSTRAP_CACHE:
        return _BOOTSTRAP_CACHE[cache_key]

    delta_point = float(roc_auc_score(y_true, p_leader) - roc_auc_score(y_true, p_other))

    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas = np.empty(n_bootstrap)
    valid_count = 0
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yb = y_true[idx]
        if np.sum(yb) == 0 or np.sum(yb) == n:
            continue
        a_lead = _fast_auroc(yb, p_leader[idx])
        a_oth = _fast_auroc(yb, p_other[idx])
        deltas[valid_count] = a_lead - a_oth
        valid_count += 1

    if valid_count == 0:
        res = (delta_point, float('nan'), float('nan'))
    else:
        valid_deltas = deltas[:valid_count]
        ci_lo = float(np.percentile(valid_deltas, 2.5))
        ci_hi = float(np.percentile(valid_deltas, 97.5))
        res = (delta_point, ci_lo, ci_hi)

    _BOOTSTRAP_CACHE[cache_key] = res
    return res


def pairwise_delta_auroc_matrix(y_true, model_probs, n_bootstrap=2000, seed=42):
    """
    Full pairwise Delta_AUROC table for a dict {model_name: p1_test}.
    Delta is AUROC(row) - AUROC(col). Returns a list of dict rows suitable for a
    DataFrame / JSON dump.
    """
    names = list(model_probs.keys())
    rows = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            d, lo, hi = paired_delta_auroc(
                y_true, model_probs[a], model_probs[b],
                n_bootstrap=n_bootstrap, seed=seed)
            tie = (lo <= 0.0 <= hi)
            rows.append({
                'model_a': a, 'model_b': b,
                'delta_auroc': d, 'ci_lo': lo, 'ci_hi': hi,
                'statistical_tie': bool(tie),
            })
    return rows


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------

def feasible_arms(configs, tau, E_budget_per_1k, use_gross=False, use_ece_ci=True):
    """Arms clearing BOTH the calibration threshold (conservative ECE_upper <= tau if use_ece_ci else ECE <= tau)
    and the energy budget (energy <= E). Returns the filtered list (order preserved)."""
    res = []
    for c in configs:
        ece_val = c.get('ece_ci_hi', c.get('ece_upper', c['ece'])) if use_ece_ci else c['ece']
        if ece_val <= tau + 1e-12 and get_energy(c, use_gross) <= E_budget_per_1k + 1e-12:
            res.append(c)
    return res


# ---------------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------------

def eccms_select_argmax(configs, tau, E_budget_per_1k, use_gross=False, use_ece_ci=True):
    """Baseline: among feasible arms, pick the max-AUROC arm."""
    feas = feasible_arms(configs, tau, E_budget_per_1k, use_gross, use_ece_ci=use_ece_ci)
    if not feas:
        return None, 0
    return max(feas, key=get_auroc), len(feas)


def eccms_select_fixed_margin(configs, tau, E_budget_per_1k, margin=0.02,
                              use_gross=False, use_ece_ci=True):
    """Pre-registered practical-equivalence margin: tie if leader_auroc - auroc
    <= margin; among tied arms pick lowest energy."""
    feas = feasible_arms(configs, tau, E_budget_per_1k, use_gross, use_ece_ci=use_ece_ci)
    if not feas:
        return None, 0
    leader = max(get_auroc(c) for c in feas)
    tied = [c for c in feas if (leader - get_auroc(c)) <= margin + 1e-12]
    best = min(tied, key=lambda c: get_energy(c, use_gross))
    return best, len(feas)


def eccms_select_bootstrap_tie(configs, tau, E_budget_per_1k, y_true,
                                model_probs, n_bootstrap=2000, seed=42,
                                use_gross=False, use_ece_ci=True,
                                tost_delta_eq=None,
                                y_ood=None, model_probs_ood=None):
    """
    Primary ECC-MS rule with empirical paired-bootstrap tie test, optional TOST
    equivalence testing (tost_delta_eq, e.g. 0.015), and optional OOD Tie-Test Gate.

    configs         : list of arm dicts; each must have keys
                      name, model, ece, auroc, inf_j_net/inf_j_gross.
    model_probs     : {model_name: p1_test} on the shared frozen test set.
    tost_delta_eq   : float or None. If set, applies TOST equivalence testing
                      (-tost_delta_eq <= ci_lo and ci_hi <= +tost_delta_eq).
    y_ood / model_probs_ood : optional OOD test labels & probabilities (e.g. CADEC).
                      If provided, enforces the OOD Tie-Test Gate: a candidate arm
                      is tied ONLY if equivalence holds on BOTH in-domain and OOD.

    Returns (selected_config, feasible_count, tie_info) where tie_info is a dict
    with the leader, the tied model set, and the Delta_AUROC CIs vs the leader.
    """
    feas = feasible_arms(configs, tau, E_budget_per_1k, use_gross, use_ece_ci=use_ece_ci)
    if not feas:
        return None, 0, {}

    # Leader = feasible arm with the highest point AUROC.
    leader_cfg = max(feas, key=get_auroc)
    leader_model = leader_cfg['model']

    tie_details = {}
    tied_models = {leader_model}
    for model_name, p1 in model_probs.items():
        if model_name == leader_model:
            tie_details[model_name] = {
                'delta_auroc': 0.0, 'ci_lo': 0.0, 'ci_hi': 0.0,
                'statistical_tie': True, 'tost_tie': True, 'ood_tie': True, 'is_leader': True}
            continue
        if model_name not in {c['model'] for c in feas}:
            continue  # model has no feasible arm under (tau, E)
        d, lo, hi = paired_delta_auroc(
            y_true, model_probs[leader_model], p1,
            n_bootstrap=n_bootstrap, seed=seed)
        
        # In-domain tie condition
        if tost_delta_eq is not None:
            is_indomain_tie = bool(-tost_delta_eq <= lo and hi <= tost_delta_eq)
        else:
            is_indomain_tie = bool(lo <= 0.0 <= hi)

        # OOD Tie-Test Gate condition (if OOD data provided)
        is_ood_tie = True
        if y_ood is not None and model_probs_ood is not None and leader_model in model_probs_ood and model_name in model_probs_ood:
            d_ood, lo_ood, hi_ood = paired_delta_auroc(
                y_ood, model_probs_ood[leader_model], model_probs_ood[model_name],
                n_bootstrap=n_bootstrap, seed=seed)
            if tost_delta_eq is not None:
                is_ood_tie = bool(-tost_delta_eq <= lo_ood and hi_ood <= tost_delta_eq)
            else:
                is_ood_tie = bool(lo_ood <= 0.0 <= hi_ood)

        # Gate pass iff both in-domain and OOD pass
        is_final_tie = is_indomain_tie and is_ood_tie

        tie_details[model_name] = {
            'delta_auroc': d, 'ci_lo': lo, 'ci_hi': hi,
            'statistical_tie': bool(is_indomain_tie),
            'ood_tie': bool(is_ood_tie),
            'final_tie': bool(is_final_tie),
            'is_leader': False}
        if is_final_tie:
            tied_models.add(model_name)

    tied_cfgs = [c for c in feas if c['model'] in tied_models]
    best = min(tied_cfgs, key=lambda c: get_energy(c, use_gross))

    tie_info = {
        'leader': leader_cfg['name'],
        'leader_model': leader_model,
        'tied_models': sorted(tied_models),
        'delta_auroc_vs_leader': tie_details,
        'selected': best['name'],
        'selected_energy_gross': best.get('inf_j_gross'),
        'selected_energy_net': best.get('inf_j_net'),
    }
    return best, len(feas), tie_info
