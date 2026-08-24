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

def paired_delta_auroc(y_true, p_leader, p_other, n_bootstrap=2000, seed=42):
    """
    Paired bootstrap of Delta_AUROC = AUROC(leader) - AUROC(other) on a single
    shared test set. Both models are scored on the SAME resampled indices.

    Returns (delta_point, ci_lo, ci_hi).
    """
    y_true = np.asarray(y_true)
    p_leader = np.asarray(p_leader, dtype=float)
    p_other = np.asarray(p_other, dtype=float)

    delta_point = float(roc_auc_score(y_true, p_leader) - roc_auc_score(y_true, p_other))

    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        a_lead = roc_auc_score(yb, p_leader[idx])
        a_oth = roc_auc_score(yb, p_other[idx])
        deltas.append(a_lead - a_oth)

    if not deltas:
        return delta_point, float('nan'), float('nan')
    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    return delta_point, ci_lo, ci_hi


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

def feasible_arms(configs, tau, E_budget_per_1k, use_gross=False):
    """Arms clearing BOTH the calibration threshold (ECE <= tau) and the energy
    budget (energy <= E). Returns the filtered list (order preserved)."""
    return [c for c in configs
            if c['ece'] <= tau + 1e-12
            and get_energy(c, use_gross) <= E_budget_per_1k + 1e-12]


# ---------------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------------

def eccms_select_argmax(configs, tau, E_budget_per_1k, use_gross=False):
    """Baseline: among feasible arms, pick the max-AUROC arm."""
    feas = feasible_arms(configs, tau, E_budget_per_1k, use_gross)
    if not feas:
        return None, 0
    return max(feas, key=get_auroc), len(feas)


def eccms_select_fixed_margin(configs, tau, E_budget_per_1k, margin=0.02,
                              use_gross=False):
    """Pre-registered practical-equivalence margin: tie if leader_auroc - auroc
    <= margin; among tied arms pick lowest energy."""
    feas = feasible_arms(configs, tau, E_budget_per_1k, use_gross)
    if not feas:
        return None, 0
    leader = max(get_auroc(c) for c in feas)
    tied = [c for c in feas if (leader - get_auroc(c)) <= margin + 1e-12]
    best = min(tied, key=lambda c: get_energy(c, use_gross))
    return best, len(feas)


def eccms_select_bootstrap_tie(configs, tau, E_budget_per_1k, y_true,
                               model_probs, n_bootstrap=2000, seed=42,
                               use_gross=False):
    """
    Primary ECC-MS rule with the empirical paired-bootstrap tie test.

    configs      : list of arm dicts; each must have keys
                   name, model, ece, auroc, inf_j_net/inf_j_gross.
    model_probs  : {model_name: p1_test} on the shared frozen test set.

    Returns (selected_config, feasible_count, tie_info) where tie_info is a dict
    with the leader, the tied model set, and the Delta_AUROC CIs vs the leader.
    """
    feas = feasible_arms(configs, tau, E_budget_per_1k, use_gross)
    if not feas:
        return None, 0, {}

    # Leader = feasible arm with the highest point AUROC.
    leader_cfg = max(feas, key=get_auroc)
    leader_model = leader_cfg['model']

    # A feasible arm is tied with the leader if its BASE MODEL's Delta_AUROC vs
    # the leader model has a bootstrap CI that includes zero (or it is the same
    # model, which is trivially tied). AUROC is recalibration-invariant.
    tie_details = {}
    tied_models = {leader_model}
    for model_name, p1 in model_probs.items():
        if model_name == leader_model:
            tie_details[model_name] = {
                'delta_auroc': 0.0, 'ci_lo': 0.0, 'ci_hi': 0.0,
                'statistical_tie': True, 'is_leader': True}
            continue
        if model_name not in {c['model'] for c in feas}:
            continue  # model has no feasible arm under (tau, E)
        d, lo, hi = paired_delta_auroc(
            y_true, model_probs[leader_model], p1,
            n_bootstrap=n_bootstrap, seed=seed)
        is_tie = (lo <= 0.0 <= hi)
        tie_details[model_name] = {
            'delta_auroc': d, 'ci_lo': lo, 'ci_hi': hi,
            'statistical_tie': bool(is_tie), 'is_leader': False}
        if is_tie:
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
