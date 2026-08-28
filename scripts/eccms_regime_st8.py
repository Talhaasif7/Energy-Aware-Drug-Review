#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST8 — ECC-MS Regime Sweep & Break-Even Analysis (Round 5 Rigorous Overhaul)

WHAT CHANGED vs Round 3
-----------------------
This script no longer hardcodes ANY energy / AUROC / ECE numbers (that was the
integrity failure Round 5 caught, incl. the stale 25.81 / 51.59 GPU comment).
It is now a pure REPORTER over the single source of truth produced by
run_frozen_split_analysis.py:

    results/frozen_split_reconciled.json

From that file it reads:
  * `catalogue`          — the full 12-arm catalogue (4 models x 3 recal) with
                           recomputed test AUROC / ECE, CADEC ECE, and energy;
  * `paired_delta_auroc` — the REAL paired-bootstrap ΔAUROC matrix (tie iff the
                           95% CI includes 0), computed on one shared frozen test
                           set. The tie rule here reuses those CIs (no re-derived
                           margin), so ST8 and the runner cannot drift.

Primary selection  = ECC-MS with the bootstrap-tie rule.
Sensitivity strip  = pre-registered fixed ΔAUROC margins {0.01, 0.02, 0.03}.
RQ4 column         = does the selected arm still satisfy tau on CADEC?
Energy framing     = NET is primary; GROSS is reported with the asymmetry caveat
                     (CPU gross includes ~6.73 W whole-machine idle; GPU gross is
                     board-only), while the feasibility BUDGET axis uses GROSS to
                     match the README table and the runner's grid.

If the reconciled JSON is absent, this script REFUSES to fall back to constants
and instead tells you to run the runner first.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")
RECONCILED_JSON = os.path.join(RESULTS_DIR, "frozen_split_reconciled.json")

MODEL_ORDER = ["Logistic Regression", "LightGBM", "DistilBERT", "PubMedBERT"]
MARGINS = [0.01, 0.02, 0.03]


def reconfigure_stdout():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Energy / AUROC accessors on catalogue arms
# ---------------------------------------------------------------------------
def get_energy(arm, use_gross=True):
    """Budget-axis energy. Default GROSS (matches the runner grid + README).
    Falls back to net then +inf when a value is PENDING (None)."""
    key = 'inf_j_gross' if use_gross else 'inf_j_net'
    v = arm.get(key)
    if v is None:
        v = arm.get('inf_j_net') if use_gross else arm.get('inf_j_gross')
    return v if v is not None else float('inf')


def feasible(arms, tau, E, use_gross=True, use_ece_ci=True):
    res = []
    for a in arms:
        ece_val = a.get('ece_ci_hi', a.get('ece_upper', a['ece'])) if use_ece_ci else a['ece']
        if ece_val <= tau + 1e-12 and get_energy(a, use_gross) <= E + 1e-12:
            res.append(a)
    return res


# ---------------------------------------------------------------------------
# Tie lookup built from the runner's paired-bootstrap matrix
# ---------------------------------------------------------------------------
def build_tie_lookup(paired_rows):
    """{frozenset({model_a, model_b}): statistical_tie(bool)} from the reconciled
    paired_delta_auroc matrix."""
    tie = {}
    for r in paired_rows:
        tie[frozenset({r['model_a'], r['model_b']})] = bool(r['statistical_tie'])
    return tie


def model_level_auroc(catalogue):
    """AUROC is recalibration-invariant, so a model's AUROC is taken from its
    Uncalibrated arm."""
    out = {}
    for a in catalogue:
        if a['recal'] == 'Uncalibrated':
            out[a['model']] = a['auroc']
    # fallback: max over any arm of that model
    for a in catalogue:
        out.setdefault(a['model'], a['auroc'])
    return out


def select_bootstrap_tie(arms, tau, E, tie_lookup, m_auroc, use_gross=True, use_ece_ci=True):
    """Primary ECC-MS rule reusing the precomputed bootstrap tie decisions.
      leader = feasible model with max AUROC;
      tied   = {leader} U {feasible models whose CI vs leader includes 0};
      pick lowest-energy arm among arms whose model is tied.
    """
    feas = feasible(arms, tau, E, use_gross, use_ece_ci=use_ece_ci)
    if not feas:
        return None, 0
    feas_models = {a['model'] for a in feas}
    leader = max(feas_models, key=lambda m: m_auroc.get(m, 0.0))
    tied = {leader}
    for m in feas_models:
        if m == leader:
            continue
        if tie_lookup.get(frozenset({leader, m}), False):
            tied.add(m)
    tied_arms = [a for a in feas if a['model'] in tied]
    best = min(tied_arms, key=lambda a: get_energy(a, use_gross))
    return best, len(feas)


def select_argmax(arms, tau, E, m_auroc, use_gross=True, use_ece_ci=True):
    feas = feasible(arms, tau, E, use_gross, use_ece_ci=use_ece_ci)
    if not feas:
        return None, 0
    return max(feas, key=lambda a: m_auroc.get(a['model'], a['auroc'])), len(feas)


def select_fixed_margin(arms, tau, E, margin, m_auroc, use_gross=True, use_ece_ci=True):
    feas = feasible(arms, tau, E, use_gross, use_ece_ci=use_ece_ci)
    if not feas:
        return None, 0
    leader = max(m_auroc.get(a['model'], a['auroc']) for a in feas)
    tied = [a for a in feas
            if leader - m_auroc.get(a['model'], a['auroc']) <= margin + 1e-12]
    return min(tied, key=lambda a: get_energy(a, use_gross)), len(feas)


def short_label(name):
    parts = name.split(' + ')
    lab = parts[0][:6]
    if len(parts) > 1 and parts[1] != 'Uncalibrated':
        lab += '+' + parts[1][:4]
    return lab


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    reconfigure_stdout()
    print("=" * 100)
    print("  ST8 — ECC-MS REGIME SWEEP (Round 5: driven by frozen_split_reconciled.json)")
    print("=" * 100)

    if not os.path.exists(RECONCILED_JSON):
        print("\n[ABORT] results/frozen_split_reconciled.json not found.")
        print("        ST8 no longer hardcodes energy/AUROC/ECE constants.")
        print("        Run the runner first:")
        print("            python scripts/measure_cpu_energy.py        # optional (energy)")
        print("            python scripts/run_frozen_split_analysis.py  # writes the JSON")
        print("        (and drop the Colab GPU .npz + JSON into results/ beforehand).")
        return

    with open(RECONCILED_JSON, "r", encoding="utf-8") as f:
        recon = json.load(f)

    catalogue = recon["catalogue"]
    tie_lookup = build_tie_lookup(recon.get("paired_delta_auroc", []))
    m_auroc = model_level_auroc(catalogue)
    prov = recon.get("provenance", {})

    print(f"\n[source] {os.path.basename(RECONCILED_JSON)}")
    print(f"  primary_seed={prov.get('primary_seed')} | test_N={prov.get('test_N')} "
          f"| CADEC_N={prov.get('cadec_N')} | split={prov.get('split_source')} "
          f"| n_bootstrap={prov.get('n_bootstrap')}")
    print(f"  CPU energy provenance: {prov.get('cpu_energy_provenance')}")
    print(f"  GPU energy provenance: {prov.get('gpu_energy_provenance')}")
    print(f"  catalogue arms: {len(catalogue)} (expected 12 = 4 models x 3 recal)")

    # ---- energy availability guard ----
    pending = [a['name'] for a in catalogue if a.get('inf_j_gross') is None]
    if pending:
        print(f"\n[WARN] {len(pending)} arm(s) have PENDING energy (no GPU saturated "
              f"run yet): {pending}")
        print("       Regime cells needing those arms will treat them as infeasible.")

    # ---- paired-bootstrap tie matrix (echo the evidence) ----
    print("\n" + "-" * 100)
    print("  PAIRED-BOOTSTRAP ΔAUROC (shared resamples; tie iff 95% CI includes 0)")
    print("-" * 100)
    for r in recon.get("paired_delta_auroc", []):
        verdict = "TIE" if r['statistical_tie'] else "DISTINGUISHABLE"
        print(f"  {r['model_a']:22} - {r['model_b']:22} "
              f"Δ={r['delta_auroc']:+.4f}  CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  {verdict}")

    # ---- dense grids for the regime map ----
    tau_grid = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    E_grid = [0.1, 0.5, 1.0, 10.0, 30.0, 60.0, 100.0, 120.0, 150.0, 200.0]

    def print_map(title, selector):
        print(f"\n--- {title} (budget = GROSS J/1k) ---")
        print(f"{'tau (ECE)':>10}", end='')
        for E in E_grid:
            print(f"  E<={E:>6.1f}J", end='')
        print()
        print("-" * (10 + 12 * len(E_grid)))
        for tau in tau_grid:
            print(f"  tau<={tau:.2f} ", end='')
            for E in E_grid:
                sel, _ = selector(tau, E)
                print(f"  {(short_label(sel['name']) if sel else '---'):>10}", end='')
            print()

    print_map("ECC-MS REGIME MAP (argmax AUROC)",
              lambda t, e: select_argmax(catalogue, t, e, m_auroc))
    print_map("ECC-MS REGIME MAP (bootstrap-tie rule = PRIMARY)",
              lambda t, e: select_bootstrap_tie(catalogue, t, e, tie_lookup, m_auroc))

    # ---- detailed reconcile table incl. reviewer's disputed cells ----
    print("\n--- DETAILED SELECTION TABLE (bootstrap tie + fixed-margin strip + RQ4) ---")
    detail_rows = []
    reconcile_cells = [(0.03, 0.5), (0.05, 60.0), (0.07, 10.0), (0.07, 60.0),
                       (0.10, 0.5), (0.10, 10.0), (0.10, 60.0),
                       (0.05, 120.0), (0.07, 120.0), (0.10, 120.0),
                       (0.10, 150.0), (0.10, 200.0)]
    for tau, E in reconcile_cells:
        argmax_sel, n = select_argmax(catalogue, tau, E, m_auroc)
        tie_sel, _ = select_bootstrap_tie(catalogue, tau, E, tie_lookup, m_auroc)
        margins = {f"m{m}": select_fixed_margin(catalogue, tau, E, m, m_auroc)[0]
                   for m in MARGINS}
        cadec_ok = None
        cadec_tie = None
        grid_row = next((r for r in recon.get("eccms_grid", [])
                         if abs(r["tau"] - tau) < 1e-5 and abs(r["E_gross_J_per_1k"] - E) < 1e-5), {})
        if tie_sel is not None:
            cadec_ok = bool(tie_sel.get('cadec_ece', 9.9) <= tau + 1e-12)
            cadec_tie = grid_row.get("selected_in_cadec_tie_band")
        detail_rows.append({
            'tau': tau, 'E(gross J/1k)': E, 'Feasible': n,
            'Argmax': short_label(argmax_sel['name']) if argmax_sel else 'None',
            'BootstrapTie': short_label(tie_sel['name']) if tie_sel else 'None',
            'Tie AUROC': f"{tie_sel['auroc']:.4f}" if tie_sel else '-',
            'Tie NetJ/1k': (f"{tie_sel['inf_j_net']:.4f}"
                            if tie_sel and tie_sel.get('inf_j_net') is not None else '-'),
            'CADEC tau-ok (RQ4)': cadec_ok,
            'CADEC Tie-Band': cadec_tie,
            'm=0.01': short_label(margins['m0.01']['name']) if margins['m0.01'] else 'None',
            'm=0.02': short_label(margins['m0.02']['name']) if margins['m0.02'] else 'None',
            'm=0.03': short_label(margins['m0.03']['name']) if margins['m0.03'] else 'None',
        })
    print(pd.DataFrame(detail_rows).to_string(index=False))

    # ---- feasible-count reconciliation (the reviewer's 8/11 dispute) ----
    print("\n--- FEASIBLE-ARM COUNT RECONCILIATION (full 12-arm catalogue, GROSS budget) ---")
    for tau, E in [(0.07, 60.0), (0.10, 60.0)]:
        feas = feasible(catalogue, tau, E, use_gross=True)
        print(f"  tau<={tau:.2f}, E<={E:.1f}J : {len(feas)} feasible")
        for a in feas:
            print(f"      - {a['name']:32} ECE={a['ece']:.4f} "
                  f"grossJ/1k={a['inf_j_gross'] if a['inf_j_gross'] is not None else 'PENDING'}")

    # ---- energy asymmetry: NET primary, GROSS secondary w/ caveat ----
    print("\n--- ENERGY ASYMMETRY (NET is PRIMARY) ---")
    print("  Caveat: CPU GROSS includes ~6.73 W whole-machine idle; GPU GROSS is "
          "board-only.\n          NET (load - idle) is the like-for-like comparison.")
    e = {}
    for m in MODEL_ORDER:
        arm = next((a for a in catalogue if a['model'] == m and a['recal'] == 'Uncalibrated'), None)
        if arm:
            e[m] = {'net': arm.get('inf_j_net'), 'gross': arm.get('inf_j_gross')}

    def fmt(x, nd=4):
        return "PENDING" if x is None else f"{x:.{nd}f}"

    for m in MODEL_ORDER:
        if m in e:
            print(f"  {m:22} Net={fmt(e[m]['net'])} J/1k | Gross={fmt(e[m]['gross'], 2)} J/1k")

    def ratio(a, b, key):
        va = e.get(a, {}).get(key)
        vb = e.get(b, {}).get(key)
        if va is None or vb is None or vb == 0:
            return None
        return va / vb

    print("\n  NET-to-NET ratios (primary):")
    for a, b in [("PubMedBERT", "Logistic Regression"),
                 ("DistilBERT", "Logistic Regression"),
                 ("LightGBM", "Logistic Regression")]:
        rr = ratio(a, b, 'net')
        print(f"    {a} / {b:22}: {'PENDING' if rr is None else f'{rr:,.0f}x'}")
    print("  GROSS-to-GROSS ratios (secondary, board-vs-whole-machine asymmetric):")
    for a, b in [("PubMedBERT", "Logistic Regression"),
                 ("DistilBERT", "Logistic Regression")]:
        rr = ratio(a, b, 'gross')
        print(f"    {a} / {b:22}: {'PENDING' if rr is None else f'{rr:,.0f}x'}")

    # ---- break-even (gross, since it is the deployment draw) ----
    pub_gross = e.get("PubMedBERT", {}).get('gross')
    lr_gross = e.get("Logistic Regression", {}).get('gross')
    if pub_gross and lr_gross:
        print("\n--- BREAK-EVEN INFERENCE VOLUME (GROSS) ---")
        print(f"  {'Daily Budget (J)':>17} | {'PubMedBERT':>16} | {'LR':>16} | {'Ratio':>6}")
        print("  " + "-" * 66)
        for budget in [1000, 10000, 50000, 100000]:
            n_pub = budget / pub_gross * 1000
            n_lr = budget / lr_gross * 1000
            print(f"  {budget:>17,} J | {n_pub:>12,.0f} s | {n_lr:>12,.0f} s | "
                  f"{n_lr / n_pub:>5.0f}x")

        daily_j = pub_gross * 1000  # 1M sentences = 1000 x 1k
        daily_wh = daily_j / 3600
        print("\n--- ABSOLUTE ENERGY SCALE ---")
        print(f"  PubMedBERT screening 1M sentences/day = {daily_j:,.0f} J = "
              f"{daily_wh:.1f} Wh/day (~a phone charge).")
        print("  The ratio is dramatic but absolute stakes are modest at realistic "
              "volumes: this is about\n  deployment feasibility under constraint, "
              "not an environmental-impact claim.")

    # ---- regime map figure ----
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
    colors = {m: i for i, m in enumerate(MODEL_ORDER)}
    Z = np.full((len(tau_grid), len(E_grid)), -1, dtype=int)
    for i, tau in enumerate(tau_grid):
        for j, E in enumerate(E_grid):
            sel, _ = select_bootstrap_tie(catalogue, tau, E, tie_lookup, m_auroc)
            if sel:
                Z[i, j] = colors[sel['model']]
    cmap = ListedColormap(['#dfe6e9', '#0984e3', '#00b894', '#e17055', '#6c5ce7'])
    ax1.imshow(Z, aspect='auto', cmap=cmap, vmin=-1, vmax=len(MODEL_ORDER) - 1)
    ax1.set_xticks(range(len(E_grid)))
    ax1.set_xticklabels([f"{x}" for x in E_grid], fontsize=8)
    ax1.set_yticks(range(len(tau_grid)))
    ax1.set_yticklabels([f"{t:.2f}" for t in tau_grid], fontsize=8)
    ax1.set_xlabel("Energy budget E (GROSS J per 1k inferences)", fontsize=10)
    ax1.set_ylabel("Calibration threshold tau (max ECE)", fontsize=10)
    ax1.set_title("ECC-MS Regime Map (bootstrap-tie rule)", fontsize=11, fontweight='bold')
    for i in range(len(tau_grid)):
        for j in range(len(E_grid)):
            if Z[i, j] >= 0:
                ax1.text(j, i, MODEL_ORDER[Z[i, j]][:6], ha='center', va='center',
                         fontsize=6, color='black', fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "st8_regime_map.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[artifact] Regime map: {plot_path}")

    # ---- write ST8 tables JSON for README reconciliation ----
    st8_out = {
        "source": os.path.basename(RECONCILED_JSON),
        "provenance": prov,
        "detailed_selection": detail_rows,
        "feasible_counts": {
            f"tau={t}_E={E}": len(feasible(catalogue, t, E, use_gross=True))
            for (t, E) in reconcile_cells},
        "model_energy": e,
        "pending_energy_arms": pending,
    }
    st8_path = os.path.join(RESULTS_DIR, "st8_regime_reconciled.json")
    with open(st8_path, "w", encoding="utf-8") as f:
        json.dump(st8_out, f, indent=2)
    print(f"[artifact] ST8 tables: {st8_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
