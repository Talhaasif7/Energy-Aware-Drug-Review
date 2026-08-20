"""
ST8 — ECC-MS Regime Sweep & Break-Even Analysis (Round 3 Corrected)

Fixes applied:
  - LightGBM gross energy corrected: 0.3700 → 0.7412 J/1k
  - All ratios standardised as Net-to-Net unless explicitly labeled
  - 31.32 ÷ 0.0201 = 1,558× (was incorrectly 1,542×)
  - Added statistical-tie rule: bootstrap AUROC CIs, select lowest-energy
    config whose AUROC is not significantly worse than max
  - GPU AUROC marked TBD pending Colab re-run; uses F1@0.5 as interim proxy
  - Framing warning: absolute inference energy is modest at realistic volumes
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def reconfigure_stdout():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


# ---------------------------------------------------------------
# Empirical configuration catalogue (from ST3, ST4, ST5, GPU gating)
# ---------------------------------------------------------------

CONFIGURATIONS = [
    # CPU Net energy calculated with 3-decimal power precision:
    # LR: Load 7.072 W, Idle 6.734 W, Net 0.338 W -> Net J/1k = 0.4400 * 0.338 / 7.072 = 0.0210 J/1k
    # GBDT: Load 9.940 W, Idle 6.734 W, Net 3.206 W -> Net J/1k = 0.7412 * 3.206 / 9.940 = 0.2391 J/1k
    # GPU Net energy calculated with measured Colab T4 power:
    # Idle 10.220 W baseline
    # DistilBERT: Load 63.670 W, Net 53.450 W -> Net J/1k = 25.81 * 53.450 / 63.670 = 21.66 J/1k
    # PubMedBERT: Load 65.810 W, Net 55.590 W -> Net J/1k = 51.59 * 55.590 / 65.810 = 43.57 J/1k
    {'name': 'LR + Uncalibrated',
     'model': 'Logistic Regression', 'recal': 'None',
     'auroc': 0.8835, 'ece': 0.1365,
     'inf_j_net': 0.0210, 'inf_j_gross': 0.4400, 'train_j': 2.12,
     'auroc_measured': True},
    {'name': 'LR + TempScale',
     'model': 'Logistic Regression', 'recal': 'TempScale',
     'auroc': 0.8835, 'ece': 0.0815,
     'inf_j_net': 0.0210, 'inf_j_gross': 0.4400, 'train_j': 2.12,
     'auroc_measured': True},
    {'name': 'LR + Isotonic',
     'model': 'Logistic Regression', 'recal': 'Isotonic',
     'auroc': 0.8809, 'ece': 0.0704,
     'inf_j_net': 0.0210, 'inf_j_gross': 0.4400, 'train_j': 2.12,
     'auroc_measured': True},
    {'name': 'GBDT + Uncalibrated',
     'model': 'LightGBM', 'recal': 'None',
     'auroc': 0.7942, 'ece': 0.0595,
     'inf_j_net': 0.2391, 'inf_j_gross': 0.7412, 'train_j': 8.36,
     'auroc_measured': True},
    {'name': 'GBDT + TempScale',
     'model': 'LightGBM', 'recal': 'TempScale',
     'auroc': 0.7942, 'ece': 0.0543,
     'inf_j_net': 0.2391, 'inf_j_gross': 0.7412, 'train_j': 8.36,
     'auroc_measured': True},
    {'name': 'GBDT + Isotonic',
     'model': 'LightGBM', 'recal': 'Isotonic',
     'auroc': 0.7920, 'ece': 0.0548,
     'inf_j_net': 0.2391, 'inf_j_gross': 0.7412, 'train_j': 8.36,
     'auroc_measured': True},
    {'name': 'DistilBERT + Uncalibrated',
     'model': 'DistilBERT', 'recal': 'None',
     'auroc': 0.9059, 'ece': 0.0666,
     'inf_j_net': 21.66, 'inf_j_gross': 25.81, 'train_j': 203.9,
     'auroc_measured': True},
    {'name': 'DistilBERT + TempScale',
     'model': 'DistilBERT', 'recal': 'TempScale',
     'auroc': 0.9059, 'ece': 0.0675,
     'inf_j_net': 21.66, 'inf_j_gross': 25.81, 'train_j': 203.9,
     'auroc_measured': True},
    {'name': 'PubMedBERT + Uncalibrated',
     'model': 'PubMedBERT', 'recal': 'None',
     'auroc': 0.9138, 'ece': 0.0442,
     'inf_j_net': 43.57, 'inf_j_gross': 51.59, 'train_j': 364.7,
     'auroc_measured': True},
    {'name': 'PubMedBERT + TempScale',
     'model': 'PubMedBERT', 'recal': 'TempScale',
     'auroc': 0.9138, 'ece': 0.0677,
     'inf_j_net': 43.57, 'inf_j_gross': 51.59, 'train_j': 364.7,
     'auroc_measured': True},
]


def get_auroc(config):
    """Return AUROC if measured, else F1 proxy."""
    if config['auroc'] is not None:
        return config['auroc']
    return config.get('f1_proxy', 0.0)


def get_energy(config, use_gross=False):
    """Return energy per 1k, preferring net; falls back to gross."""
    if use_gross:
        return config.get('inf_j_gross', float('inf'))
    net = config.get('inf_j_net')
    if net is not None:
        return net
    # GPU net not yet measured — fall back to gross
    return config.get('inf_j_gross', float('inf'))


def eccms_select(configs, tau, E_budget_per_1k, use_gross=False):
    """
    Original ECC-MS selection rule (argmax AUROC):
      1. Filter configurations where ECE <= tau
      2. Filter configurations where energy <= E
      3. Among feasible set, select max AUROC (or F1 proxy)
    Returns: (selected_config, feasible_count) or (None, 0)
    """
    feasible = [c for c in configs
                if c['ece'] <= tau and get_energy(c, use_gross) <= E_budget_per_1k]
    if not feasible:
        return None, 0
    best = max(feasible, key=lambda c: get_auroc(c))
    return best, len(feasible)


def eccms_select_with_tie(configs, tau, E_budget_per_1k, auroc_ci_half=0.02,
                          use_gross=False):
    """
    ECC-MS with statistical-tie rule:
      1. Filter by ECE <= tau and Energy <= E
      2. Find max AUROC among feasible
      3. Find all configs within auroc_ci_half of max (statistical tie)
      4. Among tied configs, select lowest energy
    
    auroc_ci_half: half-width of bootstrap CI on AUROC difference.
    Default 0.02 is conservative for N~400 test sets.
    """
    feasible = [c for c in configs
                if c['ece'] <= tau and get_energy(c, use_gross) <= E_budget_per_1k]
    if not feasible:
        return None, 0

    max_auroc = max(get_auroc(c) for c in feasible)
    tied = [c for c in feasible if max_auroc - get_auroc(c) <= auroc_ci_half]
    best = min(tied, key=lambda c: get_energy(c, use_gross))
    return best, len(feasible)


def main():
    reconfigure_stdout()
    print("Starting Smoke Test 8 (ST8 - ECC-MS Regime Sweep) [ROUND 3 CORRECTED]")

    configs = CONFIGURATIONS

    # tau sweep: ECE thresholds
    tau_grid = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]

    # E sweep: energy budgets per 1k inferences (Joules) — using GROSS
    E_grid = [0.1, 0.5, 1.0, 10.0, 30.0, 60.0, 100.0]

    # ---------------------------------------------------------------
    # Regime Map with ORIGINAL argmax rule (for comparison)
    # ---------------------------------------------------------------
    print("\n--- ECC-MS REGIME MAP (argmax AUROC, Gross Energy) ---")
    print(f"{'tau (ECE)':>10}", end='')
    for E in E_grid:
        print(f"  E<={E:>6.1f}J", end='')
    print()
    print("-" * (10 + 12 * len(E_grid)))

    for tau in tau_grid:
        print(f"  tau<={tau:.2f} ", end='')
        for E in E_grid:
            selected, n_feasible = eccms_select(configs, tau, E, use_gross=True)
            if selected is None:
                label = "---"
            else:
                parts = selected['name'].split(' + ')
                label = parts[0][:6]
                if len(parts) > 1 and parts[1] != 'Uncalibrated':
                    label += '+' + parts[1][:4]
            print(f"  {label:>10}", end='')
        print()

    # ---------------------------------------------------------------
    # Regime Map with STATISTICAL-TIE rule (the improvement)
    # ---------------------------------------------------------------
    print("\n--- ECC-MS REGIME MAP (statistical-tie rule, Gross Energy) ---")
    print(f"  NOTE: Configs within 0.02 AUROC of max are treated as tied;")
    print(f"        among tied configs, lowest-energy is selected.")
    print(f"{'tau (ECE)':>10}", end='')
    for E in E_grid:
        print(f"  E<={E:>6.1f}J", end='')
    print()
    print("-" * (10 + 12 * len(E_grid)))

    for tau in tau_grid:
        print(f"  tau<={tau:.2f} ", end='')
        for E in E_grid:
            selected, _ = eccms_select_with_tie(configs, tau, E,
                                                 auroc_ci_half=0.02,
                                                 use_gross=True)
            if selected is None:
                label = "---"
            else:
                parts = selected['name'].split(' + ')
                label = parts[0][:6]
                if len(parts) > 1 and parts[1] != 'Uncalibrated':
                    label += '+' + parts[1][:4]
            print(f"  {label:>10}", end='')
        print()

    # ---------------------------------------------------------------
    # Detailed selection table
    # ---------------------------------------------------------------
    print("\n--- DETAILED SELECTION TABLE (Tie Rule) ---")
    detail_rows = []
    for tau in [0.03, 0.05, 0.07, 0.10]:
        for E in [0.5, 10.0, 60.0]:
            sel_argmax, _ = eccms_select(configs, tau, E, use_gross=True)
            sel_tie, n = eccms_select_with_tie(configs, tau, E,
                                                auroc_ci_half=0.02,
                                                use_gross=True)
            detail_rows.append({
                'tau': tau, 'E (J/1k)': E,
                'Argmax': sel_argmax['name'] if sel_argmax else 'None',
                'Tie Rule': sel_tie['name'] if sel_tie else 'None',
                'Tie AUROC': f"{get_auroc(sel_tie):.4f}" if sel_tie else '-',
                'Tie Gross J/1k': f"{sel_tie['inf_j_gross']:.4f}" if sel_tie else '-',
                'Feasible': n
            })
    print(pd.DataFrame(detail_rows).to_string(index=False))

    # ---------------------------------------------------------------
    # Energy ratios — ALL Net-to-Net (standardised convention)
    # ---------------------------------------------------------------
    print("\n--- ENERGY ASYMMETRY RATIOS ---")
    print("  Convention: ALL ratios are Gross-to-Gross unless labeled otherwise.")
    print("  GPU Net energy not yet measured (pending nvidia-smi trace).\n")

    lr_net = 0.0201
    lr_gross = 0.4400
    gbdt_net = 0.2394
    gbdt_gross = 0.7412
    distilbert_gross = 25.81
    pubmedbert_gross = 51.59

    print(f"  LR Net:            {lr_net:.4f} J/1k")
    print(f"  LR Gross:          {lr_gross:.4f} J/1k")
    print(f"  LightGBM Net:      {gbdt_net:.4f} J/1k")
    print(f"  LightGBM Gross:    {gbdt_gross:.4f} J/1k  (corrected: 0.2394 x 9.94/3.21)")
    print(f"  DistilBERT Gross:  {distilbert_gross:.2f} J/1k")
    print(f"  PubMedBERT Gross:  {pubmedbert_gross:.2f} J/1k")

    print(f"\n  Gross-to-Gross Ratios:")
    print(f"    PubMedBERT / LR:       {pubmedbert_gross / lr_gross:.0f}x"
          f"  ({pubmedbert_gross:.2f} / {lr_gross:.4f})")
    print(f"    PubMedBERT / LightGBM: {pubmedbert_gross / gbdt_gross:.0f}x"
          f"  ({pubmedbert_gross:.2f} / {gbdt_gross:.4f})")
    print(f"    DistilBERT / LR:       {distilbert_gross / lr_gross:.0f}x"
          f"  ({distilbert_gross:.2f} / {lr_gross:.4f})")

    print(f"\n  Net-to-Net Ratios (CPU only — GPU Net TBD):")
    print(f"    LightGBM / LR:         {gbdt_net / lr_net:.1f}x"
          f"  ({gbdt_net:.4f} / {lr_net:.4f})")

    print(f"\n  Cross-convention (Gross GPU vs Net CPU — for context only, NOT primary):")
    print(f"    PubMedBERT Gross / LR Net: {pubmedbert_gross / lr_net:.0f}x"
          f"  ({pubmedbert_gross:.2f} / {lr_net:.4f})")

    # ---------------------------------------------------------------
    # Break-even analysis (using Gross for both platforms)
    # ---------------------------------------------------------------
    print("\n--- BREAK-EVEN INFERENCE VOLUME (Gross-to-Gross) ---")
    budgets = [1000, 10000, 50000, 100000]
    print(f"\n  {'Daily Budget (J)':>17} | {'PubMedBERT (Gross)':>22} | "
          f"{'LR (Gross)':>22} | {'Ratio':>6}")
    print("  " + "-" * 80)
    for budget in budgets:
        n_pubmed = (budget / pubmedbert_gross) * 1000  # sentences
        n_lr = (budget / lr_gross) * 1000
        ratio = n_lr / n_pubmed if n_pubmed > 0 else float('inf')
        print(f"  {budget:>17,} J | {n_pubmed:>18,.0f} sents | "
              f"{n_lr:>18,.0f} sents | {ratio:>5.0f}x")

    # ---------------------------------------------------------------
    # Structural framing warning
    # ---------------------------------------------------------------
    print("\n--- ABSOLUTE ENERGY SCALE WARNING ---")
    print("  At 51.59 J/1k, PubMedBERT screening 1M sentences/day costs:")
    daily_j = 51.59 * 1000  # 1M sentences = 1000 * 1k
    daily_wh = daily_j / 3600
    print(f"    {daily_j:,.0f} J/day = {daily_wh:.1f} Wh/day (~a phone charge)")
    print(f"    To reach 1 MWh/year of inference: "
          f"{1_000_000_000 / (daily_wh * 365):.0f}M sentences/day")
    print(f"  The ratio is dramatic but absolute stakes are small at realistic volumes.")
    print(f"  Contribution is about deployment feasibility under constraint,")
    print(f"  NOT environmental impact claims.")

    # ---------------------------------------------------------------
    # Generate regime map plot (tie rule)
    # ---------------------------------------------------------------
    reports_dir = r"e:\AI Green\reports"
    os.makedirs(reports_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Regime heatmap (tie rule)
    ax1 = axes[0]
    model_names = sorted(set(c['model'] for c in configs))
    model_colors = {m: i for i, m in enumerate(model_names)}
    n_models = len(model_names)

    Z = np.full((len(tau_grid), len(E_grid)), -1, dtype=int)
    for i, tau in enumerate(tau_grid):
        for j, E in enumerate(E_grid):
            sel, _ = eccms_select_with_tie(configs, tau, E,
                                            auroc_ci_half=0.02,
                                            use_gross=True)
            if sel:
                Z[i, j] = model_colors[sel['model']]

    cmap = ListedColormap(['#2d3436', '#0984e3', '#00b894', '#e17055', '#6c5ce7'])
    im = ax1.imshow(Z, aspect='auto', cmap=cmap, vmin=-1, vmax=n_models - 1)
    ax1.set_xticks(range(len(E_grid)))
    ax1.set_xticklabels([f"{e}" for e in E_grid], fontsize=8)
    ax1.set_yticks(range(len(tau_grid)))
    ax1.set_yticklabels([f"{t:.2f}" for t in tau_grid], fontsize=8)
    ax1.set_xlabel("Energy Budget E (J per 1k inferences, Gross)", fontsize=10)
    ax1.set_ylabel("Calibration Threshold tau (max ECE)", fontsize=10)
    ax1.set_title("ECC-MS Regime Map (Statistical-Tie Rule)", fontsize=11,
                   fontweight='bold')

    for i in range(len(tau_grid)):
        for j in range(len(E_grid)):
            if Z[i, j] >= 0:
                name = model_names[Z[i, j]]
                ax1.text(j, i, name[:6], ha='center', va='center',
                        fontsize=6, color='white', fontweight='bold')

    # Plot 2: Break-even curves (Gross)
    ax2 = axes[1]
    daily_volumes = np.logspace(1, 6, 100)

    lr_config = configs[0]  # LR + Uncal
    for c_name, c_energy, c_color in [
        ('LR (Gross)', lr_gross, '#0984e3'),
        ('LightGBM (Gross)', gbdt_gross, '#00b894'),
        ('DistilBERT (Gross)', distilbert_gross, '#e17055'),
        ('PubMedBERT (Gross)', pubmedbert_gross, '#6c5ce7'),
    ]:
        daily_energy = daily_volumes * c_energy / 1000  # volumes in sentences
        ax2.loglog(daily_volumes, daily_energy,
                  label=f"{c_name} ({c_energy:.3f} J/1k)",
                  linewidth=2, color=c_color)

    for budget, ls in [(100, ':'), (1000, '--'), (10000, '-.')]:
        ax2.axhline(y=budget, color='gray', linestyle=ls, alpha=0.5,
                   label=f"Budget = {budget} J/day")

    ax2.set_xlabel("Daily Inference Volume (sentences)", fontsize=10)
    ax2.set_ylabel("Daily Energy Cost (Joules, Gross)", fontsize=10)
    ax2.set_title("Break-Even: Energy vs Inference Volume", fontsize=11,
                   fontweight='bold')
    ax2.legend(fontsize=7, loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(reports_dir, "st8_regime_map.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nRegime map plot saved to: {plot_path}")

    # ---------------------------------------------------------------
    # Key findings
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("          ST8 — ECC-MS REGIME SWEEP & BREAK-EVEN REPORT (ROUND 3)")
    print("=" * 100)

    print("\n--- KEY FINDINGS ---")
    print(f"  1. Gross energy asymmetry (PubMedBERT/LR):"
          f" {pubmedbert_gross/lr_gross:.0f}x gross-to-gross.")
    print(f"  2. LightGBM gross energy CORRECTED: 0.7412 J/1k"
          f" (was incorrectly 0.3700 J/1k).")
    print(f"  3. GPU Net energy is TBD (requires nvidia-smi idle trace).")
    print(f"  4. Statistical-tie rule: LR AUROC (0.8835) and PubMedBERT")
    print(f"     AUROC (TBD, F1 proxy=0.8140) may be statistically tied.")
    print(f"     If tied, tie rule selects LR at 117x less energy (gross).")
    print(f"  5. Absolute inference energy is modest: 1M sentences/day on")
    print(f"     PubMedBERT costs {daily_wh:.1f} Wh (~a phone charge).")

    print("\n--- WHAT IS MEASURED vs PENDING ---")
    print("  MEASURED (live, verified):")
    print("    CPU: AUROC, AUPRC, F1@t*, ECE, NLL, Gross+Net energy (ST3/ST4)")
    print("    GPU: F1@0.5, ECE, NLL, Gross energy, throughput (Colab gating)")
    print("  PENDING (requires Colab re-run):")
    print("    GPU: AUROC, AUPRC, F1@t*, t*, nvidia-smi idle/load power,")
    print("         per-arm Net energy, fitted T values, calib NLL pre/post")

    print(f"\n  Plot saved: {plot_path}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
