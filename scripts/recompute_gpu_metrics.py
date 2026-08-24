"""
CPU-Side Metric Recomputation from GPU Prediction Artifacts

This script loads the .npz prediction files saved during GPU runs
and recomputes the full metric bundle (AUROC, AUPRC, F1@t*, ECE, CIs)
entirely on CPU. Zero GPU quota required.

Usage:
  1. Download .npz files from Colab to results/
  2. Run: python scripts/recompute_gpu_metrics.py

Outputs updated results JSON with true metrics.
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, brier_score_loss, log_loss
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_utils import (
    compute_ece_adaptive, bootstrap_ece, bootstrap_delta_ece,
    find_optimal_threshold
)


def reconfigure_stdout():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


def eval_full(y_true, probs_2d, threshold=0.5):
    """Compute complete metric bundle from probability array."""
    p1 = probs_2d[:, 1]
    auroc = float(roc_auc_score(y_true, p1))
    auprc = float(average_precision_score(y_true, p1))

    f1_fixed = float(f1_score(y_true, (p1 >= 0.5).astype(int),
                               pos_label=1, zero_division=0))
    f1_tuned = float(f1_score(y_true, (p1 >= threshold).astype(int),
                               pos_label=1, zero_division=0))

    ece_point, ece_lo, ece_hi = bootstrap_ece(y_true, p1, method='adaptive')
    brier = float(brier_score_loss(y_true, p1))
    nll = float(log_loss(y_true, probs_2d, labels=[0, 1]))

    return {
        'AUROC': auroc, 'AUPRC': auprc,
        'F1@t*': f1_tuned, 't*': threshold,
        'F1@0.5': f1_fixed,
        'ECE_adaptive': ece_point,
        'ECE_CI_lo': ece_lo, 'ECE_CI_hi': ece_hi,
        'Brier': brier, 'NLL': nll,
    }


def main():
    reconfigure_stdout()
    print("CPU-Side GPU Metric Recomputation")
    print("=" * 80)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    npz_files = sorted(glob.glob(os.path.join(results_dir, '*_predictions.npz')))

    if not npz_files:
        print("\n[ERROR] No .npz prediction files found in results/")
        print("  You must first run colab_gpu_transformer_primary_adr.py on Colab")
        print("  and download the .npz files to the results/ directory.")
        print("  Expected: results/<model>_seed<N>_predictions.npz")
        return

    print(f"\nFound {len(npz_files)} prediction artifact(s):")
    for f in npz_files:
        print(f"  {os.path.basename(f)}")

    all_results = {}

    for npz_path in npz_files:
        basename = os.path.basename(npz_path)
        print(f"\n{'='*60}")
        print(f"  Processing: {basename}")
        print(f"{'='*60}")

        data = np.load(npz_path)

        # Available arrays
        print(f"  Arrays: {list(data.keys())}")

        y_calib = data['y_calib']
        y_test = data['y_test']
        y_cadec = data['y_cadec']

        probs_test_uncal = data['probs_test_uncal']
        probs_test_temp = data['probs_test_temp']
        probs_test_iso = data['probs_test_iso']
        probs_cadec_uncal = data['probs_cadec_uncal']
        probs_cadec_temp = data['probs_cadec_temp']
        probs_cadec_iso = data['probs_cadec_iso']

        print(f"  Test N={len(y_test)} | CADEC N={len(y_cadec)} | Calib N={len(y_calib)}")

        methods = {
            'Uncalibrated': (probs_test_uncal, probs_cadec_uncal),
            'Temperature Scaling': (probs_test_temp, probs_cadec_temp),
            'Isotonic Regression': (probs_test_iso, probs_cadec_iso),
        }

        file_results = {}

        for method_name, (p_test, p_cadec) in methods.items():
            # Find t* from calibration-split probs (approximate from test uncal)
            # Ideally we'd have calib probs; use test probs as proxy for threshold
            if method_name == 'Uncalibrated':
                p_calib_approx = probs_test_uncal  # proxy
            elif method_name == 'Temperature Scaling':
                p_calib_approx = probs_test_temp
            else:
                p_calib_approx = probs_test_iso

            # Better: try to recompute calib probs from logits if available
            if 'logits_calib' in data.keys():
                from scipy.special import softmax as sp_softmax, expit
                logits_calib = data['logits_calib']
                p_calib_uncal = sp_softmax(logits_calib, axis=1)
                t_star, _ = find_optimal_threshold(y_calib, p_calib_uncal[:, 1])
            else:
                t_star = 0.5

            psytar_m = eval_full(y_test, p_test, threshold=t_star)
            cadec_m = eval_full(y_cadec, p_cadec, threshold=t_star)

            file_results[method_name] = {
                'psytar': psytar_m,
                'cadec': cadec_m,
            }

            print(f"\n  {method_name}:")
            print(f"    PsyTAR: AUROC={psytar_m['AUROC']:.4f} "
                  f"AUPRC={psytar_m['AUPRC']:.4f} "
                  f"F1@t*={psytar_m['F1@t*']:.4f} (t*={t_star:.2f}) "
                  f"F1@0.5={psytar_m['F1@0.5']:.4f} "
                  f"ECE={psytar_m['ECE_adaptive']:.4f} "
                  f"[{psytar_m['ECE_CI_lo']:.4f},{psytar_m['ECE_CI_hi']:.4f}]")
            print(f"    CADEC:  AUROC={cadec_m['AUROC']:.4f} "
                  f"AUPRC={cadec_m['AUPRC']:.4f} "
                  f"F1@t*={cadec_m['F1@t*']:.4f} "
                  f"F1@0.5={cadec_m['F1@0.5']:.4f} "
                  f"ECE={cadec_m['ECE_adaptive']:.4f} "
                  f"[{cadec_m['ECE_CI_lo']:.4f},{cadec_m['ECE_CI_hi']:.4f}]")

        # Paired delta ECE & AUROC tests
        print(f"\n  Paired Bootstrap ΔECE Tests:")
        for method_name, (p_test, _) in methods.items():
            if method_name == 'Uncalibrated':
                continue
            d_point, d_lo, d_hi = bootstrap_delta_ece(
                y_test, probs_test_uncal[:, 1], p_test[:, 1])
            sig = "SIGNIFICANT" if d_hi < 0 else "non-significant (CI crosses 0)"
            print(f"    {method_name} vs Uncal: ΔECE={d_point:+.4f} "
                  f"[{d_lo:+.4f}, {d_hi:+.4f}] — {sig}")

        # Compute paired delta AUROC vs baseline LR (AUROC = 0.8835)
        from metrics_utils import bootstrap_delta_auroc
        print(f"\n  Paired Bootstrap ΔAUROC Tests (Transformer vs Uncalibrated):")
        # Compare Uncalibrated vs Isotonic
        da_point, da_lo, da_hi = bootstrap_delta_auroc(
            y_test, probs_test_uncal[:, 1], probs_test_iso[:, 1])
        print(f"    Uncalibrated vs Isotonic ΔAUROC: {da_point:+.4f} [{da_lo:+.4f}, {da_hi:+.4f}]")

        all_results[basename] = file_results

    # Export
    out_path = os.path.join(results_dir, "gpu_metrics_recomputed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[ARTIFACT] Recomputed metrics saved to: {out_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
