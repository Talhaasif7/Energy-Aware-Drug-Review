"""
ST3 — Minimal End-to-End CPU Pipeline (Corrected)

Fixes applied per mentor review:
  - AUROC and AUPRC added as threshold-invariant discrimination metrics
  - F1 reported at fixed 0.5 (labelled) — no calib split here to tune threshold
  - Inference energy AMORTISED over 100x loop to exceed sensor resolution
  - Load wattage reported alongside Joules
  - Explicit note: energy values are GROSS (not idle-subtracted)
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

# Import shared metrics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_utils import (
    compute_ece_adaptive, compute_ece_equal_width,
    bootstrap_ece, compute_full_metrics
)

INFERENCE_AMORTISATION_LOOPS = 100


def reconfigure_stdout():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


def validate_probabilities(probs_2d, model_name):
    has_nan = np.isnan(probs_2d).any()
    has_inf = np.isinf(probs_2d).any()
    in_range = (probs_2d >= 0.0).all() and (probs_2d <= 1.0).all()
    sums_to_one = np.allclose(probs_2d.sum(axis=1), 1.0)

    print(f"\n--- Probability Validation Check: {model_name} ---")
    print(f"  * NaNs detected       : {'NO (OK)' if not has_nan else 'YES (FAIL)'}")
    print(f"  * Infs detected       : {'NO (OK)' if not has_inf else 'YES (FAIL)'}")
    print(f"  * Range bounded [0,1] : {'YES (OK)' if in_range else 'NO (FAIL)'}")
    print(f"  * Probabilities sum=1 : {'YES (OK)' if sums_to_one else 'NO (FAIL)'}")
    return not has_nan and not has_inf and in_range and sums_to_one


def main():
    reconfigure_stdout()
    print("Starting Smoke Test 3 (ST3 - Minimal CPU End-to-End Pipeline) [CORRECTED]")
    print(f"  Inference amortisation: {INFERENCE_AMORTISATION_LOOPS}x loop")
    print(f"  Energy values: GROSS (not idle-subtracted)")

    psytar_csv_path = r"e:\AI Green\data\01_primary_adr_detection\dev_psytar\psytar_harmonised.csv"
    if not os.path.exists(psytar_csv_path):
        raise FileNotFoundError(f"Not found: {psytar_csv_path}")

    print(f"\nLoading harmonised PsyTAR from: {psytar_csv_path}")
    df_full = pd.read_csv(psytar_csv_path)
    print(f"Full dataset: {len(df_full)} rows. Columns: {list(df_full.columns)}")

    subset_size = min(2000, len(df_full))
    df_subset, _ = train_test_split(
        df_full, train_size=subset_size,
        stratify=df_full['label'], random_state=42)
    print(f"Stratified subset: {len(df_subset)} sentences "
          f"(ADR=1: {int(df_subset['label'].sum())}, "
          f"ADR=0: {int((df_subset['label'] == 0).sum())})")

    train_df, test_df = train_test_split(
        df_subset, test_size=0.2,
        stratify=df_subset['label'], random_state=42)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(train_df['text']).toarray()
    X_test = vectorizer.transform(test_df['text']).toarray()
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    models = {
        'Logistic Regression (Linear)': LogisticRegression(
            max_iter=1000, random_state=42),
        'LightGBM (GBDT)': lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, num_leaves=31,
            random_state=42, n_jobs=-1, verbose=-1)
    }

    report_rows = []

    for name, clf in models.items():
        print(f"\n{'=' * 60}")
        print(f" Benchmarking Model: {name}")
        print(f"{'=' * 60}")

        # --- Training energy ---
        tracker_train = EmissionsTracker(save_to_file=False, log_level='error')
        t0 = time.perf_counter()
        tracker_train.start()
        clf.fit(X_train, y_train)
        train_kwh = tracker_train.stop()
        t1 = time.perf_counter()

        train_secs = t1 - t0
        train_j = (train_kwh or 0.0) * 3_600_000.0
        train_watts = train_j / train_secs if train_secs > 0 else 0.0

        print(f"  Training: {train_secs:.3f}s | {train_j:.4f} J | "
              f"{train_watts:.2f} W avg load")

        # --- Inference energy (AMORTISED over 100x) ---
        tracker_inf = EmissionsTracker(save_to_file=False, log_level='error')
        t0 = time.perf_counter()
        tracker_inf.start()
        for _ in range(INFERENCE_AMORTISATION_LOOPS):
            probs_2d = clf.predict_proba(X_test)
        inf_kwh = tracker_inf.stop()
        t1 = time.perf_counter()

        total_inf_secs = t1 - t0
        total_inf_j = (inf_kwh or 0.0) * 3_600_000.0
        inf_watts = total_inf_j / total_inf_secs if total_inf_secs > 0 else 0.0

        # Per-pass values
        single_inf_j = total_inf_j / INFERENCE_AMORTISATION_LOOPS
        single_inf_secs = total_inf_secs / INFERENCE_AMORTISATION_LOOPS
        inf_energy_per_1k = (single_inf_j / len(y_test)) * 1000.0

        print(f"  Inference ({INFERENCE_AMORTISATION_LOOPS}x amortised): "
              f"{total_inf_secs:.3f}s total | {total_inf_j:.4f} J total | "
              f"{inf_watts:.2f} W avg load")
        print(f"  Inference (per-pass): {single_inf_secs:.4f}s | "
              f"{single_inf_j:.6f} J | {inf_energy_per_1k:.6f} J/1k sentences")

        validate_probabilities(probs_2d, name)

        # --- Metrics (threshold-invariant + fixed 0.5) ---
        y_probs = probs_2d[:, 1]
        metrics = compute_full_metrics(y_test, y_probs, threshold=0.5)

        report_rows.append({
            'Model': name,
            'AUROC': metrics['AUROC'],
            'AUPRC': metrics['AUPRC'],
            'ADR F1@0.5': metrics['F1@0.5'],
            'Macro F1@0.5': metrics['Macro_F1@0.5'],
            'ECE (adaptive)': metrics['ECE_adaptive'],
            'ECE 95% CI': f"[{metrics['ECE_CI_lo']:.4f}, {metrics['ECE_CI_hi']:.4f}]",
            'ECE (equal-width)': metrics['ECE_EW'],
            'Brier': metrics['Brier'],
            'NLL': metrics['NLL'],
            'Train Time (s)': train_secs,
            'Train Energy (J)': train_j,
            'Train Load (W)': train_watts,
            'Inf Energy/1k (J)': inf_energy_per_1k,
            'Inf Load (W)': inf_watts,
        })

    # --- Report ---
    df_report = pd.DataFrame(report_rows)

    print("\n" + "=" * 100)
    print("            ST3 — MINIMAL CPU END-TO-END PIPELINE REPORT (CORRECTED)")
    print("=" * 100)

    print("\n--- DISCRIMINATION METRICS (Threshold-Invariant) ---")
    disc_df = pd.DataFrame({
        'Model': df_report['Model'],
        'AUROC': df_report['AUROC'].map(lambda x: f"{x:.4f}"),
        'AUPRC': df_report['AUPRC'].map(lambda x: f"{x:.4f}"),
        'ADR F1@0.5': df_report['ADR F1@0.5'].map(lambda x: f"{x:.4f}"),
        'Macro F1@0.5': df_report['Macro F1@0.5'].map(lambda x: f"{x:.4f}"),
    })
    print(disc_df.to_string(index=False))

    print("\n--- CALIBRATION METRICS ---")
    cal_df = pd.DataFrame({
        'Model': df_report['Model'],
        'ECE (adaptive)': df_report['ECE (adaptive)'].map(lambda x: f"{x:.4f}"),
        'ECE 95% CI': df_report['ECE 95% CI'],
        'ECE (equal-width)': df_report['ECE (equal-width)'].map(lambda x: f"{x:.4f}"),
        'Brier Score': df_report['Brier'].map(lambda x: f"{x:.4f}"),
        'NLL': df_report['NLL'].map(lambda x: f"{x:.4f}"),
    })
    print(cal_df.to_string(index=False))

    print("\n--- ENERGY METRICS (Gross, Not Idle-Subtracted) ---")
    en_df = pd.DataFrame({
        'Model': df_report['Model'],
        'Train Time (s)': df_report['Train Time (s)'].map(lambda x: f"{x:.3f}"),
        'Train Energy (J)': df_report['Train Energy (J)'].map(lambda x: f"{x:.4f}"),
        'Train Load (W)': df_report['Train Load (W)'].map(lambda x: f"{x:.2f}"),
        'Inf Energy/1k (J)': df_report['Inf Energy/1k (J)'].map(lambda x: f"{x:.6f}"),
        'Inf Load (W)': df_report['Inf Load (W)'].map(lambda x: f"{x:.2f}"),
    })
    print(en_df.to_string(index=False))

    print("\n--- NOTES ---")
    print(f"  * Inference energy amortised over {INFERENCE_AMORTISATION_LOOPS}x "
          f"loops to exceed RAPL/CodeCarbon polling resolution.")
    print(f"  * Energy values are GROSS (not idle-subtracted). "
          f"Compare load wattage to ST2 idle baseline (0.093 W).")
    print(f"  * No calibration split in ST3; F1 reported at fixed 0.5. "
          f"Threshold-tuned F1 is computed in ST4/ST5 with proper calib split.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
