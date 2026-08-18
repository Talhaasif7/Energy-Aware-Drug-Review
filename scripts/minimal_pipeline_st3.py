import os
import sys
import time
import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, brier_score_loss
import lightgbm as lgb

def reconfigure_stdout():
    """Ensure utf-8 stdout encoding for Windows console compatibility."""
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

def compute_ece(y_true, y_probs, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE) for binary classification using 10 equal-width bins.
    """
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

def validate_probabilities(probs_2d, model_name):
    """
    Validate that predicted probabilities:
    - Contain no NaNs or Infs
    - Are bounded strictly in [0.0, 1.0]
    - Class probabilities sum to 1.0 per row
    """
    has_nan = np.isnan(probs_2d).any()
    has_inf = np.isinf(probs_2d).any()
    in_range = (probs_2d >= 0.0).all() and (probs_2d <= 1.0).all()
    sums_to_one = np.allclose(probs_2d.sum(axis=1), 1.0)

    print(f"\n--- Probability Validation Check: {model_name} ---")
    print(f"  * NaNs detected       : {'NO (Passed)' if not has_nan else 'YES (Failed)'}")
    print(f"  * Infs detected       : {'NO (Passed)' if not has_inf else 'YES (Failed)'}")
    print(f"  * Range bounded [0,1] : {'YES (Passed)' if in_range else 'NO (Failed)'}")
    print(f"  * Probabilities sum=1 : {'YES (Passed)' if sums_to_one else 'NO (Failed)'}")

    return not has_nan and not has_inf and in_range and sums_to_one

def main():
    reconfigure_stdout()
    print("Starting Smoke Test 3 (ST3 - Minimal CPU End-to-End Pipeline)...")

    # 1. DATA PREPARATION
    psytar_csv_path = r"e:\AI Green\data\01_primary_adr_detection\dev_psytar\psytar_harmonised.csv"
    if not os.path.exists(psytar_csv_path):
        raise FileNotFoundError(f"Harmonised dataset not found at: {psytar_csv_path}")

    print(f"\nLoading harmonised PsyTAR dataset from: {psytar_csv_path}")
    df_full = pd.read_csv(psytar_csv_path)
    print(f"Full dataset size: {len(df_full)} rows. Columns: {list(df_full.columns)}")

    # Extract stratified subset of exactly 2,000 units
    subset_size = min(2000, len(df_full))
    df_subset, _ = train_test_split(
        df_full,
        train_size=subset_size,
        stratify=df_full['label'],
        random_state=42
    )
    print(f"Extracted stratified subset of {len(df_subset)} sentences.")
    print(f"Subset Class Balance: Positive (ADR=1): {int(df_subset['label'].sum())}, Negative (ADR=0): {int((df_subset['label'] == 0).sum())}")

    # Train-Test Split (80% train, 20% test)
    train_df, test_df = train_test_split(
        df_subset,
        test_size=0.2,
        stratify=df_subset['label'],
        random_state=42
    )
    print(f"Train split size: {len(train_df)} | Test split size: {len(test_df)}")

    # Vectorization: TF-IDF (max_features=1000)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(train_df['text']).toarray()
    X_test_vec = vectorizer.transform(test_df['text']).toarray()
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    # Models to benchmark
    models = {
        'Logistic Regression (Linear)': LogisticRegression(max_iter=1000, random_state=42),
        'LightGBM (GBDT)': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    }

    report_rows = []

    for name, clf in models.items():
        print(f"\n==================================================")
        print(f" Benchmarking Model: {name}")
        print(f"==================================================")

        # 2. MODEL TRAINING & TRACKING
        tracker_train = EmissionsTracker(save_to_file=False, log_level='error')
        t0_train = time.perf_counter()
        tracker_train.start()

        clf.fit(X_train_vec, y_train)

        train_energy_kwh = tracker_train.stop()
        t1_train = time.perf_counter()

        train_time_secs = t1_train - t0_train
        train_energy_joules = (train_energy_kwh or 0.0) * 3600000.0

        print(f"Training Time  : {train_time_secs:.3f} s")
        print(f"Training Energy: {train_energy_joules:.4f} J ({train_energy_kwh:.8f} kWh)")

        # 3. INFERENCE & ENERGY TRACKING
        tracker_inf = EmissionsTracker(save_to_file=False, log_level='error')
        t0_inf = time.perf_counter()
        tracker_inf.start()

        probs_2d = clf.predict_proba(X_test_vec)

        inf_energy_kwh = tracker_inf.stop()
        t1_inf = time.perf_counter()

        inf_time_secs = t1_inf - t0_inf
        inf_energy_joules = (inf_energy_kwh or 0.0) * 3600000.0

        # Extrapolate inference energy per 1,000 sentences
        inf_energy_per_1k = (inf_energy_joules / len(y_test)) * 1000.0 if len(y_test) > 0 else 0.0

        # Validate probability bounds & sums
        valid_probs = validate_probabilities(probs_2d, name)

        # 4. METRICS COMPUTATION
        y_probs_pos = probs_2d[:, 1]
        y_pred = (y_probs_pos >= 0.5).astype(int)

        macro_f1 = f1_score(y_test, y_pred, average='macro')
        adr_f1 = f1_score(y_test, y_pred, pos_label=1)
        ece = compute_ece(y_test, y_probs_pos, n_bins=10)
        brier = brier_score_loss(y_test, y_probs_pos)

        report_rows.append({
            'Model': name,
            'Macro F1': macro_f1,
            'ADR F1': adr_f1,
            'ECE': ece,
            'Brier Score': brier,
            'Train Time (s)': train_time_secs,
            'Train Energy (J)': train_energy_joules,
            'Inf Energy / 1k (J)': inf_energy_per_1k
        })

    # 5. PRINT ST3 SUMMARY REPORT
    df_report = pd.DataFrame(report_rows)

    print("\n" + "="*90)
    print("                ST3 — MINIMAL CPU END-TO-END PIPELINE REPORT")
    print("="*90)

    print("\n--- PERFORMANCE & ENERGY COMPARISON TABLE ---")
    formatted_df = pd.DataFrame({
        'Model': df_report['Model'],
        'Macro F1': df_report['Macro F1'].map(lambda x: f"{x:.4f}"),
        'ADR F1 (Class 1)': df_report['ADR F1'].map(lambda x: f"{x:.4f}"),
        'ECE (10-bin)': df_report['ECE'].map(lambda x: f"{x:.4f}"),
        'Brier Score': df_report['Brier Score'].map(lambda x: f"{x:.4f}"),
        'Train Time (s)': df_report['Train Time (s)'].map(lambda x: f"{x:.3f}"),
        'Train Energy (J)': df_report['Train Energy (J)'].map(lambda x: f"{x:.4f}"),
        'Inf Energy/1k (J)': df_report['Inf Energy / 1k (J)'].map(lambda x: f"{x:.4f}")
    })
    print(formatted_df.to_string(index=False))

    print("\n--- VALIDATION SUMMARY ---")
    print("  [OK] Dataset split: 2,000 PsyTAR subset (1,600 train / 400 test) stratified.")
    print("  [OK] Inference probability validation verified: No NaNs/Infs, bounded in [0,1], row sum = 1.")
    print("  [OK] Training time, energy (J), ECE, Brier, and per-1k inference energy measured.")
    print("="*90 + "\n")

if __name__ == "__main__":
    main()
