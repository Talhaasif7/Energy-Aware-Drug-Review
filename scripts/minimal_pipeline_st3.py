"""
ST3 — Minimal End-to-End CPU Pipeline (Corrected)

Fixes applied per mentor review:
  - AUROC and AUPRC added as threshold-invariant discrimination metrics
  - F1 reported at fixed 0.5 (labelled) — no calib split here to tune threshold
  - Inference energy AMORTISED over 100x loop to exceed sensor resolution
  - Load wattage reported alongside Joules
  - Explicit note: energy values are GROSS (not idle-subtracted)

Round 5 additions (2026-08-24)
------------------------------
  - PORTABILITY: the PsyTAR path was a hardcoded Windows absolute path
    (r"e:\\AI Green\\...") which made this script unrunnable anywhere else. It is
    now derived from the repository root, so the script runs on any host.
  - TRACEABILITY: this stage used to be print-only. Its training time and energy
    were transcribed by hand into `budget_and_subgroup_st6_st7.py` as the literal
    constants 6.041/1600 s and 3.2038/1600 J — exactly the untraceable-number
    pattern the Round 5 review objected to. It now writes
    `results/st3_cpu_energy.json`, including `n_train` so the per-sample rates
    are *derived* rather than reproduced by hand.
  - HONEST ENERGY PROVENANCE: CodeCarbon reports a real power reading only where
    it can reach a sensor. On Windows without Intel Power Gadget it falls back to
    a TDP-fraction software model, which is an estimate, not a measurement. This
    script now *also* integrates Intel RAPL package energy directly when the host
    exposes it (Linux, bare metal, readable sysfs) and records both numbers plus
    full host provenance, so a reviewer can see which is which.
"""
import os
import sys
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

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
from rapl_utils import RAPLReader, probe_environment

try:
    from codecarbon import EmissionsTracker
except Exception as _cc_exc:                                  # pragma: no cover
    EmissionsTracker = None
    _CODECARBON_IMPORT_ERROR = repr(_cc_exc)
else:
    _CODECARBON_IMPORT_ERROR = None

INFERENCE_AMORTISATION_LOOPS = 100

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
PSYTAR_CSV = os.path.join(DATA_DIR, "01_primary_adr_detection", "dev_psytar",
                          "psytar_harmonised.csv")
OUT_JSON = os.path.join(RESULTS_DIR, "st3_cpu_energy.json")

# Documented Linux RAPL benchmark host idle power.
ST2_IDLE_W = 8.650


def make_tracker():
    """Construct an EmissionsTracker across CodeCarbon versions, or None.

    `allow_multiple_runs` only exists in newer releases; passing it to an older
    one raises TypeError. ST3 starts several trackers in one process, so request
    it when supported and degrade quietly when not. Returns None if CodeCarbon is
    unusable — energy is then reported from RAPL alone rather than as a zero.
    """
    if EmissionsTracker is None:
        return None
    base = {"save_to_file": False, "log_level": "error"}
    for extra in ({"allow_multiple_runs": True}, {}):
        try:
            return EmissionsTracker(**base, **extra)
        except TypeError:
            continue
        except Exception:
            return None
    return None


def cc_measure(fn):
    """Run `fn()` timed, with CodeCarbon and (where available) RAPL in parallel.

    Returns (result, dict). Neither energy source is allowed to abort the run or
    to report 0.0 J when it actually failed — a failed source yields None, which
    the JSON preserves so a reviewer can see the gap.
    """
    rapl = RAPLReader()
    tracker = make_tracker()

    cc_started = False
    if tracker is not None:
        try:
            tracker.start()
            cc_started = True
        except Exception:
            cc_started = False

    r_before = rapl.read()
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    r_after = rapl.read()

    cc_kwh = None
    if cc_started:
        try:
            cc_kwh = tracker.stop()
        except Exception:
            cc_kwh = None

    secs = t1 - t0
    cc_j = (cc_kwh * 3_600_000.0) if cc_kwh is not None else None
    rapl_j = rapl.delta_j(r_before, r_after)

    # Preferred figure: RAPL when the host really has it, else CodeCarbon.
    if rapl_j is not None:
        energy_j, source = rapl_j, "measured_rapl_integrated"
    elif cc_j is not None:
        energy_j, source = cc_j, "codecarbon_reported"
    else:
        energy_j, source = None, "unavailable"

    return out, {
        "seconds": secs,
        "energy_j": energy_j,
        "energy_source": source,
        "energy_j_rapl": rapl_j,
        "energy_j_codecarbon": cc_j,
        "watts": (energy_j / secs) if (energy_j is not None and secs > 0) else None,
        "rapl_available": rapl.ok,
        "rapl_reason": rapl.reason,
    }


def _json_safe(obj):
    """Recursively convert to strictly-JSON-serialisable values.

    numpy scalars become Python scalars, and NaN/inf become null. Written so the
    artifact is valid strict JSON: `json` would otherwise emit bare `NaN`, which
    most parsers reject, and numpy scalars raise TypeError outright. Order
    matters — bool is a subclass of int, so it must be tested first.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if (f == f and f not in (float("inf"), float("-inf"))) else None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return str(obj)


def _f(v, nd=4, suffix=""):
    """Format a possibly-missing number. Prints 'n/a' rather than a fake 0.0.

    Energy can legitimately be unavailable (no RAPL, no CodeCarbon), and printing
    0.0000 J there would read as 'measured zero consumption'.
    """
    if v is None:
        return "n/a"
    try:
        if isinstance(v, float) and (v != v):     # NaN
            return "n/a"
        return f"{v:.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


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

    env = probe_environment()
    print(f"\n--- HOST ---")
    print(f"  OS         : {env['platform']}")
    print(f"  CPU        : {env['cpu_model']} ({env['logical_cpus']} logical)")
    print(f"  RAPL       : {'AVAILABLE' if env['rapl_available'] else 'NOT AVAILABLE'}"
          f" — {env['rapl_reason']}")
    if env["is_wsl"]:
        print("  [warn] WSL detected: powercap counters are not passed through, so "
              "energy here is a software estimate, not a measurement.")
    if env["is_container"]:
        print("  [warn] container detected: energy counters may be absent or shared "
              "with the host.")
    if EmissionsTracker is None:
        print(f"  [warn] CodeCarbon unavailable ({_CODECARBON_IMPORT_ERROR}); "
              "relying on RAPL where present.")
    if not env["rapl_available"] and EmissionsTracker is None:
        print("  [warn] no energy source at all — timings will still be valid, "
              "energy fields will be null.")

    psytar_csv_path = PSYTAR_CSV
    if not os.path.exists(psytar_csv_path):
        raise FileNotFoundError(
            f"Not found: {psytar_csv_path}\n"
            "Expected the harmonised PsyTAR CSV at "
            "data/01_primary_adr_detection/dev_psytar/psytar_harmonised.csv "
            f"relative to the repository root ({ROOT}). Run this script from a "
            "complete checkout, or regenerate the CSV with scripts/harmonise_st1.py.")

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
    energy_records = {}

    for name, clf in models.items():
        print(f"\n{'=' * 60}")
        print(f" Benchmarking Model: {name}")
        print(f"{'=' * 60}")

        # --- Training energy ---
        _, tr = cc_measure(lambda: clf.fit(X_train, y_train))
        train_secs = tr["seconds"]
        train_j = tr["energy_j"]
        train_watts = tr["watts"]

        print(f"  Training: {train_secs:.3f}s | "
              f"{'%.4f J' % train_j if train_j is not None else 'n/a'} | "
              f"{'%.2f W avg load' % train_watts if train_watts is not None else 'n/a'}"
              f"  [{tr['energy_source']}]")

        test_raw_texts = list(test_df['text'])
        def _infer():
            out = None
            for _ in range(INFERENCE_AMORTISATION_LOOPS):
                X_t = vectorizer.transform(test_raw_texts)
                out = clf.predict_proba(X_t)
            return out

        probs_2d, inf = cc_measure(_infer)
        total_inf_secs = inf["seconds"]
        total_inf_j = inf["energy_j"]
        inf_watts = inf["watts"]

        # Per-pass values
        single_inf_j = (total_inf_j / INFERENCE_AMORTISATION_LOOPS
                        if total_inf_j is not None else None)
        single_inf_secs = total_inf_secs / INFERENCE_AMORTISATION_LOOPS
        inf_energy_per_1k = ((single_inf_j / len(y_test)) * 1000.0
                             if single_inf_j is not None else None)

        print(f"  Inference ({INFERENCE_AMORTISATION_LOOPS}x amortised): "
              f"{total_inf_secs:.3f}s total | {_f(total_inf_j, 4, ' J')} total | "
              f"{_f(inf_watts, 2, ' W')} avg load  [{inf['energy_source']}]")
        print(f"  Inference (per-pass): {single_inf_secs:.4f}s | "
              f"{_f(single_inf_j, 6, ' J')} | "
              f"{_f(inf_energy_per_1k, 6, ' J/1k sentences')}")

        validate_probabilities(probs_2d, name)

        # --- Metrics (threshold-invariant + fixed 0.5) ---
        y_probs = probs_2d[:, 1]
        metrics = compute_full_metrics(y_test, y_probs, threshold=0.5)

        # Persist everything a downstream stage could need, including the
        # denominators, so per-sample rates are DERIVED and never transcribed.
        energy_records[name] = {
            "n_train": int(X_train.shape[0]),
            "n_test": int(len(y_test)),
            "n_features": int(X_train.shape[1]),
            "train_seconds": train_secs,
            "train_energy_j": train_j,
            "train_load_w": train_watts,
            "train_s_per_sample": (train_secs / X_train.shape[0]
                                   if X_train.shape[0] else None),
            "train_j_per_sample": (train_j / X_train.shape[0]
                                   if (train_j is not None and X_train.shape[0])
                                   else None),
            "train_energy_source": tr["energy_source"],
            "train_energy_j_rapl": tr["energy_j_rapl"],
            "train_energy_j_codecarbon": tr["energy_j_codecarbon"],
            "inference_loops": INFERENCE_AMORTISATION_LOOPS,
            "inference_total_seconds": total_inf_secs,
            "inference_total_energy_j": total_inf_j,
            "inference_per_pass_seconds": single_inf_secs,
            "inference_per_pass_energy_j": single_inf_j,
            "inference_j_per_1k": inf_energy_per_1k,
            "inference_s_per_sample": (single_inf_secs / len(y_test)
                                       if len(y_test) else None),
            "inference_load_w": inf_watts,
            "inference_energy_source": inf["energy_source"],
            "inference_energy_j_rapl": inf["energy_j_rapl"],
            "inference_energy_j_codecarbon": inf["energy_j_codecarbon"],
            "metrics": metrics,   # sanitised by _json_safe() at write time
        }

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
        'Train Time (s)': df_report['Train Time (s)'].map(lambda x: _f(x, 3)),
        'Train Energy (J)': df_report['Train Energy (J)'].map(lambda x: _f(x, 4)),
        'Train Load (W)': df_report['Train Load (W)'].map(lambda x: _f(x, 2)),
        'Inf Energy/1k (J)': df_report['Inf Energy/1k (J)'].map(lambda x: _f(x, 6)),
        'Inf Load (W)': df_report['Inf Load (W)'].map(lambda x: _f(x, 2)),
    })
    print(en_df.to_string(index=False))

    print("\n--- NOTES ---")
    print(f"  * Inference energy amortised over {INFERENCE_AMORTISATION_LOOPS}x "
          f"loops to exceed RAPL/CodeCarbon polling resolution.")
    print(f"  * Energy values are GROSS (not idle-subtracted). "
          f"Compare load wattage to ST2 package idle baseline "
          f"({ST2_IDLE_W:.3f} W).")
    print(f"  * No calibration split in ST3; F1 reported at fixed 0.5. "
          f"Threshold-tuned F1 is computed in ST4/ST5 with proper calib split.")
    if env["rapl_available"]:
        print("  * Energy above is INTEGRATED Intel RAPL package energy — a real "
              "sensor reading on this host.")
    else:
        print("  * Energy above is NOT a sensor reading on this host "
              f"({env['rapl_reason']}). Where CodeCarbon supplied it, treat it as "
              "a software model (TDP fraction), i.e. an estimate.")
    print("=" * 100 + "\n")

    # --- Persist the artifact (traceability; see module docstring) ---
    payload = {
        "_meta": {
            "generated_by": "minimal_pipeline_st3.py",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "ST3_minimal_cpu_pipeline",
            "psytar_csv": os.path.relpath(psytar_csv_path, ROOT).replace("\\", "/"),
            "psytar_rows_full": int(len(df_full)),
            "subset_size": int(subset_size),
            "n_train": int(X_train.shape[0]),
            "n_test": int(len(y_test)),
            "seed": 42,
            "tfidf_max_features": 1000,
            "inference_amortisation_loops": INFERENCE_AMORTISATION_LOOPS,
            "energy_is_gross": True,
            "st2_idle_power_w_documented": ST2_IDLE_W,
            "codecarbon_available": EmissionsTracker is not None,
            "codecarbon_import_error": _CODECARBON_IMPORT_ERROR,
            "host": env,
            "note": ("Per-sample rates must be DERIVED from the *_seconds / *_energy_j "
                     "fields divided by n_train, never transcribed. ST6 previously "
                     "hardcoded 6.041/1600 s and 3.2038/1600 J from a console log; "
                     "this artifact exists so that cannot recur."),
        },
        "models": energy_records,
    }

    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        # Serialise fully before opening, so a failure mid-encode cannot leave a
        # truncated file that still looks like a valid artifact.
        blob = json.dumps(_json_safe(payload), indent=2, allow_nan=False)
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            fh.write(blob)
        print(f"[artifact] wrote {OUT_JSON}")
    except Exception as exc:
        print(f"[ERROR] could not write {OUT_JSON}: {exc!r}")
        print("[ERROR] the console table above is then the only record — "
              "capture it before closing this terminal.")


if __name__ == "__main__":
    main()
