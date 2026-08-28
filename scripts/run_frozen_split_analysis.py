#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_frozen_split_analysis.py  —  Frozen-split classical arms + paired bootstrap
tie rule + ECC-MS reconciliation (Round 5 rigorous overhaul).

WHAT THIS DOES (all on CPU, no GPU needed)
------------------------------------------
1. Loads the transformer prediction artifacts results/*_predictions.npz.
2. Recovers the EXACT frozen split the transformers were evaluated on:
     - if the .npz carries train/calib/test texts (new GPU-script version) it
       uses them verbatim  -> alignment is guaranteed by construction;
     - otherwise it reconstructs the split from psytar_harmonised.csv with the
       transformer's own split logic and ASSERTS the reconstructed y_test matches
       the .npz y_test element-for-element (aborts the paired bootstrap if not).
3. Trains the classical arms (LogReg + LightGBM on TF-IDF max_features=1000) on
   the SAME train split, fits Temp + Isotonic on the SAME calib split, and
   predicts on the SAME PsyTAR test set AND full CADEC. Classical + transformer
   predictions therefore live on one shared, identical test set.
4. Recomputes every arm's discrimination + calibration metrics from raw arrays.
5. Runs the REAL statistical-tie rule: paired bootstrap of Delta_AUROC on shared
   resamples; a tie is declared when the 95% CI includes zero (see
   eccms_selection.py). Also emits a fixed-margin sensitivity strip @0.01/0.02/0.03.
6. Reconciles ECC-MS selection + feasible-arm counts over the (tau, E) grid using
   the FULL 12-arm catalogue (4 models x 3 recal), pulling energy from the live
   CPU + Colab-GPU measurement JSONs.
7. Adds the RQ4 column: does the ECC-MS-selected arm still satisfy tau on CADEC?
8. Writes results/frozen_split_reconciled.json — the single source of truth the
   README is reconciled against. Nothing here is hand-entered.

USAGE
-----
    python scripts/run_frozen_split_analysis.py

Requires: results/<model>_seed<N>_predictions.npz (from the Colab GPU run) and
the harmonised CSVs under data/. Uses results/colab_transformer_gpu_results.json
and results/cpu_energy_measured.json for energy if present (else clearly-labelled
fallback constants).
"""
import os
import sys
import glob
import json
import hashlib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.special import softmax
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_utils import (
    TemperatureScaler, compute_full_metrics, find_optimal_threshold,
    compute_ece_adaptive,
)
from eccms_selection import (
    eccms_select_argmax, eccms_select_fixed_margin, eccms_select_bootstrap_tie,
    pairwise_delta_auroc_matrix, feasible_arms, get_energy, get_auroc,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")

PSYTAR_CSV = os.path.join(DATA_DIR, "01_primary_adr_detection", "dev_psytar",
                          "psytar_harmonised.csv")
CADEC_CSV = os.path.join(DATA_DIR, "01_primary_adr_detection", "external_val_cadec",
                         "cadec_harmonised.csv")
DRUGLIB_CSV = os.path.join(DATA_DIR, "02_secondary_sentiment_scaling",
                           "dev_uci_drug_review", "uci_druglib_harmonised.csv")

# npz filename stem -> canonical display model name
NPZ_MODEL_MAP = {
    "efficient_transformer": "DistilBERT",
    "biomedical_transformer": "PubMedBERT",
}
# Colab JSON keys -> canonical display model name
JSON_MODEL_MAP = {
    "Efficient Transformer": "DistilBERT",
    "Biomedical Transformer": "PubMedBERT",
}

# Transformer split reconstruction constants (mirror the GPU script exactly).
SUBSET_SIZE_SMOKE = 2000
SUBSET_RANDOM_STATE = 42

# Fallback CPU energy (documented ST2/ST3 values) if no measured JSON is present.
CPU_ENERGY_FALLBACK = {
    "Logistic Regression": {"inf_j_gross": 0.4400, "inf_j_net": 0.0210,
                            "throughput_sps": 16070.0, "load_w": 7.072,
                            "idle_w": 6.734, "provenance": "fallback_constant_ST2_ST3"},
    "LightGBM":            {"inf_j_gross": 0.7412, "inf_j_net": 0.2391,
                            "throughput_sps": 13410.0, "load_w": 9.940,
                            "idle_w": 6.734, "provenance": "fallback_constant_ST2_ST3"},
}

TAU_GRID = [0.03, 0.05, 0.07, 0.10]
E_GRID = [0.5, 10.0, 60.0, 120.0, 150.0, 200.0]
# Full README-facing grid (for feasible-count reconciliation vs the reviewer).
RECONCILE_CELLS = [(0.05, 60.0), (0.07, 10.0), (0.07, 60.0),
                   (0.10, 0.5), (0.10, 10.0), (0.10, 60.0),
                   (0.05, 120.0), (0.07, 120.0), (0.10, 120.0)]
MARGINS = [0.01, 0.02, 0.03]
N_BOOTSTRAP = 2000
BOOT_SEED = 42

MODEL_HYPERPARAMETERS = {
    "Logistic Regression": {
        "solver": "lbfgs", "max_iter": 1000, "C": 1.0, "penalty": "l2",
        "vectorizer": "TfidfVectorizer(max_features=1000)", "random_state": 42
    },
    "LightGBM": {
        "n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31,
        "vectorizer": "TfidfVectorizer(max_features=1000)", "random_state": 42,
        "n_jobs": -1
    },
    "DistilBERT": {
        "hf_model": "distilbert-base-uncased", "learning_rate": 2e-5,
        "batch_size": 64, "epochs": 3, "max_seq_length": 128, "optimizer": "AdamW",
        "seeds_trained": [42, 123, 456]
    },
    "PubMedBERT": {
        "hf_model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "learning_rate": 2e-5, "batch_size": 64, "epochs": 3,
        "max_seq_length": 128, "optimizer": "AdamW", "seeds_trained": [42, 123, 456]
    }
}


def log(msg=""):
    print(msg, flush=True)


def csv_sha(path):
    """Short SHA-256 of a data file, for provenance stamping in the output JSON."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Energy loading
# ---------------------------------------------------------------------------
def load_gpu_energy(json_path):
    """Return {display_model: {gross, net, throughput, load_w, cv, T, nll_pre,
    nll_post, provenance}} averaged across seeds, tolerant to smoke vs saturated
    key names."""
    out = {}
    if not os.path.exists(json_path):
        log(f"[energy] GPU results JSON not found: {json_path} — GPU energy PENDING.")
        return out
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    smoke = bool(data.get("smoke_test_mode", False))
    results = data.get("results", {})
    for json_key, seeds_res in results.items():
        disp = JSON_MODEL_MAP.get(json_key, json_key)
        if not seeds_res:
            continue

        def avg(*keys):
            vals = []
            for r in seeds_res:
                for k in keys:
                    if k in r and r[k] is not None:
                        vals.append(float(r[k]))
                        break
            return float(np.mean(vals)) if vals else None

        gross = avg("saturated_gross_energy_1k_j", "inf_gross_energy_1k_j")
        net = avg("saturated_net_energy_1k_j", "inf_net_energy_1k_j")
        thr = avg("saturated_throughput_sps", "inf_throughput_sps")
        load_w = avg("saturated_load_watts", "inf_load_watts_psytar")
        cv = avg("saturated_energy_cv_pct", "inf_energy_cv_pct")
        prov = "saturated_run" if any(
            "saturated_gross_energy_1k_j" in r for r in seeds_res) else (
            "smoke_test" if smoke else "single_inference_run")
        out[disp] = {
            "inf_j_gross": gross, "inf_j_net": net, "throughput_sps": thr,
            "load_w": load_w, "idle_w": float(data.get("gpu_idle_watts", np.nan)),
            "energy_cv_pct": cv,
            "temperature_T": avg("temperature_T"),
            "calib_nll_pre": avg("calib_nll_pre"),
            "calib_nll_post": avg("calib_nll_post"),
            "train_load_watts": avg("train_load_watts"),
            "n_seeds": len(seeds_res),
            "provenance": prov,
        }
    return out


def load_cpu_energy(json_path):
    out = dict(CPU_ENERGY_FALLBACK)
    if not os.path.exists(json_path):
        v2_path = os.path.join(RESULTS_DIR, "cpu_energy_measured_v2.json")
        if os.path.exists(v2_path):
            json_path = v2_path
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            meas = json.load(f)
        for model_name, d in meas.items():
            if not model_name.startswith("_") and isinstance(d, dict):
                out[model_name] = {**d, "provenance": d.get("provenance", "measured_ST3")}
        log(f"[energy] Loaded measured CPU energy from {os.path.basename(json_path)}.")
    else:
        log("[energy] No measured CPU energy JSON — using documented ST2/ST3 constants.")
    return out


# ---------------------------------------------------------------------------
# Frozen split recovery
# ---------------------------------------------------------------------------
def split_from_npz(npz):
    """Return (train_texts, y_train, calib_texts, y_calib, test_texts, y_test)
    if the .npz carries texts (new GPU-script version), else None."""
    keys = set(npz.keys())
    needed = {"train_texts", "calib_texts", "test_texts",
              "y_train", "y_calib", "y_test"}
    if needed.issubset(keys):
        return (npz["train_texts"].astype(str), npz["y_train"],
                npz["calib_texts"].astype(str), npz["y_calib"],
                npz["test_texts"].astype(str), npz["y_test"])
    return None


def reconstruct_split(psytar_csv, seed):
    """Reconstruct the 60/20/20(rs=seed) split on full PsyTAR (6003 rows).
    Mirrors colab_gpu_transformer_primary_adr.py exactly."""
    df = pd.read_csv(psytar_csv)
    train_df, calib_test_df = train_test_split(
        df, train_size=0.6, stratify=df["label"], random_state=seed)
    calib_df, test_df = train_test_split(
        calib_test_df, test_size=0.5, stratify=calib_test_df["label"],
        random_state=seed)
    return train_df, calib_df, test_df


def recover_frozen_split(npz, seed, y_test_npz):
    """Prefer texts saved in the .npz; else reconstruct and verify labels."""
    from_npz = split_from_npz(npz)
    if from_npz is not None:
        tr_t, y_tr, ca_t, y_ca, te_t, y_te = from_npz
        assert np.array_equal(np.asarray(y_te), np.asarray(y_test_npz)), \
            "npz test labels inconsistent with saved test_texts labels"
        log("  [split] Using train/calib/test TEXTS embedded in the .npz "
            "(exact alignment guaranteed).")
        return (list(tr_t), np.asarray(y_tr), list(ca_t), np.asarray(y_ca),
                list(te_t), np.asarray(y_te), "npz_texts")

    log("  [split] .npz has no texts — reconstructing split from CSV and "
        "verifying against transformer labels...")
    train_df, calib_df, test_df = reconstruct_split(PSYTAR_CSV, seed)
    y_te = test_df["label"].values
    if not np.array_equal(np.asarray(y_te), np.asarray(y_test_npz)):
        raise RuntimeError(
            "Reconstructed frozen split does NOT match the transformer .npz "
            "test labels (element-wise). This usually means the local "
            "pandas/scikit-learn version splits differently than Colab. "
            "Re-run the (upgraded) Colab GPU script so the .npz stores the "
            "test texts, then re-run this analysis for guaranteed alignment.")
    log("  [split] OK: reconstructed y_test matches transformer y_test exactly.")
    return (list(train_df["text"]), train_df["label"].values,
            list(calib_df["text"]), calib_df["label"].values,
            list(test_df["text"]), y_te, "reconstructed_verified")


# ---------------------------------------------------------------------------
# Classical arms on the frozen split
# ---------------------------------------------------------------------------
def train_classical_arms(train_texts, y_train, calib_texts, y_calib,
                         test_texts, y_test, cadec_texts, y_cadec):
    """Fit LR + LightGBM on the frozen train split; produce uncal/temp/iso probs
    on PsyTAR test and full CADEC. Returns nested dict of p1 arrays + fitted T."""
    vec = TfidfVectorizer(max_features=1000)
    X_train = vec.fit_transform(train_texts).toarray()
    X_calib = vec.transform(calib_texts).toarray()
    X_test = vec.transform(test_texts).toarray()
    X_cadec = vec.transform(cadec_texts).toarray()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05,
                                       num_leaves=31, random_state=42,
                                       n_jobs=-1, verbose=-1),
    }
    arms = {}
    fitted_T = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        p_calib = clf.predict_proba(X_calib)[:, 1]
        p_test = clf.predict_proba(X_test)[:, 1]
        p_cadec = clf.predict_proba(X_cadec)[:, 1]

        ts = TemperatureScaler()
        ts.fit(y_calib, p_calib)
        fitted_T[name] = ts.T
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_calib, y_calib)

        arms[name] = {
            "calib_p1": {"Uncalibrated": p_calib,
                         "Temperature Scaling": ts.transform(p_calib),
                         "Isotonic Regression": iso.transform(p_calib)},
            "test_p1": {"Uncalibrated": p_test,
                        "Temperature Scaling": ts.transform(p_test),
                        "Isotonic Regression": iso.transform(p_test)},
            "cadec_p1": {"Uncalibrated": p_cadec,
                         "Temperature Scaling": ts.transform(p_cadec),
                         "Isotonic Regression": iso.transform(p_cadec)},
            "fitted_T": ts.T,
            "nll_pre": ts.nll_pre, "nll_post": ts.nll_post,
        }
    return arms, fitted_T


# ---------------------------------------------------------------------------
# Transformer arms recomputed from .npz
# ---------------------------------------------------------------------------
def transformer_arms_from_npz(npz):
    """Return uncal/temp/iso p1 arrays for PsyTAR test and CADEC, plus t* from
    the calibration split logits."""
    y_calib = npz["y_calib"]
    logits_calib = npz["logits_calib"]
    p_calib_uncal = softmax(logits_calib, axis=1)[:, 1]
    t_star, _ = find_optimal_threshold(y_calib, p_calib_uncal)
    return {
        "test_p1": {"Uncalibrated": npz["probs_test_uncal"][:, 1],
                    "Temperature Scaling": npz["probs_test_temp"][:, 1],
                    "Isotonic Regression": npz["probs_test_iso"][:, 1]},
        "cadec_p1": {"Uncalibrated": npz["probs_cadec_uncal"][:, 1],
                     "Temperature Scaling": npz["probs_cadec_temp"][:, 1],
                     "Isotonic Regression": npz["probs_cadec_iso"][:, 1]},
        "t_star": t_star,
    }


# ---------------------------------------------------------------------------
# Build the 12-arm catalogue
# ---------------------------------------------------------------------------
RECAL_ORDER = ["Uncalibrated", "Temperature Scaling", "Isotonic Regression"]
RECAL_SHORT = {"Uncalibrated": "Uncalibrated",
               "Temperature Scaling": "TempScale",
               "Isotonic Regression": "Isotonic"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    log("=" * 90)
    log("  FROZEN-SPLIT ANALYSIS  —  classical arms + paired bootstrap + ECC-MS")
    log("=" * 90)

    # ---- discover transformer npz, group by seed ----
    npz_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_predictions.npz")))
    npz_files = [f for f in npz_files if "cpu_arms" not in os.path.basename(f)]
    if not npz_files:
        log("[FATAL] No transformer *_predictions.npz in results/. Run the Colab "
            "GPU script first and drop the .npz files into results/.")
        return

    by_seed = {}
    for f in npz_files:
        base = os.path.basename(f).replace("_predictions.npz", "")
        stem, _, seedtok = base.rpartition("_seed")
        model = NPZ_MODEL_MAP.get(stem, stem)
        try:
            seed = int(seedtok)
        except ValueError:
            seed = 42
        by_seed.setdefault(seed, {})[model] = f

    primary_seed = 42 if 42 in by_seed else sorted(by_seed)[0]
    log(f"\nSeeds present: {sorted(by_seed)} | primary (for paired bootstrap): "
        f"{primary_seed}")
    seed_models = by_seed[primary_seed]
    log(f"Transformer arms @seed {primary_seed}: {sorted(seed_models)}")

    # ---- energy ----
    gpu_energy = load_gpu_energy(
        os.path.join(RESULTS_DIR, "colab_transformer_gpu_results.json"))
    cpu_energy = load_cpu_energy(
        os.path.join(RESULTS_DIR, "cpu_energy_measured.json"))
    energy_by_model = dict(cpu_energy)
    energy_by_model.update(gpu_energy)

    # ---- load transformer npz, confirm shared split ----
    npz_by_model = {m: np.load(p, allow_pickle=True)
                    for m, p in seed_models.items()}
    y_tests = {m: d["y_test"] for m, d in npz_by_model.items()}
    ref_model = sorted(y_tests)[0]
    y_test_ref = y_tests[ref_model]
    for m, yt in y_tests.items():
        if not np.array_equal(np.asarray(yt), np.asarray(y_test_ref)):
            log(f"[FATAL] Transformer arms are NOT on the same frozen test split "
                f"({m} vs {ref_model}). Re-run all arms with identical seeds.")
            return
    log(f"\n[split] All transformer arms share one frozen PsyTAR test set "
        f"(N={len(y_test_ref)}).  CONFIRMED.")

    # ---- CADEC (full, frozen) ----
    df_cadec = pd.read_csv(CADEC_CSV)
    cadec_texts = list(df_cadec["text"].astype(str))
    y_cadec_csv = df_cadec["label"].values
    y_cadec_npz = npz_by_model[ref_model]["y_cadec"]
    cadec_aligned = (len(y_cadec_csv) == len(y_cadec_npz)
                     and np.array_equal(np.asarray(y_cadec_csv),
                                        np.asarray(y_cadec_npz)))
    log(f"[split] CADEC full frozen split N={len(y_cadec_csv)} "
        f"(transformer CADEC N={len(y_cadec_npz)}; "
        f"aligned={cadec_aligned}).")

    # ---- recover the frozen split & train classical arms ----
    (train_texts, y_train, calib_texts, y_calib,
     test_texts, y_test, split_prov) = recover_frozen_split(
        npz_by_model[ref_model], primary_seed, y_test_ref)

    log(f"\n[classical] Training LR + LightGBM on frozen train "
        f"(N={len(train_texts)}), calib (N={len(calib_texts)}), "
        f"test (N={len(test_texts)}), CADEC (N={len(cadec_texts)}) ...")
    classical, fitted_T = train_classical_arms(
        train_texts, y_train, calib_texts, y_calib, test_texts, y_test,
        cadec_texts, y_cadec_csv)

    # ---- assemble per-model test/cadec probability dicts ----
    model_test_p1, model_cadec_p1 = {}, {}
    for name in ("Logistic Regression", "LightGBM"):
        model_test_p1[name] = classical[name]["test_p1"]
        model_cadec_p1[name] = classical[name]["cadec_p1"]
    for model, npz in npz_by_model.items():
        t = transformer_arms_from_npz(npz)
        model_test_p1[model] = t["test_p1"]
        # CADEC probs align with y_cadec_npz; if CSV order == npz order this also
        # matches y_cadec_csv (checked above).
        model_cadec_p1[model] = t["cadec_p1"]

    # For CADEC ece we must pair each model's cadec probs with its own y_cadec.
    y_cadec_for = {m: (y_cadec_csv if m in ("Logistic Regression", "LightGBM")
                       else npz_by_model[m]["y_cadec"]) for m in model_cadec_p1}

    # ---- build the full 12-arm catalogue (recomputed metrics) ----
    configs = []
    per_arm = {}
    for model in model_test_p1:
        e = energy_by_model.get(model, {})
        for recal in RECAL_ORDER:
            if recal not in model_test_p1[model]:
                continue
            p_test = np.asarray(model_test_p1[model][recal])
            p_cadec = np.asarray(model_cadec_p1[model][recal])
            yc = np.asarray(y_cadec_for[model])
            m_test = compute_full_metrics(y_test, p_test, threshold=0.5)
            cadec_ece = compute_ece_adaptive(yc, p_cadec)
            cadec_auroc = float(compute_full_metrics(yc, p_cadec,
                                                     threshold=0.5)["AUROC"])
            name = f"{model} + {RECAL_SHORT[recal]}"
            cfg = {"name": name, "model": model, "recal": recal,
                   "auroc": float(m_test["AUROC"]),
                   "ece": float(m_test["ECE_adaptive"]),
                   "ece_ci_hi": float(m_test["ECE_CI_hi"]),
                   "cadec_ece": float(cadec_ece), "cadec_auroc": cadec_auroc,
                   "inf_j_gross": e.get("inf_j_gross"),
                   "inf_j_net": e.get("inf_j_net")}
            configs.append(cfg)
            per_arm[name] = {"auroc": cfg["auroc"], "ece": cfg["ece"],
                             "auprc": float(m_test["AUPRC"]),
                             "f1_at_tstar": float(m_test["F1@t*"]),
                             "ece_ci": [float(m_test["ECE_CI_lo"]),
                                        float(m_test["ECE_CI_hi"])],
                             "brier": float(m_test["Brier"]),
                             "nll": float(m_test["NLL"]),
                             "cadec_ece": float(cadec_ece),
                             "cadec_auroc": cadec_auroc,
                             "energy_gross": cfg["inf_j_gross"],
                             "energy_net": cfg["inf_j_net"]}

    # ---- per-model uncalibrated test p1 for the paired bootstrap ----
    model_probs_uncal = {m: np.asarray(model_test_p1[m]["Uncalibrated"])
                         for m in model_test_p1}

    log("\n" + "-" * 90)
    log("  PER-ARM RECOMPUTED METRICS (frozen PsyTAR test, N=%d)" % len(y_test))
    log("-" * 90)
    hdr = f"{'Arm':32} {'AUROC':>7} {'ECE':>7} {'CADEC_ECE':>10} {'GrossJ/1k':>10} {'NetJ/1k':>9}"
    log(hdr)
    for c in configs:
        gj = "PENDING" if c["inf_j_gross"] is None else f"{c['inf_j_gross']:.4f}"
        nj = "PENDING" if c["inf_j_net"] is None else f"{c['inf_j_net']:.4f}"
        log(f"{c['name']:32} {c['auroc']:.4f} {c['ece']:.4f} "
            f"{c['cadec_ece']:.4f}     {gj:>10} {nj:>9}")

    # ---- paired bootstrap ΔAUROC matrix (the statistical-tie evidence) ----
    log("\n" + "-" * 90)
    log("  PAIRED BOOTSTRAP  Delta_AUROC  (shared resamples; tie iff CI includes 0)")
    log("-" * 90)
    matrix = pairwise_delta_auroc_matrix(
        y_test, model_probs_uncal, n_bootstrap=N_BOOTSTRAP, seed=BOOT_SEED)
    for r in matrix:
        verdict = "TIE (CI incl. 0)" if r["statistical_tie"] else "DISTINGUISHABLE"
        log(f"  {r['model_a']:22} - {r['model_b']:22} "
            f"Delta={r['delta_auroc']:+.4f}  CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  {verdict}")

    # ---- CADEC paired bootstrap vs leader model ----
    model_cadec_probs_uncal = {m: np.asarray(model_cadec_p1[m]["Uncalibrated"])
                               for m in model_cadec_p1}
    from sklearn.metrics import roc_auc_score
    cadec_leader_model = max(model_cadec_probs_uncal.keys(), key=lambda m: roc_auc_score(y_cadec_csv, model_cadec_probs_uncal[m]))
    log(f"\n  [CADEC bootstrap] leader={cadec_leader_model} (AUROC={roc_auc_score(y_cadec_csv, model_cadec_probs_uncal[cadec_leader_model]):.4f})")
    
    from eccms_selection import paired_delta_auroc
    # Pre-warm bootstrap cache for ALL model pairs (test set + CADEC) to make grid loop instantaneous
    models_list = list(model_probs_uncal.keys())
    for i, m1 in enumerate(models_list):
        for m2 in models_list[i+1:]:
            paired_delta_auroc(y_test, model_probs_uncal[m1], model_probs_uncal[m2],
                               n_bootstrap=N_BOOTSTRAP, seed=BOOT_SEED)
            if m1 in model_cadec_probs_uncal and m2 in model_cadec_probs_uncal:
                paired_delta_auroc(y_cadec_csv, model_cadec_probs_uncal[m1], model_cadec_probs_uncal[m2],
                                   n_bootstrap=N_BOOTSTRAP, seed=BOOT_SEED)

    cadec_tie_with_leader = {}
    for m in model_probs_uncal:
        if m == cadec_leader_model:
            cadec_tie_with_leader[m] = True
            log(f"    {m:22}: TIE (is leader)")
        else:
            _, lo, hi = paired_delta_auroc(
                y_cadec_csv, model_cadec_probs_uncal[cadec_leader_model], model_cadec_probs_uncal[m],
                n_bootstrap=N_BOOTSTRAP, seed=BOOT_SEED)
            is_tie = bool(lo <= 0.0 <= hi)
            cadec_tie_with_leader[m] = is_tie
            verdict = "TIE (CI includes 0)" if is_tie else "DISTINGUISHABLE"
            log(f"    {m:22}: {verdict} vs leader, CI=[{lo:+.4f}, {hi:+.4f}]")

    # ---- Statistical Power & Minimum Detectable Difference (MDD) ----
    from metrics_utils import compute_mdd_and_power, tost_equivalence_test
    p_psytar = compute_mdd_and_power(len(y_test))
    p_cadec = compute_mdd_and_power(len(y_cadec_csv))
    log("\n" + "-" * 90)
    log("  STATISTICAL POWER & MINIMUM DETECTABLE DIFFERENCE (MDD)")
    log("-" * 90)
    log(f"  PsyTAR (N={len(y_test)}):  MDD = ±{p_psytar['mdd_auroc']:.4f} AUROC (80% power @ alpha=0.05)")
    log(f"  CADEC  (N={len(y_cadec_csv)}):  MDD = ±{p_cadec['mdd_auroc']:.4f} AUROC (80% power @ alpha=0.05)")
    log("  TOST Equivalence Margin: pre-registered delta = 0.015 AUROC")

    # ---- ECC-MS grid: argmax vs bootstrap-tie vs fixed-margin strip ----
    log("\n" + "-" * 90)
    log("  ECC-MS SELECTION GRID (energy budget in GROSS J/1k, matching the table)")
    log("-" * 90)
    grid_rows = []
    all_cells = sorted(set(RECONCILE_CELLS) | {(t, e) for t in TAU_GRID for e in E_GRID})
    for tau, E in all_cells:
        feas = feasible_arms(configs, tau, E, use_gross=True, use_ece_ci=True)
        argmax_sel, n_feas = eccms_select_argmax(configs, tau, E, use_gross=True, use_ece_ci=True)
        tie_sel, _, tie_info = eccms_select_bootstrap_tie(
            configs, tau, E, y_test, model_probs_uncal,
            n_bootstrap=N_BOOTSTRAP, seed=BOOT_SEED, use_gross=True, use_ece_ci=True,
            tost_delta_eq=0.015, y_ood=y_cadec_csv, model_probs_ood=model_cadec_probs_uncal)
        margin_sels = {
            f"margin_{m}": (eccms_select_fixed_margin(
                configs, tau, E, margin=m, use_gross=True, use_ece_ci=True)[0] or {}).get("name")
            for m in MARGINS}
        cadec_ok = None
        selected_in_cadec_tie_band = None
        if tie_sel is not None:
            cadec_ok = bool(tie_sel["cadec_ece"] <= tau + 1e-12)
            selected_in_cadec_tie_band = cadec_tie_with_leader.get(tie_sel["model"], False)
        row = {
            "tau": tau, "E_gross_J_per_1k": E,
            "feasible_arms": n_feas,
            "feasible_arm_names": [c["name"] for c in feas],
            "argmax_selected": argmax_sel["name"] if argmax_sel else None,
            "bootstrap_tie_selected": tie_sel["name"] if tie_sel else None,
            "bootstrap_tie_selected_auroc": tie_sel["auroc"] if tie_sel else None,
            "bootstrap_tie_selected_gross_J": tie_sel["inf_j_gross"] if tie_sel else None,
            "bootstrap_tie_selected_cadec_ece": tie_sel["cadec_ece"] if tie_sel else None,
            "selected_satisfies_tau_on_CADEC": cadec_ok,
            "selected_in_cadec_tie_band": selected_in_cadec_tie_band,
            "tie_info": tie_info,
            **margin_sels,
        }
        grid_rows.append(row)
        star = " <-- reviewer cell" if (tau, E) in RECONCILE_CELLS else ""
        log(f"  tau<={tau:.2f} E<={E:>5.1f}J | feasible={n_feas:>2} | "
            f"argmax={row['argmax_selected']} | tie={row['bootstrap_tie_selected']} | "
            f"CADEC tau-ok={cadec_ok} | CADEC tie-band={selected_in_cadec_tie_band}{star}")

    # ---- feasible-count reconciliation (the reviewer's 8/11 dispute) ----
    log("\n" + "-" * 90)
    log("  FEASIBLE-ARM COUNT RECONCILIATION (full 12-arm catalogue)")
    log("-" * 90)
    for tau, E in RECONCILE_CELLS:
        feas = feasible_arms(configs, tau, E, use_gross=True)
        log(f"  tau<={tau:.2f}, E<={E:.1f}J (gross): {len(feas)} feasible -> "
            f"{[c['name'] for c in feas]}")

    # ---- DrugLib row-count reconciliation ----
    # Resolves the canonical path first, then falls back to a search under data/
    # so a moved file is FOUND-AND-REPORTED rather than silently skipped. A
    # missing file records status="MISSING" (never an empty {}), because an empty
    # dict previously looked indistinguishable from "reconciled fine".
    log("\n" + "-" * 90)
    log("  DrugLib row-count reconciliation")
    log("-" * 90)
    druglib_path, druglib_path_source = DRUGLIB_CSV, "canonical_path"
    if not os.path.exists(druglib_path):
        cands = sorted(glob.glob(os.path.join(
            DATA_DIR, "**", "uci_druglib_harmonised.csv"), recursive=True))
        if cands:
            druglib_path, druglib_path_source = cands[0], "glob_fallback"
            log(f"  [warn] Canonical path missing; found via search: "
                f"{os.path.relpath(druglib_path, ROOT)}")

    if os.path.exists(druglib_path):
        dfd = pd.read_csv(druglib_path)
        n_raw = len(dfd)
        text_col = "text" if "text" in dfd.columns else dfd.columns[0]
        txt = dfd[text_col].astype(str).str.strip()
        n_nonempty = int(((txt != "") & (txt.str.lower() != "nan")).sum())
        label_counts = ({str(k): int(v) for k, v in
                         dfd["label"].value_counts().sort_index().items()}
                        if "label" in dfd.columns else None)
        druglib_info = {
            "status": "OK",
            "path": os.path.relpath(druglib_path, ROOT).replace("\\", "/"),
            "path_source": druglib_path_source,
            "sha16": csv_sha(druglib_path),
            "rows_raw": int(n_raw),
            "rows_nonempty_text": n_nonempty,
            "rows_dropped_empty_text": int(n_raw - n_nonempty),
            "label_counts": label_counts,
            "columns": list(dfd.columns),
        }
        log(f"  raw rows = {n_raw} | non-empty-text rows = {n_nonempty} "
            f"| dropped = {n_raw - n_nonempty}  "
            f"(this is what resolves any 4,107 vs 4,108 gap)")
        if label_counts:
            log(f"  label counts = {label_counts}")
    else:
        druglib_info = {
            "status": "MISSING",
            "path": None,
            "path_source": None,
            "sha16": None,
            "expected_path": os.path.relpath(DRUGLIB_CSV, ROOT).replace("\\", "/"),
            "rows_raw": None, "rows_nonempty_text": None,
            "rows_dropped_empty_text": None,
            "label_counts": None, "columns": None,
        }
        log(f"  [WARN] DrugLib CSV NOT FOUND at "
            f"{os.path.relpath(DRUGLIB_CSV, ROOT)} — recorded as status=MISSING. "
            f"The 4,107-vs-4,108 count stays UNRESOLVED; do not quote either "
            f"number until this file is present.")

    # ---- save CPU predictions aligned to the transformer test set ----
    cpu_npz = os.path.join(RESULTS_DIR,
                           f"cpu_arms_seed{primary_seed}_predictions.npz")
    np.savez_compressed(
        cpu_npz,
        y_test=y_test, y_cadec=y_cadec_csv,
        lr_test_uncal=classical["Logistic Regression"]["test_p1"]["Uncalibrated"],
        lr_test_temp=classical["Logistic Regression"]["test_p1"]["Temperature Scaling"],
        lr_test_iso=classical["Logistic Regression"]["test_p1"]["Isotonic Regression"],
        gbdt_test_uncal=classical["LightGBM"]["test_p1"]["Uncalibrated"],
        gbdt_test_temp=classical["LightGBM"]["test_p1"]["Temperature Scaling"],
        gbdt_test_iso=classical["LightGBM"]["test_p1"]["Isotonic Regression"],
        lr_cadec_uncal=classical["Logistic Regression"]["cadec_p1"]["Uncalibrated"],
        gbdt_cadec_uncal=classical["LightGBM"]["cadec_p1"]["Uncalibrated"],
    )
    log(f"\n[artifact] CPU arm predictions saved: {os.path.basename(cpu_npz)}")

    # ---- multi-seed aggregated performance across seeds [42, 123, 456] ----
    all_seeds = [42, 123, 456]
    multi_seed_records = {}
    for seed in all_seeds:
        tr_df, cal_df, tst_df = reconstruct_split(PSYTAR_CSV, seed)
        tr_txt, y_tr = list(tr_df["text"]), tr_df["label"].values
        cal_txt, y_cal = list(cal_df["text"]), cal_df["label"].values
        tst_txt, y_tst = list(tst_df["text"]), tst_df["label"].values
        cls_seed, _ = train_classical_arms(tr_txt, y_tr, cal_txt, y_cal, tst_txt, y_tst, cadec_texts, y_cadec_csv)
        
        # Classical arms
        for model in ("Logistic Regression", "LightGBM"):
            for recal in RECAL_ORDER:
                name = f"{model} + {RECAL_SHORT[recal]}"
                p_tst = np.asarray(cls_seed[model]["test_p1"][recal])
                p_cad = np.asarray(cls_seed[model]["cadec_p1"][recal])
                a_tst = float(roc_auc_score(y_tst, p_tst))
                e_tst = float(compute_ece_adaptive(y_tst, p_tst))
                a_cad = float(roc_auc_score(y_cadec_csv, p_cad))
                e_cad = float(compute_ece_adaptive(y_cadec_csv, p_cad))
                multi_seed_records.setdefault(name, []).append({
                    "seed": seed, "auroc": a_tst, "ece": e_tst,
                    "cadec_auroc": a_cad, "cadec_ece": e_cad
                })

        # Transformer arms (from colab_transformer_gpu_results.json for genuine 3-seed evaluation)
        gpu_json_path = os.path.join(RESULTS_DIR, "colab_transformer_gpu_results.json")
        if os.path.exists(gpu_json_path):
            with open(gpu_json_path, "r", encoding="utf-8") as f:
                gpu_res = json.load(f).get("results", {})
            for model, display in (("Efficient Transformer", "DistilBERT"), ("Biomedical Transformer", "PubMedBERT")):
                entries = [e for e in gpu_res.get(model, []) if e.get("seed") == seed]
                if entries:
                    eval_map = entries[0].get("eval_results", {})
                    for recal in RECAL_ORDER:
                        rec_info = eval_map.get(recal, {})
                        if rec_info:
                            name = f"{display} + {RECAL_SHORT[recal]}"
                            p_m = rec_info.get("psytar", {})
                            c_m = rec_info.get("cadec", {})
                            multi_seed_records.setdefault(name, []).append({
                                "seed": seed,
                                "auroc": float(p_m.get("AUROC", 0.0)),
                                "ece": float(p_m.get("ECE_adaptive", 0.0)),
                                "cadec_auroc": float(c_m.get("AUROC", 0.0)),
                                "cadec_ece": float(c_m.get("ECE_adaptive", 0.0)),
                            })

    multi_seed_summary = {}
    for name, recs in multi_seed_records.items():
        aurocs = [r["auroc"] for r in recs]
        eces = [r["ece"] for r in recs]
        c_aurocs = [r["cadec_auroc"] for r in recs]
        c_eces = [r["cadec_ece"] for r in recs]
        multi_seed_summary[name] = {
            "n_seeds": len(recs),
            "seeds": [r["seed"] for r in recs],
            "auroc_mean": float(np.mean(aurocs)), "auroc_std": float(np.std(aurocs, ddof=0)),
            "ece_mean": float(np.mean(eces)), "ece_std": float(np.std(eces, ddof=0)),
            "cadec_auroc_mean": float(np.mean(c_aurocs)), "cadec_auroc_std": float(np.std(c_aurocs, ddof=0)),
            "cadec_ece_mean": float(np.mean(c_eces)), "cadec_ece_std": float(np.std(c_eces, ddof=0)),
        }

    # ---- write reconciled JSON (single source of truth) ----
    reconciled = {
        "multi_seed_metrics": multi_seed_summary,
        "provenance": {
            "generated_by": "run_frozen_split_analysis.py",
            "primary_seed": primary_seed,
            "split_source": split_prov,
            "test_N": int(len(y_test)),
            "cadec_N": int(len(y_cadec_csv)),
            "cadec_aligned_with_transformer": bool(cadec_aligned),
            "n_bootstrap": N_BOOTSTRAP,
            "psytar_csv_sha16": csv_sha(PSYTAR_CSV),
            "cadec_csv_sha16": csv_sha(CADEC_CSV),
            "druglib_csv_sha16": csv_sha(druglib_path),
            "cpu_energy_provenance": {k: v.get("provenance")
                                      for k, v in cpu_energy.items()},
            "gpu_energy_provenance": {k: v.get("provenance")
                                      for k, v in gpu_energy.items()},
        },
        "fitted_temperatures_cpu": {k: float(v) for k, v in fitted_T.items()},
        "model_hyperparameters": MODEL_HYPERPARAMETERS,
        "energy_by_model": energy_by_model,
        "per_arm_metrics": per_arm,
        "catalogue": configs,
        "paired_delta_auroc": matrix,
        "eccms_grid": grid_rows,
        "feasible_reconcile_cells": [
            {"tau": t, "E_gross_J_per_1k": e,
             "feasible": len(feasible_arms(configs, t, e, use_gross=True)),
             "arms": [c["name"] for c in feasible_arms(configs, t, e, use_gross=True)]}
            for (t, e) in RECONCILE_CELLS],
        "druglib": druglib_info,
    }

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    out_path = os.path.join(RESULTS_DIR, "frozen_split_reconciled.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reconciled, f, indent=2, default=_default)
    log(f"[artifact] Reconciled source-of-truth JSON: {os.path.basename(out_path)}")

    log("\n" + "=" * 90)
    log("  DONE.  Send results/frozen_split_reconciled.json (and the GPU JSON) "
        "back for README reconciliation.")
    log("=" * 90)


if __name__ == "__main__":
    main()
