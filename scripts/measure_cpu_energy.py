#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measure_cpu_energy.py  —  Saturated-batch CPU inference energy for the classical
arms (Round 5 rigorous overhaul; CPU analogue of the GPU saturated-run fix).

WHY THIS EXISTS
---------------
Round 5 required that energy be measured from a real *saturated* run where power,
throughput and energy are captured TOGETHER (not stitched from separate numbers).
This script does exactly that for the CPU arms:

  * idle package power is measured over a quiet window,
  * each classical model (LogReg, LightGBM) is driven to steady state on a large
    tiled batch, and over the measurement window we capture — simultaneously —
    the integrated package energy (Intel RAPL), the wall-clock throughput, and
    hence the load power,
  * 3 repeats -> mean + coefficient of variation (CV%),
  * gross J/1k = energy_window / n_inferences x 1000
    net   J/1k = gross x (load_power - idle_power)/load_power.

PROVENANCE (never fabricated)
-----------------------------
  * If Intel RAPL is readable (Linux, /sys/class/powercap/intel-rapl:*), energy is
    truly measured  -> provenance = "measured_rapl_saturated".
  * If RAPL is NOT readable (e.g. Windows host, or root-only sysfs), the script
    STILL measures throughput honestly at steady state, and combines it with the
    documented ST2 package-power constants to derive energy. That case is tagged
    "measured_throughput_x_ST2_power" so a reviewer sees throughput is fresh but
    power is the earlier ST2 measurement — nothing is passed off as a live power
    reading when it was not.

The frozen split and the model hyper-parameters mirror
run_frozen_split_analysis.train_classical_arms EXACTLY, so the timed model is the
same object the metrics are computed on.

USAGE
-----
    python scripts/measure_cpu_energy.py
    python scripts/measure_cpu_energy.py --measure-s 20 --repeats 3

Writes results/cpu_energy_measured.json, consumed by run_frozen_split_analysis.py.
"""
from __future__ import annotations

import os
import sys
import glob
import json
import time
import platform
import subprocess
from datetime import datetime, timezone
import argparse
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")
CONFIGS_DIR = os.path.join(ROOT, "configs")
CONFIG_PATH = os.path.join(CONFIGS_DIR, "default_config.json")
PSYTAR_CSV = os.path.join(DATA_DIR, "01_primary_adr_detection", "dev_psytar",
                          "psytar_harmonised.csv")

sys.path.insert(0, HERE)
# Reuse the EXACT split logic used to compute the metrics (no drift).
from run_frozen_split_analysis import reconstruct_split  # noqa: E402
from rapl_utils import RAPLReader, probe_environment  # noqa: E402

MODEL_ORDER = ["Logistic Regression", "LightGBM"]


def log(msg=""):
    print(msg, flush=True)


def load_tfidf_config():
    """Load canonical TF-IDF configuration from configs/default_config.json."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cm = cfg.get("primary_adr_detection", {}).get("classical_models", {}).get("tfidf", {})
        return {
            "ngram_range": tuple(cm.get("ngram_range", [1, 2])),
            "max_features": cm.get("max_features", 2500),
        }
    return {"ngram_range": (1, 2), "max_features": 2500}


def _read_domains(domains):
    return [int(open(e, "r").read().strip()) for (e, _) in domains]


def _delta_j(domains, before, after):
    total_uj = 0
    for (path, maxr), b, a in zip(domains, before, after):
        d = a - b
        if d < 0 and maxr > 0:
            d += maxr
        total_uj += d
    return total_uj / 1e6


# ---------------------------------------------------------------------------
# Idle power
# ---------------------------------------------------------------------------
def measure_idle_power(rapl, duration_s):
    if not rapl.ok:
        return None
    before = _read_domains(rapl.domains)
    t0 = time.perf_counter()
    time.sleep(duration_s)
    t1 = time.perf_counter()
    after = _read_domains(rapl.domains)
    ej = _delta_j(rapl.domains, before, after)
    return ej / (t1 - t0)


# ---------------------------------------------------------------------------
# Saturated inference benchmark for one fitted model
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Saturated inference benchmark for one fitted model (end-to-end throughput)
# ---------------------------------------------------------------------------
def saturated_infer(clf, vec, raw_texts_bench, rapl, warmup_s, measure_s):
    """Drive `vec` + `clf` to steady state on raw_texts_bench (end-to-end vectorization
    + inference), then over the measurement window capture energy (if RAPL),
    throughput and load power together.

    Returns dict with throughput_sps, and (if RAPL) energy_window_j / load_w.
    """
    n_batch = len(raw_texts_bench)

    # Warm up (fill caches, spin up threads) — not measured.
    t_end = time.perf_counter() + warmup_s
    while time.perf_counter() < t_end:
        X_vec = vec.transform(raw_texts_bench)
        clf.predict_proba(X_vec)

    # Measurement window: energy + throughput captured simultaneously.
    before = _read_domains(rapl.domains) if rapl.ok else None
    t0 = time.perf_counter()
    n_infer = 0
    # Run in whole-batch units; stop once we pass measure_s.
    while (time.perf_counter() - t0) < measure_s:
        X_vec = vec.transform(raw_texts_bench)
        clf.predict_proba(X_vec)
        n_infer += n_batch
    t1 = time.perf_counter()
    after = _read_domains(rapl.domains) if rapl.ok else None

    elapsed = t1 - t0
    throughput = n_infer / elapsed
    out = {"throughput_sps": throughput, "n_infer": n_infer,
           "elapsed_s": elapsed}
    if rapl.ok:
        ej = _delta_j(rapl.domains, before, after)
        out["energy_window_j"] = ej
        out["load_w"] = ej / elapsed
        out["gross_j_1k_integrated"] = ej / n_infer * 1000.0
    return out


def get_run_id():
    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        pass
    boot_id = "unknown"
    if os.path.exists("/proc/sys/kernel/random/boot_id"):
        try:
            with open("/proc/sys/kernel/random/boot_id", "r") as f:
                boot_id = f.read().strip()
        except Exception:
            pass
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "argv": sys.argv,
        "host_boot_id": boot_id,
        "python_version": platform.python_version(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup-s", type=float, default=2.0)
    ap.add_argument("--measure-s", type=float, default=20.0,
                    help="measurement window per repeat in seconds (default: 20.0)")
    ap.add_argument("--idle-s", type=float, default=8.0)
    ap.add_argument("--repeats", type=int, default=7,
                    help="number of benchmark repeats (default: 7, uses median)")
    ap.add_argument("--bench-rows", type=int, default=50000,
                    help="target rows in the tiled saturation batch")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    log("=" * 88)
    log("  CPU SATURATED-BATCH ENERGY  —  classical arms (power+end-to-end throughput)")
    log("=" * 88)

    rapl = RAPLReader()
    if not rapl.ok:
        raise SystemExit(
            f"\n[FATAL] Intel RAPL is unavailable ({rapl.reason}).\n"
            "        Refusing to emit energy numbers from a hardcoded constant.\n"
            "        Run on bare-metal Linux with readable powercap counters (/sys/class/powercap/intel-rapl:*)."
        )

    log(f"[rapl] Intel RAPL available: {len(rapl.domains)} package domain(s) "
        f"{rapl.domain_names}. Energy will be measured live.")
    log(f"       NOTE: Only top-level package domains (package-*) are summed.")
    log(f"       Platform (psys) and subzones (core, uncore, dram) are NOT included.")

    # ---- host hardware disclosure ----
    env = probe_environment()
    log(f"\n[host] CPU model    : {env.get('cpu_model', 'unknown')}")
    log(f"[host] Physical cores: {env.get('physical_cores', 'unknown')}")
    log(f"[host] Logical CPUs  : {env.get('logical_cpus', 'unknown')}")
    log(f"[host] Sockets       : {env.get('socket_count', 'unknown')}")
    log(f"[host] TDP (RAPL PL1): {env.get('tdp_watts', 'not available')} W")
    log(f"[host] Platform      : {env.get('platform', 'unknown')}")

    # ---- rebuild the exact frozen split + fit the exact models ----
    if not os.path.exists(PSYTAR_CSV):
        log(f"[FATAL] Missing {PSYTAR_CSV}"); return
    train_df, calib_df, test_df = reconstruct_split(PSYTAR_CSV, args.seed)
    tfidf_cfg = load_tfidf_config()
    vec = TfidfVectorizer(**tfidf_cfg)
    X_train = vec.fit_transform(list(train_df["text"]))
    y_train = train_df["label"].values
    test_texts = list(test_df["text"])
    log(f"[data] train={X_train.shape} test={len(test_texts)} texts "
        f"(features={X_train.shape[1]}, config={tfidf_cfg})")

    # Tile the raw test texts into a large steady-state batch for end-to-end vectorization.
    reps = max(1, int(np.ceil(args.bench_rows / max(1, len(test_texts)))))
    raw_texts_bench = test_texts * reps
    log(f"[bench] benchmark batch: {len(raw_texts_bench):,} texts ({reps}x tiled test split)")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05,
                                       num_leaves=31, random_state=42,
                                       n_jobs=-1, verbose=-1),
    }
    for clf in models.values():
        clf.fit(X_train, y_train)

    # ---- idle power (once, shared) ----
    idle_w = measure_idle_power(rapl, args.idle_s)
    if idle_w is None:
        raise SystemExit("[FATAL] Could not measure idle power via RAPL counters.")
    log(f"[idle] package idle power = {idle_w:.3f} W "
        f"(RAPL, {args.idle_s:.0f}s window)")

    # ---- per-model saturated benchmark, `repeats` times ----
    results = {}
    for name in MODEL_ORDER:
        clf = models[name]
        log(f"\n[bench] {name}: {args.repeats} repeat(s) x "
            f"({args.warmup_s:.0f}s warmup + {args.measure_s:.0f}s measure)")
        gross_list, thr_list, load_list = [], [], []

        for r in range(args.repeats):
            # End-to-end benchmark
            res = saturated_infer(clf, vec, raw_texts_bench, rapl, args.warmup_s, args.measure_s)
            thr = res["throughput_sps"]
            thr_list.append(thr)
            gross = res["gross_j_1k_integrated"]
            load_w = res["load_w"]
            gross_list.append(gross)
            load_list.append(load_w)

            log(f"    repeat {r+1}: End-to-End thr={thr:,.0f} s/s | load={load_w:.3f} W | gross={gross:.4f} J/1k")

        # Record median across repeats
        gross_val = float(np.median(gross_list))
        thr_val = float(np.median(thr_list))
        load_val = float(np.median(load_list))
        cv = float(np.std(gross_list, ddof=0) / gross_val * 100.0) if gross_val else 0.0
        net_power = max(0.0, load_val - idle_w)
        net_gross_ratio = (net_power / load_val) if load_val else 0.0
        net_1k = gross_val * net_gross_ratio

        results[name] = {
            "inf_j_gross": gross_val,
            "inf_j_net": net_1k,
            "throughput_sps": thr_val,
            "load_w": load_val,
            "idle_w": float(idle_w),
            "net_power_w": net_power,
            "energy_cv_pct": cv,
            "scopes": {
                "end_to_end": {
                    "throughput_sps": thr_val,
                    "load_w": load_val,
                    "gross_j_1k": gross_val,
                    "net_j_1k": net_1k,
                    "includes_tfidf_vectorization": True,
                }
            },
            "n_repeats": args.repeats,
            "measure_window_s": args.measure_s,
            "method": "rapl_integrated_saturated_end2end",
            "includes_tfidf_vectorization": True,
            "summary_stat": "median",
            "provenance": "measured_rapl_saturated",
        }
        log(f"  -> median gross={gross_val:.4f} J/1k | net={net_1k:.4f} J/1k | "
            f"throughput={thr_val:,.0f} s/s | CV={cv:.2f}%")
        if cv > 10.0:
            log(f"  [WARN] {name} energy CV ({cv:.2f}%) exceeds the 10% stability gate.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "cpu_energy_measured.json")
    payload = {
        "_meta": {
            "generated_by": "measure_cpu_energy.py",
            "rapl_available": True,
            "rapl_domains": len(rapl.domains),
            "rapl_domain_names": rapl.domain_names,
            "rapl_note": ("Only top-level package domains (package-*) are summed. "
                         "Platform/psys and subzones (core, uncore, dram) are excluded to avoid double-counting."),
            "idle_power_w": float(idle_w),
            "idle_source": "rapl",
            "bench_rows": len(raw_texts_bench),
            "seed": args.seed,
            "benchmark_scope": "end_to_end_tfidf_plus_predict",
            "scope_description": ("End-to-End: Raw text -> TfidfVectorizer.transform -> "
                                  "clf.predict_proba."),
            "_run_id": get_run_id(),
            "host": {
                "cpu_model": env.get("cpu_model", "unknown"),
                "physical_cores": env.get("physical_cores"),
                "logical_cpus": env.get("logical_cpus"),
                "socket_count": env.get("socket_count"),
                "tdp_watts": env.get("tdp_watts"),
                "platform": env.get("platform", "unknown"),
                "os": env.get("os"),
                "os_release": env.get("os_release"),
                "is_wsl": env.get("is_wsl"),
                "is_container": env.get("is_container"),
            },
        },
        **results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log(f"\n[artifact] wrote {out_path}")
    log("Provenance: live RAPL energy (measured_rapl_saturated)")
    log("=" * 88)


if __name__ == "__main__":
    main()
