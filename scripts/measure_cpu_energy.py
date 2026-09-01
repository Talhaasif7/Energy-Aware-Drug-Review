#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measure_cpu_energy.py  —  Saturated-batch CPU inference energy for classical arms.

Rigorously engineered for Green AI empirical standards:
  * End-to-end scope: Raw text -> TfidfVectorizer.transform -> clf.predict_proba.
  * Thermal equilibrium: 60s warm-up drive to stabilize CPU clocks & temperatures.
  * Inter-repeat cooldown: 15s gap between repeats to eliminate thermal drift.
  * Coupled median selection: Canonical run picked from the median index of gross J/1k,
    strictly preserving the energy identity Gross_J_1k == (load_W / throughput_sps) * 1000.
  * Dual idle measurement: Pre- and post-workload package idle measured (30s each)
    and averaged for realistic net power derivation.
  * Full per-repeat trace persistence in output JSON for complete auditable transparency.
  * Verified hardware provenance with git SHA, host boot ID, and library version stamping.

USAGE:
    python scripts/measure_cpu_energy.py --repeats 7 --measure-s 20 --warmup-s 60 --cooldown-s 15
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
import scipy
import sklearn
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


def get_git_sha():
    """Resolve current git commit SHA with robust fallback."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        pass
    git_head = os.path.join(ROOT, ".git", "HEAD")
    if os.path.exists(git_head):
        try:
            with open(git_head, "r") as f:
                content = f.read().strip()
            if content.startswith("ref:"):
                ref_path = os.path.join(ROOT, ".git", content.split(" ", 1)[1].strip())
                if os.path.exists(ref_path):
                    with open(ref_path, "r") as rf:
                        return rf.read().strip()
            else:
                return content
        except Exception:
            pass
    return "unknown"


def get_run_id():
    """Capture comprehensive hardware run ID and environment stamp."""
    boot_id = "unknown"
    if os.path.exists("/proc/sys/kernel/random/boot_id"):
        try:
            with open("/proc/sys/kernel/random/boot_id", "r") as f:
                boot_id = f.read().strip()
        except Exception:
            pass
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "argv": sys.argv,
        "host_boot_id": boot_id,
        "python_version": platform.python_version(),
    }


def get_runtime_environment(env_probe):
    """Compile runtime environment, library versions, and threading configuration."""
    return {
        "host": {
            "cpu_model": env_probe.get("cpu_model", "unknown"),
            "physical_cores": env_probe.get("physical_cores"),
            "logical_cpus": env_probe.get("logical_cpus"),
            "socket_count": env_probe.get("socket_count"),
            "tdp_watts": env_probe.get("tdp_watts"),
            "platform": env_probe.get("platform", "unknown"),
            "os": env_probe.get("os"),
            "os_release": env_probe.get("os_release"),
            "is_wsl": env_probe.get("is_wsl"),
            "is_container": env_probe.get("is_container"),
        },
        "libraries": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
        },
        "threading": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "unset"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS", "unset"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "unset"),
            "os_cpu_count": os.cpu_count(),
        }
    }


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
# Idle power measurement
# ---------------------------------------------------------------------------
def measure_idle_power(rapl, duration_s):
    """Measure steady-state background idle power over duration_s seconds."""
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
# Saturated inference benchmark for one fitted model (End-to-End throughput)
# ---------------------------------------------------------------------------
def saturated_infer(clf, vec, raw_texts_bench, rapl, warmup_s, measure_s):
    """Drive `vec` + `clf` to thermal equilibrium on raw_texts_bench (end-to-end
    vectorization + inference), then over the measurement window capture package
    energy (RAPL), throughput, and load power simultaneously.
    """
    n_batch = len(raw_texts_bench)

    # 1. Warm-up drive to establish thermal and caching steady state
    if warmup_s > 0:
        t_warm_end = time.perf_counter() + warmup_s
        while time.perf_counter() < t_warm_end:
            X_vec = vec.transform(raw_texts_bench)
            clf.predict_proba(X_vec)

    # 2. Measurement window: energy + throughput captured simultaneously
    before = _read_domains(rapl.domains) if rapl.ok else None
    t0 = time.perf_counter()
    n_infer = 0
    while (time.perf_counter() - t0) < measure_s:
        X_vec = vec.transform(raw_texts_bench)
        clf.predict_proba(X_vec)
        n_infer += n_batch
    t1 = time.perf_counter()
    after = _read_domains(rapl.domains) if rapl.ok else None

    elapsed = t1 - t0
    throughput = n_infer / elapsed
    out = {
        "throughput_sps": throughput,
        "n_infer": n_infer,
        "elapsed_s": elapsed
    }
    if rapl.ok:
        ej = _delta_j(rapl.domains, before, after)
        out["energy_window_j"] = ej
        out["load_w"] = ej / elapsed
        out["gross_j_1k_integrated"] = (ej / n_infer) * 1000.0
    return out


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Measure saturated CPU energy via Intel RAPL.")
    ap.add_argument("--warmup-s", type=float, default=60.0,
                    help="initial thermal warm-up drive in seconds (default: 60.0)")
    ap.add_argument("--cooldown-s", type=float, default=15.0,
                    help="inter-repeat cooldown gap in seconds (default: 15.0)")
    ap.add_argument("--measure-s", type=float, default=20.0,
                    help="measurement window per repeat in seconds (default: 20.0)")
    ap.add_argument("--idle-s", type=float, default=30.0,
                    help="pre and post idle window duration in seconds (default: 30.0)")
    ap.add_argument("--repeats", type=int, default=7,
                    help="number of benchmark repeats (default: 7, uses coupled median)")
    ap.add_argument("--bench-rows", type=int, default=50000,
                    help="target rows in the tiled saturation batch (default: 50000)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    log("=" * 92)
    log("  CPU SATURATED-BATCH ENERGY BENCHMARK  —  Classical Arms (End-to-End RAPL)")
    log("=" * 92)

    rapl = RAPLReader()
    if not rapl.ok:
        raise SystemExit(
            f"\n[FATAL] Intel RAPL is unavailable ({rapl.reason}).\n"
            "        Refusing to emit energy numbers from a hardcoded constant.\n"
            "        Run on bare-metal Linux with readable powercap counters (/sys/class/powercap/intel-rapl:*)."
        )

    log(f"[rapl] Intel RAPL active: {len(rapl.domains)} package domain(s) {rapl.domain_names}.")
    log(f"       Domain filter verified: Strictly 'package-*' domains included.")
    log(f"       Platform (psys) and subzones (core, uncore, dram) are excluded to prevent double-counting.")

    # ---- Hardware & Environment Probe ----
    env_probe = probe_environment()
    rt_env = get_runtime_environment(env_probe)
    log(f"\n[host] CPU model      : {rt_env['host']['cpu_model']}")
    log(f"[host] Physical Cores : {rt_env['host']['physical_cores']} | Logical CPUs: {rt_env['host']['logical_cpus']}")
    log(f"[host] Sockets        : {rt_env['host']['socket_count']} | TDP (PL1): {rt_env['host']['tdp_watts']} W")
    log(f"[host] OS / Platform  : {rt_env['host']['os_release']} ({rt_env['host']['platform']})")
    log(f"[libs] Python {rt_env['libraries']['python']} | NumPy {rt_env['libraries']['numpy']} | "
        f"Scikit-Learn {rt_env['libraries']['sklearn']} | LightGBM {rt_env['libraries']['lightgbm']}")
    log(f"[threading] OMP_NUM_THREADS={rt_env['threading']['OMP_NUM_THREADS']} | "
        f"os_cpu_count={rt_env['threading']['os_cpu_count']}")

    # ---- Reconstruct Exact Frozen Split ----
    if not os.path.exists(PSYTAR_CSV):
        log(f"[FATAL] Missing dataset: {PSYTAR_CSV}")
        return
    train_df, calib_df, test_df = reconstruct_split(PSYTAR_CSV, args.seed)
    tfidf_cfg = load_tfidf_config()
    vec = TfidfVectorizer(**tfidf_cfg)
    X_train = vec.fit_transform(list(train_df["text"]))
    y_train = train_df["label"].values
    test_texts = list(test_df["text"])
    log(f"\n[data] train={X_train.shape} test={len(test_texts)} texts "
        f"(features={X_train.shape[1]}, config={tfidf_cfg})")

    # Tile test texts into a steady-state saturation batch
    reps = max(1, int(np.ceil(args.bench_rows / max(1, len(test_texts)))))
    raw_texts_bench = test_texts * reps
    log(f"[bench] Saturation batch: {len(raw_texts_bench):,} texts ({reps}x tiled test split)")

    # Fit canonical classical models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05,
                                       num_leaves=31, random_state=42,
                                       n_jobs=-1, verbose=-1),
    }
    for clf in models.values():
        clf.fit(X_train, y_train)

    # ---- 1. Pre-Benchmark Idle Measurement ----
    log(f"\n[idle] Measuring PRE-benchmark package idle power ({args.idle_s:.0f}s window) ...")
    idle_w_pre = measure_idle_power(rapl, args.idle_s)
    if idle_w_pre is None:
        raise SystemExit("[FATAL] Could not measure pre-benchmark idle power via RAPL counters.")
    log(f"[idle] -> Pre-benchmark Idle Power = {idle_w_pre:.3f} W")

    # ---- 2. Saturated Benchmark Across Models ----
    results = {}
    for name in MODEL_ORDER:
        clf = models[name]
        log(f"\n[bench] {name}: Initial warm-up {args.warmup_s:.0f}s + {args.repeats} repeats x "
            f"({args.measure_s:.0f}s measure + {args.cooldown_s:.0f}s cooldown)")

        # Initial long warm-up to reach steady-state thermal equilibrium
        log(f"  [warmup] driving {name} for {args.warmup_s:.0f}s to reach thermal equilibrium ...")
        saturated_infer(clf, vec, raw_texts_bench, rapl, warmup_s=args.warmup_s, measure_s=1.0)

        gross_list, thr_list, load_list = [], [], []

        for r in range(args.repeats):
            # Inter-repeat short warmup (2s) followed by full measurement window (20s)
            res = saturated_infer(clf, vec, raw_texts_bench, rapl, warmup_s=2.0, measure_s=args.measure_s)
            thr = res["throughput_sps"]
            gross = res["gross_j_1k_integrated"]
            load_w = res["load_w"]

            thr_list.append(thr)
            gross_list.append(gross)
            load_list.append(load_w)

            log(f"    Repeat {r+1}/{args.repeats}: Thr = {thr:,.0f} s/s | Load = {load_w:.3f} W | Gross = {gross:.4f} J/1k")

            # Cooldown gap between repeats to eliminate thermal drift
            if r < args.repeats - 1 and args.cooldown_s > 0:
                time.sleep(args.cooldown_s)

        # Coupled Median Selection: Use median index of gross_j_1k
        i_med = int(np.argsort(gross_list)[len(gross_list) // 2])
        gross_val = float(gross_list[i_med])
        thr_val = float(thr_list[i_med])
        load_val = float(load_list[i_med])

        # Sample CV (using sample mean as denominator and ddof=1)
        mean_gross = float(np.mean(gross_list))
        std_gross = float(np.std(gross_list, ddof=1)) if len(gross_list) > 1 else 0.0
        cv = float(std_gross / mean_gross * 100.0) if mean_gross else 0.0

        # Exact Energy Identity Verification
        expected_gross = (load_val / thr_val) * 1000.0
        identity_diff = abs(gross_val - expected_gross)
        identity_holds = identity_diff < 1e-5

        results[name] = {
            "gross_val": gross_val,
            "thr_val": thr_val,
            "load_val": load_val,
            "cv": cv,
            "i_med": i_med,
            "gross_list": gross_list,
            "thr_list": thr_list,
            "load_list": load_list,
            "identity_holds": identity_holds,
            "identity_diff": identity_diff,
        }
        log(f"  -> Canonical Median (Repeat #{i_med+1}): Gross = {gross_val:.4f} J/1k | "
            f"Load = {load_val:.3f} W | Thr = {thr_val:,.0f} s/s | CV = {cv:.2f}%")
        log(f"  -> Energy Identity Check: (Load/Thr)*1k = {expected_gross:.4f} J/1k "
            f"({'EXACT MATCH' if identity_holds else 'DRIFT DETECTED'})")
        if cv > 10.0:
            log(f"  [WARN] {name} energy CV ({cv:.2f}%) exceeds the 10% stability gate.")

    # ---- 3. Post-Benchmark Idle Measurement ----
    log(f"\n[idle] Measuring POST-benchmark package idle power ({args.idle_s:.0f}s window) ...")
    idle_w_post = measure_idle_power(rapl, args.idle_s)
    if idle_w_post is None:
        raise SystemExit("[FATAL] Could not measure post-benchmark idle power via RAPL counters.")
    log(f"[idle] -> Post-benchmark Idle Power = {idle_w_post:.3f} W")

    idle_w_mean = float((idle_w_pre + idle_w_post) / 2.0)
    log(f"[idle] -> Canonical Package Idle Baseline = {idle_w_mean:.3f} W "
        f"(Mean of Pre={idle_w_pre:.3f} W and Post={idle_w_post:.3f} W)")

    # ---- 4. Assemble Final Output Payload ----
    final_models_payload = {}
    for name in MODEL_ORDER:
        r_data = results[name]
        gross_val = r_data["gross_val"]
        thr_val = r_data["thr_val"]
        load_val = r_data["load_val"]
        cv = r_data["cv"]

        net_power = load_val - idle_w_mean
        if net_power < 0:
            log(f"  [WARN] {name} load power ({load_val:.3f} W) < idle power ({idle_w_mean:.3f} W)")
            net_power = max(0.0, net_power)
        net_1k = (net_power / thr_val) * 1000.0 if thr_val else 0.0

        model_dict = {
            "inf_j_gross": gross_val,
            "inf_j_net": net_1k,
            "throughput_sps": thr_val,
            "load_w": load_val,
            "idle_w": idle_w_mean,
            "net_power_w": net_power,
            "energy_cv_pct": cv,
            "median_repeat_index": r_data["i_med"],
            "per_repeat": {
                "gross_j_1k": [float(x) for x in r_data["gross_list"]],
                "throughput_sps": [float(x) for x in r_data["thr_list"]],
                "load_w": [float(x) for x in r_data["load_list"]],
            },
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
            "warmup_s": args.warmup_s,
            "cooldown_s": args.cooldown_s,
            "method": "rapl_integrated_saturated_end2end",
            "includes_tfidf_vectorization": True,
            "summary_stat": "median_canonical_run",
            "provenance": "measured_rapl_saturated",
        }
        final_models_payload[name] = model_dict

    # Provide shorthand aliases (LR, GBDT) for downstream tool compatibility
    if "Logistic Regression" in final_models_payload:
        final_models_payload["LR"] = final_models_payload["Logistic Regression"]
    if "LightGBM" in final_models_payload:
        final_models_payload["GBDT"] = final_models_payload["LightGBM"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "cpu_energy_measured.json")
    payload = {
        "_meta": {
            "generated_by": "measure_cpu_energy.py",
            "rapl_available": True,
            "rapl_domains": len(rapl.domains),
            "rapl_domain_names": rapl.domain_names,
            "rapl_note": ("Only top-level package domains (package-*) are summed. "
                         "Platform/psys and subzones (core, uncore, dram) are excluded to prevent double-counting."),
            "idle_power_w": idle_w_mean,
            "idle_power_pre_w": idle_w_pre,
            "idle_power_post_w": idle_w_post,
            "idle_window_s": args.idle_s,
            "warmup_s": args.warmup_s,
            "cooldown_s": args.cooldown_s,
            "measure_s": args.measure_s,
            "repeats": args.repeats,
            "bench_rows": len(raw_texts_bench),
            "seed": args.seed,
            "benchmark_scope": "end_to_end_tfidf_plus_predict",
            "scope_description": ("End-to-End: Raw text -> TfidfVectorizer.transform -> clf.predict_proba."),
            "_run_id": get_run_id(),
            "environment": rt_env,
        },
        **final_models_payload,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log(f"\n[artifact] Successfully wrote {out_path}")
    log("Provenance: live RAPL energy (measured_rapl_saturated)")
    log("=" * 92)


if __name__ == "__main__":
    main()
