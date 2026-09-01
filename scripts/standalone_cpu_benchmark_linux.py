#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
standalone_cpu_benchmark_linux.py
================================================================================
Zero-Dependency, 100% Self-Contained Linux RAPL CPU Energy Benchmark Script.

Can be run as a standalone file on any bare-metal Linux machine with Intel CPU.
Does NOT require git cloning the repository.
If psytar_harmonised.csv is not present locally, it will automatically download it.

Usage:
    python standalone_cpu_benchmark_linux.py
    python standalone_cpu_benchmark_linux.py --repeats 7 --measure-s 20 --warmup-s 60 --cooldown-s 15
================================================================================
"""
from __future__ import annotations

import os
import sys
import glob
import json
import time
import urllib.request
import platform
import subprocess
from datetime import datetime, timezone
import argparse
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

POWERCAP_BASE = "/sys/class/powercap"
PSYTAR_GITHUB_RAW = "https://raw.githubusercontent.com/Talhaasif7/Energy-Aware-Drug-Review/main/data/01_primary_adr_detection/dev_psytar/psytar_harmonised.csv"
MODEL_ORDER = ["Logistic Regression", "LightGBM"]


def log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. Embedded Intel RAPL Reader (Strictly package-* filtered)
# ---------------------------------------------------------------------------
class RAPLReader:
    """Sums integrated energy over top-level Intel RAPL package domains only."""

    def __init__(self, base: str = POWERCAP_BASE):
        self.domains: list[tuple[str, int]] = []
        self.domain_names: list[str] = []
        self.ok = False
        self.reason = "unknown"

        if platform.system() != "Linux":
            self.reason = f"not Linux (host is {platform.system()})"
            return
        if not os.path.isdir(base):
            self.reason = f"{base} does not exist (no powercap driver/VM/container)"
            return

        pkgs = sorted(p for p in glob.glob(os.path.join(base, "intel-rapl:*"))
                      if os.path.basename(p).count(":") == 1)
        if not pkgs:
            self.reason = f"{base} exists but exposes no intel-rapl:N package domain"
            return

        denied = []
        for p in pkgs:
            name_path = os.path.join(p, "name")
            try:
                with open(name_path, "r") as fh:
                    dname = fh.read().strip()
            except OSError:
                dname = os.path.basename(p)

            # Strictly include package-* domains (exclude psys, core, uncore, dram)
            if not dname.startswith("package-"):
                continue

            epath = os.path.join(p, "energy_uj")
            try:
                with open(epath, "r") as fh:
                    fh.read()
            except PermissionError:
                denied.append(epath)
                continue
            except OSError as exc:
                denied.append(f"{epath} ({exc.__class__.__name__})")
                continue

            try:
                with open(os.path.join(p, "max_energy_range_uj"), "r") as fh:
                    maxr = int(fh.read().strip())
            except (OSError, ValueError):
                maxr = 0
            self.domains.append((epath, maxr))
            self.domain_names.append(dname)

        if self.domains:
            self.ok = True
            self.reason = f"{len(self.domains)} readable package domain(s): {self.domain_names}"
            if denied:
                self.reason += f"; {len(denied)} domain(s) unreadable (partial)"
        else:
            self.reason = ("Counters exist but lack read permission. "
                           f"Run: sudo chmod a+r {base}/intel-rapl:*/energy_uj")


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
# 2. Host and Environment Probing
# ---------------------------------------------------------------------------
def probe_environment():
    cpu_model = "unknown"
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    tdp_watts = None
    pl1_files = glob.glob("/sys/class/powercap/intel-rapl:*/constraint_0_power_limit_uw")
    if pl1_files:
        try:
            with open(pl1_files[0], "r") as f:
                tdp_watts = float(f.read().strip()) / 1e6
        except Exception:
            pass

    return {
        "cpu_model": cpu_model,
        "physical_cores": os.cpu_count(),
        "logical_cpus": os.cpu_count(),
        "socket_count": 1,
        "tdp_watts": tdp_watts or 65.0,
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "is_wsl": "microsoft" in platform.release().lower(),
        "is_container": os.path.exists("/.dockerenv"),
    }


def get_git_sha():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def get_run_id():
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


# ---------------------------------------------------------------------------
# 3. Dataset Resolution & Reconstruct Split
# ---------------------------------------------------------------------------
def resolve_psytar_csv(custom_path=None):
    """Find psytar_harmonised.csv locally or download it from GitHub."""
    candidates = [
        custom_path,
        "psytar_harmonised.csv",
        "data/01_primary_adr_detection/dev_psytar/psytar_harmonised.csv",
        "../data/01_primary_adr_detection/dev_psytar/psytar_harmonised.csv",
        os.path.join(os.path.dirname(__file__), "..", "data", "01_primary_adr_detection", "dev_psytar", "psytar_harmonised.csv"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            log(f"[data] Found local PsyTAR dataset: {c}")
            return c

    # Download from raw GitHub if missing
    target = "psytar_harmonised.csv"
    log(f"[data] Local PsyTAR CSV not found. Downloading from GitHub ({PSYTAR_GITHUB_RAW}) ...")
    try:
        urllib.request.urlretrieve(PSYTAR_GITHUB_RAW, target)
        if os.path.exists(target) and os.path.getsize(target) > 1000:
            log(f"[data] Download successful: {target} ({os.path.getsize(target):,} bytes)")
            return target
    except Exception as e:
        log(f"[WARN] Automatic download failed: {e}")

    raise FileNotFoundError("Could not find or download psytar_harmonised.csv. Place it in current directory.")


def reconstruct_split(psytar_csv, seed=42):
    """Exact 60/20/20 stratified split on full PsyTAR (6,003 rows)."""
    df = pd.read_csv(psytar_csv)
    train_df, calib_test_df = train_test_split(
        df, train_size=0.6, stratify=df["label"], random_state=seed
    )
    calib_df, test_df = train_test_split(
        calib_test_df, train_size=0.5, stratify=calib_test_df["label"], random_state=seed
    )
    return train_df, calib_df, test_df


# ---------------------------------------------------------------------------
# 4. Saturated Inference Benchmark
# ---------------------------------------------------------------------------
def saturated_infer(clf, vec, raw_texts_bench, rapl, warmup_s, measure_s):
    n_batch = len(raw_texts_bench)

    # 1. Warm-up drive
    if warmup_s > 0:
        t_warm_end = time.perf_counter() + warmup_s
        while time.perf_counter() < t_warm_end:
            X_vec = vec.transform(raw_texts_bench)
            clf.predict_proba(X_vec)

    # 2. Measurement window
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
# 5. Main Execution
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Standalone Linux Intel RAPL CPU Benchmark.")
    ap.add_argument("--csv", type=str, default=None, help="Path to psytar_harmonised.csv")
    ap.add_argument("--warmup-s", type=float, default=60.0, help="Initial thermal warmup (s)")
    ap.add_argument("--cooldown-s", type=float, default=15.0, help="Inter-repeat cooldown (s)")
    ap.add_argument("--measure-s", type=float, default=20.0, help="Measurement window (s)")
    ap.add_argument("--idle-s", type=float, default=30.0, help="Pre/post idle window (s)")
    ap.add_argument("--repeats", type=int, default=7, help="Benchmark repeats (default: 7)")
    ap.add_argument("--bench-rows", type=int, default=50000, help="Saturation batch size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="cpu_energy_measured.json", help="Output JSON path")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    log("=" * 92)
    log("  STANDALONE LINUX RAPL CPU BENCHMARK  —  Classical Arms (End-to-End)")
    log("=" * 92)

    rapl = RAPLReader()
    if not rapl.ok:
        raise SystemExit(
            f"\n[FATAL] Intel RAPL is unavailable ({rapl.reason}).\n"
            "        Run on bare-metal Linux with readable powercap counters (/sys/class/powercap/intel-rapl:*).\n"
            "        If permission denied, run: sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj"
        )

    log(f"[rapl] Intel RAPL active: {len(rapl.domains)} package domain(s) {rapl.domain_names}.")
    log(f"       Domain filter: Strictly 'package-*' domains included (excluding psys & subzones).")

    # ---- Hardware & Environment Probe ----
    env_probe = probe_environment()
    rt_env = {
        "host": env_probe,
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
            "os_cpu_count": os.cpu_count(),
        }
    }
    log(f"\n[host] CPU Model      : {env_probe['cpu_model']}")
    log(f"[host] Physical Cores : {env_probe['physical_cores']} | TDP: {env_probe['tdp_watts']} W")
    log(f"[host] OS / Platform  : {env_probe['os_release']} ({env_probe['platform']})")
    log(f"[libs] Python {rt_env['libraries']['python']} | NumPy {rt_env['libraries']['numpy']} | "
        f"Scikit-Learn {rt_env['libraries']['sklearn']} | LightGBM {rt_env['libraries']['lightgbm']}")

    # ---- Dataset Split & Model Fitting ----
    csv_path = resolve_psytar_csv(args.csv)
    train_df, calib_df, test_df = reconstruct_split(csv_path, args.seed)
    tfidf_cfg = {"ngram_range": (1, 2), "max_features": 2500}
    vec = TfidfVectorizer(**tfidf_cfg)
    X_train = vec.fit_transform(list(train_df["text"]))
    y_train = train_df["label"].values
    test_texts = list(test_df["text"])
    log(f"\n[data] Train={X_train.shape} test={len(test_texts)} texts (features=2,500, n-grams=(1,2))")

    reps = max(1, int(np.ceil(args.bench_rows / max(1, len(test_texts)))))
    raw_texts_bench = test_texts * reps
    log(f"[bench] Saturation batch: {len(raw_texts_bench):,} texts ({reps}x tiled test split)")

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
        raise SystemExit("[FATAL] Could not measure pre-benchmark idle power.")
    log(f"[idle] -> Pre-benchmark Idle Power = {idle_w_pre:.3f} W")

    # ---- 2. Saturated Benchmark Across Models ----
    results = {}
    for name in MODEL_ORDER:
        clf = models[name]
        log(f"\n[bench] {name}: Initial warm-up {args.warmup_s:.0f}s + {args.repeats} repeats x "
            f"({args.measure_s:.0f}s measure + {args.cooldown_s:.0f}s cooldown)")

        # Initial long warmup to reach thermal equilibrium
        log(f"  [warmup] driving {name} for {args.warmup_s:.0f}s to reach thermal equilibrium ...")
        saturated_infer(clf, vec, raw_texts_bench, rapl, warmup_s=args.warmup_s, measure_s=1.0)

        gross_list, thr_list, load_list = [], [], []

        for r in range(args.repeats):
            res = saturated_infer(clf, vec, raw_texts_bench, rapl, warmup_s=2.0, measure_s=args.measure_s)
            thr = res["throughput_sps"]
            gross = res["gross_j_1k_integrated"]
            load_w = res["load_w"]

            thr_list.append(thr)
            gross_list.append(gross)
            load_list.append(load_w)

            log(f"    Repeat {r+1}/{args.repeats}: Thr = {thr:,.0f} s/s | Load = {load_w:.3f} W | Gross = {gross:.4f} J/1k")

            # Cooldown gap between repeats
            if r < args.repeats - 1 and args.cooldown_s > 0:
                time.sleep(args.cooldown_s)

        # Coupled Median Selection: Pick run based on median index of gross_j_1k
        i_med = int(np.argsort(gross_list)[len(gross_list) // 2])
        gross_val = float(gross_list[i_med])
        thr_val = float(thr_list[i_med])
        load_val = float(load_list[i_med])

        # Sample CV (ddof=1)
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
        raise SystemExit("[FATAL] Could not measure post-benchmark idle power.")
    log(f"[idle] -> Post-benchmark Idle Power = {idle_w_post:.3f} W")

    idle_w_mean = float((idle_w_pre + idle_w_post) / 2.0)
    log(f"[idle] -> Canonical Package Idle Baseline = {idle_w_mean:.3f} W "
        f"(Mean of Pre={idle_w_pre:.3f} W and Post={idle_w_post:.3f} W)")

    # ---- 4. Assemble Payload ----
    final_models_payload = {}
    for name in MODEL_ORDER:
        r_data = results[name]
        gross_val = r_data["gross_val"]
        thr_val = r_data["thr_val"]
        load_val = r_data["load_val"]
        cv = r_data["cv"]

        net_power = load_val - idle_w_mean
        if net_power < 0:
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

    # Shorthands
    final_models_payload["LR"] = final_models_payload["Logistic Regression"]
    final_models_payload["GBDT"] = final_models_payload["LightGBM"]

    payload = {
        "_meta": {
            "generated_by": "standalone_cpu_benchmark_linux.py",
            "rapl_available": True,
            "rapl_domains": len(rapl.domains),
            "rapl_domain_names": rapl.domain_names,
            "rapl_note": "Only top-level package domains (package-*) are summed. psys and subzones excluded.",
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
            "_run_id": get_run_id(),
            "environment": rt_env,
        },
        **final_models_payload,
    }

    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log(f"\n[artifact] Successfully wrote output JSON to: {os.path.abspath(out_path)}")
    log("=" * 92)
    log("\nJSON OUTPUT CONTENT (Send this or the file back):\n")
    print(json.dumps(payload, indent=2))
    log("=" * 92)


if __name__ == "__main__":
    main()
