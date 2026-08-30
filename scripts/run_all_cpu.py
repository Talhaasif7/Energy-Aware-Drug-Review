#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_cpu.py  —  One-command CPU-side orchestrator (Round 5 rigorous overhaul).

WHAT THIS RUNS (in order, all on CPU — no GPU needed)
-----------------------------------------------------
  1. measure_cpu_energy.py          -> results/cpu_energy_measured.json
        Saturated-batch CPU energy for LR + LightGBM. Uses live Intel RAPL when
        available (Linux); otherwise measures throughput live and combines it with
        the documented ST2 package power (clearly tagged). NON-FATAL: if it fails,
        the analysis below falls back to the documented ST2/ST3 constants.
  2. run_frozen_split_analysis.py   -> results/frozen_split_reconciled.json  (+ cpu_arms_*.npz)
        THE core step. Recovers the exact frozen split from the transformer .npz,
        trains the classical arms on it, recomputes every metric, runs the paired
        bootstrap tie rule, and writes the single source of truth. CRITICAL.
  3. eccms_regime_st8.py            -> results/st8_regime_reconciled.json (+ reports/st8_regime_map.png)
        ST8 regime sweep, reported purely from the source-of-truth JSON. CRITICAL.
  4. budget_and_subgroup_st6_st7.py (optional; --skip-budget to omit)
        ST6 budget extrapolation (GPU energy now DERIVED from the Colab JSON) + ST7
        subgroup audit. Needs the PsyTAR Excel; harmless if energy is still PENDING.

PREREQUISITES (must exist in results/ BEFORE running this)
----------------------------------------------------------
  * results/efficient_transformer_seed42_predictions.npz   (from the Colab GPU run)
  * results/biomedical_transformer_seed42_predictions.npz  (from the Colab GPU run)
  * results/colab_transformer_gpu_results.json             (from the Colab GPU run)
  These come from scripts/colab_gpu_transformer_primary_adr.py, which you run ONCE
  on a Colab T4 (free tier) using UPLOADED datasets. Download those 3 files into
  results/ first. This orchestrator never touches the GPU.

USAGE
-----
    python scripts/run_all_cpu.py
    python scripts/run_all_cpu.py --skip-energy      # skip step 1 (use constants)
    python scripts/run_all_cpu.py --skip-budget      # skip step 4
    python scripts/run_all_cpu.py --measure-s 20     # longer CPU energy window

WHAT TO SEND BACK
-----------------
    results/frozen_split_reconciled.json      (source of truth for README numbers)
    results/st8_regime_reconciled.json        (ST8 tables)
    results/cpu_energy_measured.json          (CPU energy provenance)
    results/colab_transformer_gpu_results.json (the refreshed GPU JSON w/ saturated_*)
    the full console log of this run.
"""
from __future__ import annotations

import os
import sys
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(ROOT, "results")

REQUIRED_GPU_ARTIFACTS = [
    "efficient_transformer_seed42_predictions.npz",
    "biomedical_transformer_seed42_predictions.npz",
    "colab_transformer_gpu_results.json",
]


def log(msg=""):
    print(msg, flush=True)


def banner(title):
    log("\n" + "=" * 92)
    log(f"  {title}")
    log("=" * 92)


def check_prerequisites():
    missing = [f for f in REQUIRED_GPU_ARTIFACTS
               if not os.path.exists(os.path.join(RESULTS_DIR, f))]
    if missing:
        log("[PREREQ] Missing GPU artifacts in results/:")
        for f in missing:
            log(f"           - {f}")
        log("\n[PREREQ] Run scripts/colab_gpu_transformer_primary_adr.py on a Colab T4")
        log("         (free tier) with the UPLOADED datasets, then download the 3 files")
        log("         above into results/ and re-run this orchestrator.")
        return False
    log("[PREREQ] All required GPU artifacts present in results/.")
    return True


def run_step(script, extra_args=None, critical=True):
    """Run a script as a subprocess, streaming its output. Returns True on success."""
    script_path = os.path.join(HERE, script)
    cmd = [sys.executable, script_path] + (extra_args or [])
    banner(f"STEP: {script}  ({'CRITICAL' if critical else 'optional'})")
    log(f"[cmd] {' '.join(cmd)}\n")
    try:
        proc = subprocess.run(cmd, cwd=ROOT)
    except FileNotFoundError:
        log(f"[ERROR] Script not found: {script_path}")
        return False
    if proc.returncode != 0:
        log(f"\n[{'FATAL' if critical else 'WARN'}] {script} exited with "
            f"code {proc.returncode}.")
        return False
    log(f"\n[ok] {script} completed.")
    return True


def main():
    ap = argparse.ArgumentParser(description="CPU-side reconciliation orchestrator.")
    ap.add_argument("--skip-energy", action="store_true",
                    help="skip measure_cpu_energy.py (analysis uses ST2/ST3 constants)")
    ap.add_argument("--skip-budget", action="store_true",
                    help="skip budget_and_subgroup_st6_st7.py (ST6/ST7)")
    ap.add_argument("--measure-s", type=float, default=None,
                    help="CPU energy measurement window seconds (passed to step 1)")
    ap.add_argument("--repeats", type=int, default=None,
                    help="CPU energy repeats (passed to step 1)")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    banner("AI-GREEN  CPU RECONCILIATION PIPELINE  (Round 5)")
    log(f"[env] python   : {sys.version.split()[0]}")
    log(f"[env] repo root : {ROOT}")
    log(f"[env] results/  : {RESULTS_DIR}")

    if not check_prerequisites():
        sys.exit(2)

    # ---- Step 1: CPU energy (non-fatal) ----
    if args.skip_energy:
        log("\n[skip] Step 1 (CPU energy) skipped by flag; using documented constants.")
    else:
        e_args = []
        if args.measure_s is not None:
            e_args += ["--measure-s", str(args.measure_s)]
        if args.repeats is not None:
            e_args += ["--repeats", str(args.repeats)]
        ok_energy = run_step("measure_cpu_energy.py", e_args, critical=False)
        if not ok_energy:
            log("[warn] CPU energy step failed — the analysis will fall back to the "
                "documented ST2/ST3 constants (clearly tagged in provenance).")

    # ---- Step 2: frozen-split analysis (CRITICAL) ----
    if not run_step("run_frozen_split_analysis.py", critical=True):
        log("\n[FATAL] Core reconciliation failed. Fix the error above and re-run.")
        sys.exit(1)

    # ---- Step 3: ST8 regime sweep (CRITICAL) ----
    if not run_step("eccms_regime_st8.py", critical=True):
        log("\n[FATAL] ST8 reporting failed. Fix the error above and re-run.")
        sys.exit(1)

    # ---- Step 4: ST6/ST7 budget (optional) ----
    if args.skip_budget:
        log("\n[skip] Step 4 (ST6/ST7 budget) skipped by flag.")
    else:
        run_step("budget_and_subgroup_st6_st7.py", critical=False)

    # ---- Step 5: Render README (CRITICAL) ----
    if not run_step("render_readme.py", critical=True):
        log("\n[FATAL] README rendering failed.")
        sys.exit(1)

    # ---- final summary ----
    banner("DONE — artifacts to send back")
    produced = [
        "frozen_split_reconciled.json",
        "st8_regime_reconciled.json",
        "st6_st7_reconciled.json",
        "cpu_energy_measured.json",
        "colab_transformer_gpu_results.json",
        "cpu_arms_seed42_predictions.npz",
    ]
    for f in produced:
        p = os.path.join(RESULTS_DIR, f)
        mark = "OK " if os.path.exists(p) else "-- "
        log(f"  [{mark}] results/{f}")
    log("\n  Send the files marked OK above (plus this console log) back for the")
    log("  README numeric reconciliation (Task 7). Nothing in the README's energy")
    log("  section should be cited as measured until frozen_split_reconciled.json")
    log("  and the saturated colab_transformer_gpu_results.json are regenerated.")
    log("=" * 92)


if __name__ == "__main__":
    main()
