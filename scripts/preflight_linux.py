#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preflight_linux.py  —  Run this FIRST, before the pipeline.

Checks, in about two seconds, everything that could make the real run fail or
produce a number that cannot be published. Prints a PASS / WARN / FAIL line per
check with the exact remediation, then a verdict.

    python3 scripts/preflight_linux.py

Exit codes
    0  ready to run (possibly with warnings)
    1  at least one blocking failure — fix it before running the pipeline

Why a pre-flight at all: this package is run once, on someone else's machine, by
someone who did not write it. A silent failure discovered afterwards costs a
whole round trip. Every check below corresponds to something that has actually
gone wrong in this project at least once.

FAIL vs WARN
    FAIL  the pipeline will crash or produce unusable output.
    WARN  the pipeline will complete, but some number will be an estimate rather
          than a measurement — worth knowing before, not after.
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import hashlib
import importlib
import platform

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

sys.path.insert(0, HERE)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
GLYPH = {PASS: "[ PASS ]", WARN: "[ WARN ]", FAIL: "[ FAIL ]"}

# Packages the CPU pipeline actually imports. torch/transformers are NOT needed:
# the transformer arms were run once on a Colab T4 and arrive as .npz artifacts.
REQUIRED_PKGS = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("lightgbm", "lightgbm"),
    ("matplotlib", "matplotlib"),
    ("openpyxl", "openpyxl"),          # pandas needs it to read PsyTAR_dataset.xlsx
]
OPTIONAL_PKGS = [
    ("codecarbon", "codecarbon"),      # ST3 only; RAPL is preferred where present
]

# Input data. Paths are relative to the repo root.
REQUIRED_DATA = [
    ("data/01_primary_adr_detection/dev_psytar/psytar_harmonised.csv",
     "859fb25f0fb1854f", "PsyTAR dev corpus (6,003 sentences)"),
    ("data/01_primary_adr_detection/external_val_cadec/cadec_harmonised.csv",
     "87f88871fa9b07c7", "CADEC external validation (7,823 sentences)"),
    ("data/02_secondary_sentiment_scaling/dev_uci_drug_review/"
     "uci_druglib_harmonised.csv",
     "960d3f25de715bdb", "UCI DrugLib secondary corpus (4,107 rows)"),
    ("data/02_secondary_sentiment_scaling/external_val_webmd/webmd_harmonised.csv",
     None, "WebMD secondary corpus (320,093 rows)"),
    ("data/01_primary_adr_detection/dev_psytar/PsyTAR_dataset.xlsx",
     None, "PsyTAR raw workbook (ST7 subgroup audit)"),
]

# Transformer artifacts from the one-off Colab GPU run. Without these the frozen
# split cannot be reconstructed and the whole comparison is impossible.
REQUIRED_RESULTS = [
    ("results/efficient_transformer_seed42_predictions.npz",
     "DistilBERT predictions + frozen split texts"),
    ("results/biomedical_transformer_seed42_predictions.npz",
     "PubMedBERT predictions + frozen split texts"),
    ("results/colab_transformer_gpu_results.json",
     "GPU saturated-run energy + metrics"),
]

_rows: list[tuple[str, str, str, str]] = []   # (status, label, detail, fix)


def check(status, label, detail="", fix=""):
    _rows.append((status, label, detail, fix))
    line = f"{GLYPH[status]} {label}"
    if detail:
        line += f" — {detail}"
    print(line, flush=True)
    if fix and status != PASS:
        for ln in fix.strip().splitlines():
            print(f"          fix: {ln.strip()}", flush=True)


def sha16(path):
    """First 16 hex of SHA-256. Identical to csv_sha() in
    run_frozen_split_analysis.py, so the digests are directly comparable."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


# ---------------------------------------------------------------------------
def section(title):
    print(f"\n--- {title} " + "-" * max(0, 66 - len(title)))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 78)
    print("  PRE-FLIGHT CHECK  —  Energy-Aware Drug-Review CPU pipeline")
    print("=" * 78)
    print(f"  repo root : {ROOT}")

    # ---------------- interpreter ----------------
    section("Interpreter")
    v = sys.version_info
    if v < (3, 9):
        check(FAIL, "Python version", f"{platform.python_version()}",
              "Python 3.9 or newer is required (f-strings with =, dict merge, "
              "and modern scikit-learn wheels). Try: python3.11 -m venv .venv")
    elif v >= (3, 13):
        check(WARN, "Python version", f"{platform.python_version()}",
              "3.13+ is newer than this code was exercised on; if lightgbm or "
              "scikit-learn wheels fail to install, use 3.10–3.12 instead.")
    else:
        check(PASS, "Python version", platform.python_version())

    check(PASS, "Interpreter path", sys.executable)

    # ---------------- packages ----------------
    section("Packages")
    for mod, pipname in REQUIRED_PKGS:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            check(PASS, f"{pipname}", f"version {ver}")
        except Exception as exc:
            check(FAIL, f"{pipname}", f"import failed ({exc.__class__.__name__})",
                  f"pip install {pipname}")
    for mod, pipname in OPTIONAL_PKGS:
        try:
            m = importlib.import_module(mod)
            check(PASS, f"{pipname} (optional)",
                  f"version {getattr(m, '__version__', '?')}")
        except Exception:
            check(WARN, f"{pipname} (optional)", "not installed",
                  f"pip install {pipname} — only ST3 uses it, and where Intel RAPL "
                  "is readable RAPL is the preferred source anyway.")

    # ---------------- input data ----------------
    section("Input data")
    for rel, expect_sha, desc in REQUIRED_DATA:
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.exists(p):
            check(FAIL, rel, "missing",
                  f"{desc} was not found. Copy the complete data/ tree from the "
                  "handoff archive into the repo root.")
            continue
        size = os.path.getsize(p)
        if size == 0:
            check(FAIL, rel, "zero bytes",
                  "File exists but is empty — the copy or extraction truncated it.")
            continue
        if expect_sha is None:
            check(PASS, rel, f"{human(size)}, sha16={sha16(p)}")
            continue
        got = sha16(p)
        if got == expect_sha:
            check(PASS, rel, f"{human(size)}, sha16={got} (matches expected)")
        else:
            check(FAIL, rel, f"sha16={got}, expected {expect_sha}",
                  "This is NOT the same data file the published numbers were "
                  "computed on. Results would not be comparable. Re-copy it from "
                  "the handoff archive without opening or re-saving it — opening a "
                  "CSV in Excel and saving rewrites line endings and changes the "
                  "digest.")

    # ---------------- GPU artifacts ----------------
    section("Transformer artifacts from the Colab GPU run")
    for rel, desc in REQUIRED_RESULTS:
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.exists(p):
            check(FAIL, rel, "missing",
                  f"{desc}. The frozen 1,201-sentence test split is recovered from "
                  "these .npz files; without them nothing downstream can run.")
        elif os.path.getsize(p) == 0:
            check(FAIL, rel, "zero bytes", "Truncated during copy — re-extract.")
        else:
            check(PASS, rel, human(os.path.getsize(p)))

    # sanity-load the GPU JSON, since a truncated JSON passes a size check
    gj = os.path.join(RESULTS_DIR, "colab_transformer_gpu_results.json")
    if os.path.exists(gj):
        try:
            with open(gj, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            n = len((blob.get("results") or {})) if isinstance(blob, dict) else 0
            check(PASS, "GPU JSON parses", f"{n} result entries")
        except Exception as exc:
            check(FAIL, "GPU JSON parses", f"{exc.__class__.__name__}: {exc}",
                  "The file is corrupt or truncated. Re-copy it.")

    # ---------------- output writability ----------------
    section("Output")
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        probe = os.path.join(RESULTS_DIR, ".preflight_write_probe")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
        check(PASS, "results/ is writable", RESULTS_DIR)
    except Exception as exc:
        check(FAIL, "results/ is writable", f"{exc.__class__.__name__}: {exc}",
              "The pipeline writes its artifacts here. Fix permissions, or copy "
              "the repo somewhere writable (not a read-only mount).")

    try:
        free = shutil.disk_usage(ROOT).free
        if free < 500 * 1024 * 1024:
            check(FAIL, "Free disk space", human(free),
                  "Under 500 MB free. The run writes prediction .npz files and "
                  "figures; free some space first.")
        elif free < 2 * 1024 * 1024 * 1024:
            check(WARN, "Free disk space", human(free), "Under 2 GB; should be "
                  "enough, but tight.")
        else:
            check(PASS, "Free disk space", human(free))
    except Exception:
        check(WARN, "Free disk space", "could not determine")

    # ---------------- energy sensor ----------------
    section("Energy measurement (the reason for running on Linux)")
    try:
        from rapl_utils import probe_environment
    except Exception as exc:
        check(FAIL, "rapl_utils import", f"{exc.__class__.__name__}: {exc}",
              "scripts/rapl_utils.py is missing from the package — re-extract.")
        env = {}
    else:
        env = probe_environment()
        check(PASS, "Host", f"{env['platform']}")
        check(PASS, "CPU", f"{env['cpu_model']} ({env['logical_cpus']} logical)")

        if env["os"] != "Linux":
            check(WARN, "Operating system", env["os"],
                  "Intel RAPL exists only on Linux. The pipeline still runs, but "
                  "CPU energy will be throughput x a documented power constant — "
                  "an estimate. Running on Linux is the entire point of this "
                  "handoff.")
        if env["is_wsl"]:
            check(FAIL, "WSL detected", "running under Windows Subsystem for Linux",
                  "WSL2 does not pass the RAPL MSRs through, so energy here is NOT "
                  "measured — it is the same software estimate we already have. "
                  "Use a natively-booted Linux install (dual boot or a live USB), "
                  "or a bare-metal machine.")
        if env["is_container"]:
            check(WARN, "Container detected", "docker/lxc/kubepods cgroup",
                  "Energy counters may be absent, or shared with other tenants on "
                  "the host, which makes the reading unattributable. Bare metal is "
                  "strongly preferred.")

        if env["rapl_available"]:
            check(PASS, "Intel RAPL", env["rapl_reason"],
                  "")
            print("          -> CPU energy will be a REAL integrated sensor "
                  "reading. This is what we want.")
        else:
            check(WARN, "Intel RAPL", env["rapl_reason"],
                  "sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj\n"
                  "then re-run this pre-flight. If the path does not exist at all, "
                  "this host cannot measure CPU energy and the run will fall back "
                  "to the documented power constant (still useful, but it does not "
                  "give us the measurement we are after).")

    # ---------------- verdict ----------------
    fails = [r for r in _rows if r[0] == FAIL]
    warns = [r for r in _rows if r[0] == WARN]

    print("\n" + "=" * 78)
    print(f"  VERDICT: {len(fails)} failure(s), {len(warns)} warning(s)")
    print("=" * 78)
    if fails:
        print("  BLOCKING — do not run the pipeline until these are fixed:")
        for _, label, detail, _fix in fails:
            print(f"    * {label}" + (f" ({detail})" if detail else ""))
        print("\n  Re-run this pre-flight after fixing.")
        return 1

    rapl_ok = bool(env.get("rapl_available"))
    print("  Ready to run:")
    print("      bash run_on_linux.sh")
    if not rapl_ok:
        print("\n  NOTE: RAPL is unavailable, so CPU energy will be an ESTIMATE,")
        print("  not a measurement. The run is still worth doing (it validates the")
        print("  pipeline and the metrics), but it will not settle the energy")
        print("  question. See the warning above for how to enable RAPL.")
    if warns:
        print(f"\n  {len(warns)} warning(s) above — none blocking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
