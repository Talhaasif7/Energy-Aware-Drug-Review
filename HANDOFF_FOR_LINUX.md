# Run instructions — CPU pipeline, Linux host

Thank you for running this. It needs **no GPU**, **no internet**, and about
**20 minutes**. Everything is CPU-only.

## What this is, in one paragraph

This is the CPU half of an energy-measurement study on adverse-drug-reaction
detection. Two transformer models were already trained once on a Colab T4 and
their predictions ship inside this package as `.npz` files, so nothing here needs
a GPU. What is still missing is a **trustworthy CPU energy measurement**. On
Windows there is no way to read the processor's energy counters, so the current
figures are a software estimate. On Linux, `/sys/class/powercap/intel-rapl:*`
exposes the CPU's own integrated energy counter (Intel RAPL), which turns the
estimate into a measurement. That is the one thing this run provides that we
cannot get any other way.

## Requirements

- **Linux booted on the hardware itself.** WSL, a virtual machine, or a Docker
  container will not work for the energy part — none of them pass the RAPL
  counters through. The pipeline still runs there, it just cannot measure power,
  which defeats the purpose. The pre-flight check in step 2 detects this and
  tells you.
- **An Intel CPU** (or AMD with the powercap RAPL driver bound).
- **Python 3.10–3.12.**
- ~2 GB free disk space.

## Step 1 — install the packages

```bash
cd /path/to/this/folder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 2 — pre-flight check (about two seconds)

```bash
python3 scripts/preflight_linux.py
```

This verifies the Python version, every package, every input file (by SHA-256, so
we know the data is identical to ours), and whether the energy counters are
readable. It prints a `PASS` / `WARN` / `FAIL` line per check plus the exact fix
for anything that fails. **Please do not skip it** — it takes seconds and catches
in advance the things that would otherwise waste a whole run.

If it reports RAPL as unreadable, that is expected on most distributions: since
CVE-2020-8694 the counters ship root-only. One command fixes it:

```bash
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
python3 scripts/preflight_linux.py          # confirm it now says PASS
```

(That grants read access to a power counter. It reverts on reboot. If you would
rather not, running the next step under `sudo` works too, though then remember to
use the venv's interpreter explicitly.)

## Step 3 — run it

```bash
bash run_on_linux.sh
```

**Please leave the machine otherwise idle while this runs** — no browser, no
IDE, no other jobs. This measures power draw, so background load contaminates the
result directly. On our Windows host two runs of the identical benchmark
disagreed by 30% for exactly this reason, which is part of why we are asking for
a clean Linux run. The script prints the load average before it starts; near zero
is what we want.

It runs the pipeline, then writes everything into `handoff_out/` and packs it as
`handoff_results_<timestamp>.tar.gz` in the folder root.

## Step 4 — send back one file

```
handoff_results_<timestamp>.tar.gz
```

That archive already contains the artifacts, the machine's hardware details, the
resolved package versions, SHA-256 digests of every output, and the complete raw
console log. Nothing else is needed.

**Please send the log file as-is rather than pasting or summarising it.** A
reformatted summary of an earlier run turned out to contain accuracy figures that
no run had actually produced, which cost a round of work to detect. The raw file
avoids that entirely.

If something fails partway, **please send the archive anyway** — the log inside
it is exactly what is needed to diagnose the problem.

## One thing to avoid

Do not open the `.csv` files in Excel or LibreOffice and save them. That rewrites
line endings and changes their checksums, and the pre-flight will then correctly
refuse to proceed because the data no longer matches the published numbers.

## If you want to know what actually runs

`RUN_ORDER.md` documents every stage. In short, `run_on_linux.sh` calls:

1. `scripts/preflight_linux.py` — the environment checks above.
2. `scripts/run_all_cpu.py` — which in turn runs the saturated-batch CPU energy
   benchmark, reconstructs the frozen 1,201-sentence test split from the
   transformer `.npz` files, retrains the two classical baselines
   (logistic regression, LightGBM) on that exact split, recomputes all
   discrimination and calibration metrics with a paired bootstrap, and produces
   the model-selection and budget tables.
3. `scripts/minimal_pipeline_st3.py` — measures training time and training
   energy, which the budget extrapolation depends on.

Every number each stage produces is written to a JSON file rather than only
printed, so nothing has to be transcribed by hand.
