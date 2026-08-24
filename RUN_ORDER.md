# RUN ORDER — Round 5 Rigorous Overhaul

This is the exact order to run things, **which machine each file runs on**, and
**what to send back**. There are only three buckets.

The golden rule that drives all of this: **energy, throughput and power must be
measured together in one saturated run** (that is the Round 5 fix), and **no number
goes into the README unless a run produced it.** The scripts enforce this — anything
not yet measured prints `PENDING`, never a hand-typed value.

---

## 🟦 BUCKET A — run on **Colab T4 GPU (free tier)**  ·  run **ONCE**

**File:** `scripts/colab_gpu_transformer_primary_adr.py`

This fine-tunes DistilBERT + PubMedBERT and runs the **saturated-batch energy
benchmark** (fixed padded batch driven to steady state; 100 ms `nvidia-smi` power
sampling + trapezoidal energy integration; power + throughput + energy captured
together; 3 repeats + CV). It needs a real GPU + `nvidia-smi`, so it only runs on
Colab.

**Important — uploaded data, NOT git clone (repo is private):**
1. Open the script in a Colab notebook on a **T4** runtime.
2. When prompted (or up front), **upload these two files** into the Colab session:
   - `psytar_harmonised.csv`
   - `cadec_harmonised.csv`
   The script searches the working dir / `content/` for filenames containing
   `psytar` and `cadec`; if missing it opens an upload dialog. It **does not clone
   the GitHub repo.**
3. Run all cells. `SMOKE_TEST_MODE` is already `False` (full run).

**Download these 3 files from Colab into your local `results/` folder:**
- `efficient_transformer_seed42_predictions.npz`
- `biomedical_transformer_seed42_predictions.npz`
- `colab_transformer_gpu_results.json`   ← now contains `saturated_*` keys

> The `.npz` files now embed the train/calib/test/CADEC **texts**, so the CPU side
> reproduces the *identical* frozen split by construction (no split drift).

---

## 🟩 BUCKET B — run on **Linux** (for real CPU energy)  ·  optional but recommended

**File:** `scripts/measure_cpu_energy.py`  (also invoked automatically by Bucket C)

This measures **CPU** inference energy for LR + LightGBM the same saturated way,
using **Intel RAPL** (`/sys/class/powercap/intel-rapl:*`), which is **Linux-only**.

- On **Linux with readable RAPL** → energy is truly measured
  (`provenance = measured_rapl_saturated`).
- On **Windows / no RAPL** → it still measures throughput live and combines it with
  the documented ST2 package power, tagged `measured_throughput_x_ST2_power`
  (honest: throughput is fresh, power is the earlier ST2 measurement).

You do **not** have to run this separately — Bucket C runs it as step 1. Run it
standalone only if you want to do the energy measurement on a dedicated Linux box:

```bash
python scripts/measure_cpu_energy.py --measure-s 20 --repeats 3
```

Produces `results/cpu_energy_measured.json`.

---

## 🟨 BUCKET C — run on **plain CPU** (any OS, incl. your Windows machine)

**One command does everything CPU-side:**

```bash
python scripts/run_all_cpu.py
```

It runs, in order:
1. `measure_cpu_energy.py`        → `results/cpu_energy_measured.json`   *(non-fatal; falls back to ST2/ST3 constants if RAPL isn't available)*
2. `run_frozen_split_analysis.py` → `results/frozen_split_reconciled.json` **(the source of truth)** + `results/cpu_arms_seed42_predictions.npz`
3. `eccms_regime_st8.py`          → `results/st8_regime_reconciled.json` + `reports/st8_regime_map.png`
4. `budget_and_subgroup_st6_st7.py` → ST6/ST7 tables (GPU energy derived from the Colab JSON)

Prerequisite: the **3 Bucket-A files must already be in `results/`.** The orchestrator
checks and stops with a clear message if any are missing.

Useful flags: `--skip-energy` (use constants), `--skip-budget`, `--measure-s 20`,
`--repeats 3`.

> **Best case:** run Bucket C **on a Linux machine** — then step 1 gets live RAPL
> energy *and* steps 2–4 run, all in one go, and Bucket B is already included.
> **Windows-only is fine too:** everything still runs; CPU energy is just tagged as
> `throughput_x_ST2_power` instead of live RAPL.

If you prefer to run them by hand instead of the orchestrator, the order is exactly
steps 1 → 2 → 3 → 4 above.

---

## 📦 WHAT TO SEND BACK

After Bucket A (Colab) **and** Bucket C (CPU) have both run, send me:

| File | From | Why I need it |
| :--- | :--- | :--- |
| `results/frozen_split_reconciled.json` | Bucket C step 2 | **the** source of truth — every README number reconciles to this |
| `results/st8_regime_reconciled.json` | Bucket C step 3 | ST8 regime + selection tables |
| `results/cpu_energy_measured.json` | Bucket B/C step 1 | CPU energy + its provenance tag |
| `results/colab_transformer_gpu_results.json` | Bucket A | GPU **saturated** energy (`saturated_*` keys) |
| the full **console log** of `run_all_cpu.py` | Bucket C | to verify split alignment + tie CIs |

(The two transformer `.npz` and `cpu_arms_seed42_predictions.npz` are handy but
optional — the JSONs above already carry everything needed for the README.)

Once I have these, I finish **Task 7**: reconcile the README (GPU energy table, the
264×/157×/136×/4,664× ratios, ST6 transformer energy, ST8 gross J/1k, break-even,
Wh/day, CADEC N, DrugLib raw-vs-non-empty count, transformer fitted T / calib NLL /
GPU repeatability CV, and the split-alignment confirmation) to the measured numbers —
replacing every `PENDING` with a value that traces to your run.

---

## Quick reference — file → machine

| Script | Machine | Notes |
| :--- | :--- | :--- |
| `colab_gpu_transformer_primary_adr.py` | **Colab T4 GPU** | uploaded CSVs, not git clone; run once |
| `measure_cpu_energy.py` | **Linux** (best) / any CPU | live RAPL on Linux; ST2-power fallback elsewhere |
| `run_all_cpu.py` | **plain CPU** (any OS) | runs the whole CPU pipeline; needs Bucket-A files in `results/` |
| `run_frozen_split_analysis.py` | plain CPU | core reconciliation (called by `run_all_cpu.py`) |
| `eccms_regime_st8.py` | plain CPU | ST8 reporter (called by `run_all_cpu.py`) |
| `budget_and_subgroup_st6_st7.py` | plain CPU | ST6/ST7 (called by `run_all_cpu.py`) |
