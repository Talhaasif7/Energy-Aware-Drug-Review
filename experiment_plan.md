# Experiment Closure Plan — ECC-MS / Energy-Aware Drug-Review NLP

**Audit basis:** full clone of `Talhaasif7/Energy-Aware-Drug-Review` @ main.
Metrics independently recomputed from `results/*.npz`; JSONs parsed directly;
`scripts/measure_cpu_energy.py`, `scripts/rapl_utils.py`,
`scripts/run_frozen_split_analysis.py` read line by line.

**Verdict: you are ~2 working days from closing the experiment phase.** One
measurement is invalid, one script has a config split-brain, and the README is
hand-typed. Nothing else is wrong.

---

## 1. Why you have been looping

There are three layers in this project. Two are sound. Every round has been
spent on the third.

| Layer | State | Evidence |
| :--- | :--- | :--- |
| **Modelling & metrics** | ✅ Correct, verified independently | I recomputed AUROC / AUPRC / adaptive ECE for all 12 arms straight from the `.npz` files. Every seed-42 number in the README reproduces. |
| **Selection logic (ECC-MS)** | ✅ Correct and self-consistent | `st8_regime_reconciled.json` is internally coherent; feasible counts, tie tests, and the τ=0.03 empty set all check out. |
| **Energy + document assembly** | ❌ Broken | Energy rests on a hardcoded constant from an undocumented host. The README is transcribed by hand and now disagrees with its own source-of-truth JSON in 4 of 8 rows. |

**The loop mechanism:** each round you fix a number in the README by hand. The
JSON does not change, or changes separately. Next round a different cell
disagrees. Nine rounds of this is not a research problem — it is a build
problem. Section 4 fixes it permanently.

### Verified — stop re-checking these

Recomputed from `results/*.npz`, matching README to ±0.002:

```
LR      uncal  AUROC 0.8760  AUPRC 0.8125  ECE 0.0642   CADEC AUROC 0.8379  ECE 0.0841
LR      temp   AUROC 0.8760  AUPRC 0.8125  ECE 0.0448
LR      iso    AUROC 0.8742  AUPRC 0.7933  ECE 0.0296
GBDT    uncal  AUROC 0.8627  AUPRC 0.8011  ECE 0.0198   CADEC AUROC 0.7989  ECE 0.0505
DistilBERT uncal AUROC 0.9181 AUPRC 0.8760 ECE 0.0716   CADEC AUROC 0.9042  ECE 0.0652
PubMedBERT uncal AUROC 0.9276 AUPRC 0.8885 ECE 0.0814   CADEC AUROC 0.9191  ECE 0.0583
```

The frozen split (`test_N=1201`, CADEC `N=7823`), the `.npz` embedded-text split
recovery, the paired bootstrap, the OOD tie gate, and the τ=0.03 empty feasible
set are all sound. **These are finished. Do not touch them again.**

---

## 2. Defect 1 — CPU energy is measured on a host that cannot produce those numbers

### The finding

`scripts/measure_cpu_energy.py`, lines 74–75:

```python
ST2_POWER = {
    "Logistic Regression": {"load_w": 157.090, "throughput_sps": 54465.0, ...},
    "LightGBM":            {"load_w": 231.960, "throughput_sps": 34954.0, ...},
}
```

These are hardcoded, named `ST2_POWER`, and used whenever RAPL is unavailable.

`results/cpu_energy_measured.json` — the file ST6, ST8 and the README all read —
declares:

```json
"rapl_available": false,
"idle_source": "ST2_constant",
"provenance": "measured_throughput_x_ST2_power",
"host": { "cpu_model": "Intel64 Family 6 Model 142 ...",
          "physical_cores": 4, "logical_cpus": 8,
          "platform": "Windows-11-10.0.26200-SP0" }
```

**Intel Family 6 Model 142 is a mobile U-series part** (Kaby Lake-U / Whiskey
Lake-U class, e.g. i5-8265U / i7-8565U), 4 cores / 8 threads, **15 W nominal TDP,
~44 W PL2 burst ceiling.** A sustained package draw of 157 W — let alone 232 W —
is physically impossible on it.

### Three hosts, silently mixed

| Host | Where it appears | Idle | LightGBM load | Cores |
| :--- | :--- | :---: | :---: | :---: |
| A — ST2 Linux box | `reports/st2_energy_sanity_report.md` | 6.734 W | **9.94 W** | 6 / 6 |
| B — Windows laptop | `cpu_energy_measured.json`, `st3_cpu_energy.json` | — | no RAPL | 4 / 8 |
| C — unidentified | `cpu_energy_measured_v2.json` (no `host` block) | 8.650 W | **231.96 W** | unknown |

The constants named `ST2_POWER` are **not** from ST2 — ST2 measured 9.94 W, not
232 W. They came from host C, which has no provenance record anywhere in the
repo. They are then multiplied by host B's throughput.

So every energy figure, every ratio, ST6, and the ECC-MS `E` axis derive from a
power constant belonging to a machine that is not described in the repository.

### Why this is the whole ballgame

The ratio is *entirely* set by the CPU host's power draw:

| CPU load power | LR gross J/1k @ 54,465 s/s | DistilBERT ÷ LR |
| :--- | :---: | :---: |
| 15 W (edge / laptop, realistic) | 0.275 | **207×** |
| 25 W (laptop turbo) | 0.459 | 124× |
| 65 W (desktop) | 1.19 | 48× |
| 157 W (current, invalid) | 2.884 | 19.8× |

Your published 8.6–38× is not merely unverified — it is almost certainly **too
low by an order of magnitude**, because the phantom 157/232 W inflates the CPU
side. Correcting this likely restores a headline in the 100–250× range, which is
much closer to your original story. This is the one fix that gains you something
rather than costing you.

### Fix — do this first

1. **Delete the `ST2_POWER` fallback entirely.** Replace with a hard failure:

   ```python
   if not rapl.ok:
       raise SystemExit(
           f"RAPL unavailable ({rapl.reason}). "
           "Refusing to emit energy numbers from a hardcoded constant. "
           "Run on bare-metal Linux with readable powercap counters."
       )
   ```
   A missing measurement must stop the pipeline, never silently substitute.

2. **Run `measure_cpu_energy.py` on bare-metal Linux on the machine you intend to
   describe as the deployment target.** `HANDOFF_FOR_LINUX.md` is already correct
   about the requirements (no WSL, no VM, no container). Use host A (the ST2 box)
   if it is still available — that keeps ST2, ST3 and the energy table on one
   machine and removes the last provenance gap.

3. **Emit the host block into `cpu_energy_measured.json` unconditionally**:
   CPU model string, physical cores, socket count, RAPL TDP (`constraint_0_power_limit_uw`),
   kernel, and the `rapl_domain_names` actually summed.

4. **Delete `cpu_energy_measured_v2.json`.** It has no host provenance and is the
   source of the phantom constants. One canonical energy file only.

5. **Report the ratio as host-conditional.** Add one sentence and one small table:
   *"The CPU–GPU energy ratio is a function of the CPU host's power envelope; we
   report it for the measured host and give the scaling relation."* This turns
   your biggest weakness into a stated, controlled parameter — and it is true.

**Acceptance:** `cpu_energy_measured.json` contains
`"provenance": "measured_rapl_saturated"`, a populated `host` block, non-empty
`rapl_domain_names`, and a load power consistent with the disclosed TDP.

---

## 3. Defect 2 — two different CPU models are being reported as one

`scripts/run_frozen_split_analysis.py`:

- lines 118 / 122 record `TfidfVectorizer(max_features=1000)` into
  `model_hyperparameters` — this matches the seed-42 `.npz`;
- **line 286** builds the multi-seed arms with
  `TfidfVectorizer(ngram_range=(1,2), max_features=2500)`.

So the seed-42 tables and the multi-seed table describe **different models**.
That is why LR uncalibrated in-domain ECE reads 0.0638 in one table and
0.0907 ± 0.0051 in the other — seed 42 sits 5.3 SD below its own mean, which is
impossible if it were in the set.

**Fix:** pick one vectorizer config, put it in `configs/default_config.json`, and
have *both* code paths read it from there. There must be no TF-IDF constructor
anywhere in the codebase with inline parameters. Then rerun and regenerate
`cpu_arms_seed42_predictions.npz`.

Recommendation: keep `ngram_range=(1,2), max_features=2500` — bigrams matter for
ADR phrases ("weight gain", "dry mouth") and it is the stronger baseline.

**Also:** `efficient_transformer_seed456_predictions.npz` **does not exist** in
`results/`, yet `multi_seed_metrics["DistilBERT + Uncalibrated"]` claims
`n_seeds: 3, seeds: [42, 123, 456]`. Either the file was never downloaded from
Colab or the aggregation silently proceeded on two seeds. Recover it or fix the
count — and add an assertion that fails loudly when a declared seed's artifact is
absent.

---

## 4. Defect 3 — the README is typed by hand. This is the actual loop.

The README disagrees with `st8_regime_reconciled.json` in half its ST8 rows:

| τ | E | JSON feasible | README feasible | JSON AUROC | README AUROC | JSON RQ4 | README RQ4 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.07 | 10 | **5** | 2 | **0.8786** | 0.8880 | **false** | ✅ |
| 0.10 | 10 | **5** | 2 | **0.8786** | 0.8880 | true | ✅ |
| 0.07 | 120 | 9 | 9 | **0.9276** | 0.9277 | true | ✅ |
| 0.10 | 120 | 11 | 11 | **0.9276** | 0.9277 | true | ✅ |

The JSON also contains two rows (τ=0.10 at E=150 and E=200) that never made it
into the README. And the τ=0.07/E=10 row is marked RQ4-safe in the README when
the JSON says it fails — that is a scientific claim inverted by transcription.

**This is why round 10 will find new errors unless you change the process.**

### The loop-breaker: generate the README

Write `scripts/render_readme.py`. It reads the JSONs and emits `README.md` from a
template with placeholders. Rules:

- Every number in the README comes from a JSON via the renderer.
- `README.md` gets a banner: `<!-- GENERATED by render_readme.py — DO NOT EDIT -->`.
- A CI check (or a pre-commit hook) reruns the renderer and fails if the output
  differs from the committed file.
- Prose paragraphs that state numbers (Insights, Deployment Framing) use
  placeholders too — that is where the stale `906–4,975×` figures survived three
  rounds.

Roughly 150 lines of Python. It permanently ends the class of error that has
consumed rounds 3 through 9.

**Acceptance:** `python scripts/render_readme.py && git diff --exit-code README.md`
returns clean.

---

## 5. Defect 4 — the multi-seed table refutes claims still in the prose

From `frozen_split_reconciled.json`, CADEC OOD ECE against τ = 0.07:

| Arm | CADEC ECE mean ± SD | vs τ=0.07 |
| :--- | :---: | :--- |
| PubMedBERT + Uncalibrated | 0.0794 ± 0.0303 | **violates** |
| LR + TempScale | 0.0828 ± 0.0080 | **violates** |
| LightGBM + TempScale | 0.0672 ± 0.0139 | passes on mean, violates within 1 SD |
| DistilBERT + Uncalibrated | 0.0664 ± 0.0054 | passes |
| LR + Isotonic | 0.0409 ± 0.0043 | passes |

Insight 3 still asserts *"all transformer arms hold ECE ≤ τ = 0.07 on CADEC"* and
*"the one headline arm that violates τ = 0.07 is uncalibrated LR."* Both are
seed-42 artifacts.

Note PubMedBERT's OOD ECE SD of **0.0303** — 38% of its own mean. The
highest-capacity arm has the least stable out-of-domain calibration.

**Fix:** rewrite Insight 3 around what the three seeds actually show:

> Out-of-domain calibration is seed-unstable, and instability scales with model
> capacity. PubMedBERT's CADEC ECE varies by ±0.030 across seeds — comparable to
> the entire τ budget — while isotonic-recalibrated LR is stable at 0.041 ± 0.004.
> Point-estimate τ-feasibility is therefore not a safe deployment criterion; the
> conservative upper-CI gate is required, not optional.

That is a stronger, more defensible claim than the one it replaces, and it
motivates machinery you have already built.

**Also:** compute every τ-safety verdict in ST8 from multi-seed means with CIs,
not from seed 42. And extend `multi_seed_metrics` to the recalibrated transformer
arms — those are the arms ST8 actually selects, and they are currently
uncalibrated-only in the summary.

---

## 6. Open items — decide, don't re-litigate

| Item | Decision needed | Recommendation |
| :--- | :--- | :--- |
| **drugsCom vs DrugLib** | Which is the secondary dev corpus? | `dev_drugscom_50k.csv` is committed; use it, retire DrugLib, delete `uci_druglib_harmonised.csv` from the pipeline. Verify its class proportions against raw ratings — they currently match WebMD to 4 s.f. (25.93/15.98/58.09), which cannot be coincidence. |
| **WebMD** | It is `.gitignore`d and absent | State it as external-validation-only, or drop it. Do not leave it in ST6's corpus counts while absent from ST1b. |
| **ST1b Alt A** | Identical to the chosen cutoff for both datasets | Replace with a genuine second variant or report one sensitivity variant honestly. |
| **δ (TOST margin)** | 0.0150 ≈ the observed CADEC effect (0.0149) | Circular as set. Justify δ on clinical grounds and fix it before comparison, or drop TOST and report the CI directly. |
| **Deployment framing** | 157 W server vs "clinical edge hardware" | Resolved by §2 if you measure on representative hardware. |
| **`cadec_harmonised.csv`** | Not in the repo; raw CADEC is | Commit it or document that `harmonise_st1.py` regenerates it — its SHA is pinned in provenance but the file is unreachable. |

---

## 7. Execution order

**Day 1 — CPU energy (blocks everything downstream)**
1. Delete `ST2_POWER` fallback; make RAPL absence fatal. (30 min)
2. Unify the TF-IDF config into `configs/default_config.json`. (30 min)
3. Boot bare-metal Linux; `python scripts/preflight_linux.py`; confirm RAPL readable. (30 min)
4. `python scripts/measure_cpu_energy.py --measure-s 20 --repeats 5` → new `cpu_energy_measured.json` with host block. (30 min)
5. Delete `cpu_energy_measured_v2.json`. (1 min)

**Day 1 — GPU top-up (one Colab/Kaggle session, ~30 min quota)**
6. Recover `efficient_transformer_seed456_predictions.npz`. Download **before** the session ends.

**Day 2 — regenerate and lock**
7. `python scripts/run_all_cpu.py` → all JSONs regenerate from the new energy + unified config.
8. Extend `multi_seed_metrics` to recalibrated arms; recompute ST8 τ-safety from multi-seed means + CIs.
9. Write `scripts/render_readme.py`; regenerate `README.md`; add the CI check.
10. Rewrite Insight 3 and the deployment framing (§5) as template prose.
11. Settle the six items in §6 and record each decision in the README.

**Total: two focused days.** Nothing on this list requires new experiments.

---

## 8. Stop condition

Declare the experiment phase closed when all five hold:

1. `cpu_energy_measured.json` has `provenance: measured_rapl_saturated`, a
   populated `host` block, and a load power consistent with that host's TDP.
2. Exactly one TF-IDF configuration exists in the codebase, read from config.
3. `python scripts/render_readme.py && git diff --exit-code README.md` is clean.
4. Every τ-safety verdict derives from multi-seed statistics, and every declared
   seed has a corresponding artifact on disk.
5. The six §6 decisions are recorded in the README as decisions, not open questions.

When those pass, stop measuring and start writing. Do not open another review
round to look for a better number.

---

## 9. What the paper actually says now

The energy asymmetry has moved three times and will move once more. It is not
the contribution. These are, and all three are already in your data:

1. **Strict calibration constraints can be infeasible under uncertainty.** At
   τ = 0.03, conservative upper-CI filtering returns an empty feasible set. A
   selection framework that can honestly return "nothing qualifies" is more
   credible than one that always answers.

2. **In-domain statistical ties do not survive distribution shift.** DistilBERT ≈
   PubMedBERT on PsyTAR (CI [−0.0014, +0.0207]) but PubMedBERT is significantly
   ahead on CADEC (CI [+0.0097, +0.0203]). The in-domain tie is partly an
   artifact of N=1,201 versus N=7,823 — which is exactly why the OOD gate is
   needed, and why the tie rule correctly changes no selection in this study.
   Report that as a designed negative result.

3. **Out-of-domain calibration is seed-unstable, worst at high capacity.**
   PubMedBERT ±0.030 versus recalibrated LR ±0.004. This is a deployment-risk
   finding with direct clinical relevance, and it justifies the conservative gate.

Lead with the framework and the OOD findings. Let the energy ratio be a
supporting measurement with an honest host-dependency caveat. That paper is
publishable at BMC MIDM or JMIR Medical Informatics. The one you have been
chasing — a clean three-order-of-magnitude green headline — was never going to
survive review, and you are better off without it.
