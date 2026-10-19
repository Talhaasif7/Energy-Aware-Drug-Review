# AIH Round 1: completed revision and external runs

The submitted `AIH Paper.docx` and original result artifacts are preserved. Returned Colab/Linux component measurements are validated and integrated into the analysis, manuscript draft and reviewer response. Complete-system energy remains unmeasured, so this is not a claim that every reviewer request has been fully met. Author review is necessary before resubmission.

Manuscript DOCX files and the generated reviewer-response letter are author-local materials. The professor's code handoff does not need them; author-side DOCX regeneration requires the supplied original `AIH Paper.docx`.

## Current status: 19 October 2026

The planned returned runs are complete. No new Colab/Linux component measurement is currently needed. The final paper retains the original PubMedBERT GPU baseline and excludes its extra CPU FP32/INT8 experiment at the author's scope instruction. Raw supplied transfer files remain local. The paper uses 15 common configurations over ten grouped splits, plus 12 CPU calibration configurations on primary seed 42 (27 primary configurations in total). CPU quantisation covers DistilBERT and BERT-tiny.

See [LINUX_RESULTS_REVIEW.md](reports/revision/LINUX_RESULTS_REVIEW.md), [revision_status.json](reports/revision/revision_status.json) and the numerical CSVs. Common-catalogue frequencies include all ten splits; the primary CPU-only grid uses a separate catalogue and budgets through 4,000 J/1,000. The initial PubMedBERT tokenizer CV failure and its passing follow-up are both retained. The author-local revised DOCX and response are regenerated from the returned results.

The remaining hardware limitation is actual complete-system energy: the available CPU-only Linux host and Colab runtime have no wall-power meter or measured Colab host-package energy. The unsupported clinical workload guarantee is withdrawn; no independent expert margin validation is invented. Remaining work is author review and a resubmission decision about the disclosed instrumentation limitation.

## Returned Colab artifacts — 15 October 2026

The author has returned `revision_predictions.zip` and `revision_colab_run.log`. The checkpoint ZIP is saved separately in the author's Google Drive. All 30 prediction files match the run-manifest hashes and the corpus texts, labels and grouped partitions. All three encoders completed the 15-cell GPU-board grid, with three repeats per cell; every cell passed the configured repeatability check. No new Colab run is needed for these completed experiments.

The common-catalogue numerical supplements cover fifteen arms across all ten grouped splits and 2,000 cluster-bootstrap draws; the primary CPU extension brings seed 42 to 27 arms. Numerical artifact checks pass for all ten splits. See [COLAB_RESULTS_REVIEW.md](reports/revision/COLAB_RESULTS_REVIEW.md). The revised DOCX and reviewer response have now been regenerated after Linux integration; see the current status above.

The Linux results have since been returned and integrated. [PROFESSOR_LINUX_STEPS.md](PROFESSOR_LINUX_STEPS.md) now serves as reproduction documentation; no repeat run is requested.

## Already completed locally

- Exact CADEC post identities recovered; all 7,823 texts and labels verified unchanged, 1,248 posts, 2,865 positives.
- 2,000 whole-review/post percentile bootstrap draws, exact tied-score AUROC, 10-bin ECE plus 5/10/15/20-bin sensitivity.
- Complete 11 × 15 constraint grid per split; monotonicity checks; ten δ values; complete selection on all ten common-catalogue grouped splits, with a separate 27-arm primary catalogue.
- Review/post workload at calibration-target sensitivities 85%, 90%, 95%, plus separate retrospective matched-sensitivity diagnostics; all arm pairs within each declared catalogue; common-catalogue summaries do not pool the extra primary CPU arms.
- Figures and manuscript tables regenerated with returned hardware results, correct captions, formal equations and scoped claims.
- Primary source-only selection; CADEC-assisted historical gate separately labelled. No fabricated clinical δ justification or whole-system energy.

Cluster intervals need not all be wider than row intervals. The original plan's universal-width acceptance rule is invalid. A δ fitted to these same outcomes would not be independent clinical justification. The original heuristic power/MDD claim is withdrawn. The original GPU board readings exclude host tokenization and cannot be added to another machine's RAPL readings and called a measured complete system.

## Reproduction step 1 — Aap ka Colab run (completed)

1. Open `AIH_Revision_Colab.ipynb` in Colab and select a **T4 GPU** runtime.
2. Run the cells and upload **`revision_colab_bundle.zip`** when asked (about 0.44 MB). It contains the scripts and both CSVs; no private GitHub login is needed.
3. Defaults train DistilBERT, PubMedBERT and compact BERT-tiny over **ten grouped splits** (42/123/456/1001–1007). Primary seed 42 saves weights and measures the full **5 batch sizes × 3 lengths** board-energy grid. Outputs are incremental; actual run time depends on the runtime and is not promised by the old plan.
4. Run the final download cell to save **`revision_predictions.zip`**, **`revision_checkpoints.zip`**, and **`revision_colab_run.log`** before disconnecting. The first ZIP contains lightweight prediction/energy results; the second contains trained weights and may be large. If the browser blocks downloads, use Colab's Files pane to download these files from `/content/aih_revision/`.
5. If quota is tight, set notebook `extra=["--seeds","42"]` for the first run. In the same retained session, resume with `extra=["--seeds","123","456","1001","1002","1003","1004","1005","1006","1007","--skip-sweep"]`. Existing files are reused. A new Colab runtime loses local files; restore prior outputs/checkpoints or rerun as needed. Do not overwrite completed outputs accidentally.

The compact model is **BERT-tiny (`prajjwal1/bert-tiny`)**, not TinyBERT. Tokenization is included in the new elapsed-time scope, but only GPU-board energy is measured. Host and wall energy remain unavailable and are not estimated.

## Reproduction step 2 — Professor ka Linux CPU run (completed)

Give the professor the code on the revision branch (or a copy of this workspace) **and** `revision_checkpoints.zip`. Extract the ZIP at the repository root: it must create `results/revision/checkpoints/efficient_transformer_seed42/`, `biomedical_transformer_seed42/`, and `compact_transformer_seed42/`. These large weights must be transferred separately; **do not commit them as normal GitHub files**.

Use an isolated environment:

```bash
python3 -m venv .venv-revision
source .venv-revision/bin/activate
python -m pip install -r requirements-revision.txt
python scripts/revision_linux.py --preflight
python -u scripts/revision_linux.py
```

Preflight must identify readable `package-*` RAPL counters on bare-metal Linux. If permissions deny access, the operator must arrange appropriate read access under the institution's policy. The script never substitutes TDP constants for missing measurements.

Default Linux runs measure tokenizer package energy for three tokenizers, then same-checkpoint FP32 and dynamic INT8 DistilBERT/BERT-tiny accuracy and raw-text pipeline package energy at **batch 64, length 128**, with seven repeats. They do not retrain encoders or assume a local GPU. This is a measured CPU-package comparison; RAM/whole-system power is not included.

An optional broader CPU operating grid is available if resources permit:

```bash
python -u scripts/revision_linux.py --batch-sizes 8 16 32 64 128 --lengths 64 128 256
```

The full CPU grid is much longer than the canonical run; the original plan's short time estimates are not guaranteed. Return **`revision_linux_results.zip`**. Keep the console log too. Raw per-repeat counts, RAPL readings and energy provenance are retained; CV failures are reported, not hidden.

## Reproduction step 3 — Return and integrate (completed)

Send back these three files:

1. `revision_predictions.zip` from Colab.
2. `revision_checkpoints.zip` from Colab (also needed by the professor).
3. `revision_linux_results.zip` from Linux.

Extract the result ZIPs at the repository root, then integrate explicitly:

```bash
python scripts/revision_analysis.py --seeds 42 123 456 1001 1002 1003 1004 1005 1006 1007 --predictions results/revision/gpu_predictions results/revision/cpu_encoder_predictions
python scripts/revision_workload.py
python scripts/render_revision.py
```

Explicit prediction directories supersede submitted predictions; chosen file paths and SHA-256 checksums are recorded. FP32/INT8 checkpoint comparisons exist only for the primary split, so selection frequencies are **not pooled across unequal candidate catalogues**. Unknown/missing energy is not treated as measured. Returned baseline results and measured scopes are integrated; author review remains necessary before resubmission.

## Remaining limitation that code cannot remove

Reviewer 2 requests actual **complete-system energy**. With a CPU-only Linux host, no wall-power meter, and unavailable Colab host counters, this part cannot be measured with the stated hardware. The response explicitly acknowledges it. The prepared component and scoped pipeline measurements improve the evidence but do **not** fully fulfil that instrumentation request. No independent expert validation of δ is available either; the unsupported clinical workload guarantee is withdrawn rather than invented.

## Reproduction and validation

```bash
python -B -m unittest discover -s tests -v
python scripts/revision_analysis.py
python scripts/revision_workload.py
python scripts/render_revision.py
python scripts/package_revision.py
```

`results/revision/` contains machine-readable outputs; `reports/revision/` contains figures, the response draft, positioning/formalisation and the status ledger. The legacy `run_all_cpu.py` reproduces the archived submission pipeline, not this corrected revision.
