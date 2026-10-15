# AIH Round 1: completed revision and external runs

The submitted `AIH Paper.docx` and original `results/*.json` / prediction arrays are preserved. Read `AIH Paper - Revised.docx` and `reports/revision/REVIEW_RESPONSE.md` for the current working revision. Hardware extensions are still pending; this is not a submission-ready claim that every requested experiment has been completed.

Manuscript DOCX files and the generated reviewer-response letter are author-local materials. The professor's code handoff does not need them; author-side DOCX regeneration requires the supplied original `AIH Paper.docx`.

## Already completed locally

- Exact CADEC post identities recovered; all 7,823 texts and labels verified unchanged, 1,248 posts, 2,865 positives.
- 2,000 whole-review/post percentile bootstrap draws, exact tied-score AUROC, 10-bin ECE plus 5/10/15/20-bin sensitivity.
- Complete 11 × 15 constraint grid per split; monotonicity checks; ten δ values; complete selection on seeds 42/123/456.
- Review/post workload at calibration-target sensitivities 85%, 90%, 95%, plus separate retrospective matched-sensitivity diagnostics; all 66 original-arm pairs.
- New measured-data figures with correct captions, formal selection equations, positioning comparison and scoped claims.
- Primary source-only selection; CADEC-assisted historical gate separately labelled. No fabricated clinical δ justification or whole-system energy.

Cluster intervals need not all be wider than row intervals. The original plan's universal-width acceptance rule is invalid. A δ fitted to these same outcomes would not be independent clinical justification. The original heuristic power/MDD claim is withdrawn. The original GPU board readings exclude host tokenization and cannot be added to another machine's RAPL readings and called a measured complete system.

## Step 1 — Aap ka Colab run

1. Open `AIH_Revision_Colab.ipynb` in Colab and select a **T4 GPU** runtime.
2. Run the cells and upload **`revision_colab_bundle.zip`** when asked (about 0.44 MB). It contains the scripts and both CSVs; no private GitHub login is needed.
3. Defaults train DistilBERT, PubMedBERT and compact BERT-tiny over **ten grouped splits** (42/123/456/1001–1007). Primary seed 42 saves weights and measures the full **5 batch sizes × 3 lengths** board-energy grid. Outputs are incremental; actual run time depends on the runtime and is not promised by the old plan.
4. Download **`revision_predictions.zip`** and **`revision_checkpoints.zip`** before disconnecting. The first contains lightweight prediction/energy results; the second contains trained weights and may be large.
5. If quota is tight, set notebook `extra=["--seeds","42"]` for the first run. In the same retained session, resume with `extra=["--seeds","123","456","1001","1002","1003","1004","1005","1006","1007","--skip-sweep"]`. Existing files are reused. A new Colab runtime loses local files; restore prior outputs/checkpoints or rerun as needed. Do not overwrite completed outputs accidentally.

The compact model is **BERT-tiny (`prajjwal1/bert-tiny`)**, not TinyBERT. Tokenization is included in the new elapsed-time scope, but only GPU-board energy is measured. Host and wall energy remain unavailable and are not estimated.

## Step 2 — Professor ka Linux CPU run

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

## Step 3 — Return and integrate

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

Explicit prediction directories supersede submitted predictions; chosen file paths and SHA-256 checksums are recorded. FP32/INT8 checkpoint comparisons exist only for the primary split, so selection frequencies are **not pooled across unequal candidate catalogues**. Unknown/missing energy is not treated as measured. Additional baseline results and energy scopes still need author review before the working-draft/pending labels and response letter are finalised.

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
