# Linux CPU experiment: completed run and reproduction

The returned Colab and Linux component experiments have been validated and integrated. **No repeat run is currently requested.** The final CPU extension uses DistilBERT and BERT-tiny in FP32/INT8, plus tokenizer-only measurements for all three encoder families. Original PubMedBERT GPU configurations remain in the paper; the separately returned PubMedBERT CPU experiment is outside the planned revision catalogue and is excluded from the paper and selection. Raw transfer files are preserved locally. See [LINUX_RESULTS_REVIEW.md](reports/revision/LINUX_RESULTS_REVIEW.md).

The commands below are for reproduction if needed. No GPU or wall-power meter is assumed; the measurement domain is CPU package only.

## Files to obtain

1. The full repository on the follow-up branch: https://github.com/Talhaasif7/Energy-Aware-Drug-Review/tree/aih-linux-results-paper-revision-20261019
2. `revision_checkpoints.zip`, shared separately by the author from Google Drive. This contains the trained seed-42 checkpoints; it is not a normal GitHub file.

The script is **`scripts/revision_linux.py`**. Keep the full repository because the script imports helper modules, configuration and corpus files.

## Commands

For a fresh checkout:

```bash
git clone --branch aih-linux-results-paper-revision-20261019 https://github.com/Talhaasif7/Energy-Aware-Drug-Review.git
cd Energy-Aware-Drug-Review
```

For an existing checkout, open a terminal in its repository root and select the follow-up branch:

```bash
git fetch origin
git switch aih-linux-results-paper-revision-20261019
git pull --ff-only
```

The follow-up branch contains the returned Colab results and these instructions; an older `main` checkout will not contain them until the follow-up PR is merged. Create an isolated Python environment:

```bash
python3 -m venv .venv-revision
source .venv-revision/bin/activate
python -m pip install -r requirements-revision.txt
```

Download the checkpoint ZIP from Drive into the repository root, then extract it:

```bash
python -m zipfile -e revision_checkpoints.zip .
```

Expected checkpoint directories:

```text
results/revision/checkpoints/efficient_transformer_seed42/
results/revision/checkpoints/biomedical_transformer_seed42/
results/revision/checkpoints/compact_transformer_seed42/
```

Run the energy-counter preflight:

```bash
python scripts/revision_linux.py --preflight
```

If preflight reports failure, send its full output to the author before attempting the experiment. The CPU-package RAPL counters must be available and readable; missing measurements are not replaced with estimates.

If preflight succeeds, run the default experiment and retain the console log:

```bash
set -o pipefail
python -u scripts/revision_linux.py 2>&1 | tee revision_linux_run.log
```

The defaults measure three tokenizers and evaluate DistilBERT/BERT-tiny in FP32 and dynamic INT8, using the saved checkpoints without retraining. The canonical operating point is batch 64, maximum length 128, with seven measurement repeats. The five-by-three GPU operating grid has already been completed in Colab; the optional large CPU grid is not required for this handoff. Allow the CPU run to finish; duration depends on the host.

## Return to the author

- `revision_linux_results.zip` from the repository root.
- `revision_linux_run.log` from the repository root.
- The complete error/preflight output if the run fails.

Do not rerun Colab training or upload model weights into GitHub. These measurements cover CPU-package energy, including tokenization in the pipeline measurements. They do not measure wall power or complete-system energy.
