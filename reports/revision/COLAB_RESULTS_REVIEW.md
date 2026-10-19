> Update, 19 October 2026: the planned Linux results are now validated and integrated. The current paper has 27 primary configurations and the original fifteen-arm common catalogue across ten splits. Original PubMedBERT GPU configurations remain; its extra CPU run is excluded. The revised manuscript and response have been regenerated. See [Linux audit](LINUX_RESULTS_REVIEW.md) and [current status](revision_status.json). The original Colab audit below records the earlier handoff stage.

# Returned Colab results: validation and next step

The returned prediction ZIP and console log have been inspected. All 30 predictions (three models across ten grouped splits) are present; their SHA-256 values match the run manifest. Exact texts, labels and review-group identities match the local corpora. The three train/calibration/audit group sets are disjoint within every split. Every probability array can be loaded without pickle and has finite, normalized values. The ZIP passes its CRC check.

All 45 planned board-energy settings are present: three encoders, five batch sizes, and three maximum lengths. All 135 repeat measurements are recorded, and every cell passes the configured repeatability check. The manifest reports a Tesla T4, PyTorch 2.11.0+cu130 and Transformers 4.57.6. The log reaches the final archive-creation message and contains no training traceback. The TensorFlow CPU/CUDA startup messages do not describe the PyTorch device; the GPU manifest and measured board traces confirm the T4 run.

## Canonical operating point and uncalibrated discrimination

The energy column uses batch 64, maximum length 128 and the seed-42 checkpoint. These board measurements are reused as fixed configuration costs across the ten grouped-split selection analyses; they are not ten independent hardware measurements. AUROC means are descriptive means over the ten splits.

| Model | Seed-42 source AUROC | Seed-42 external AUROC | Ten-split mean source AUROC | Ten-split mean external AUROC | Board J/1,000 | Sentences/s |
|---|---:|---:|---:|---:|---:|---:|
| DistilBERT | 0.9332 | 0.9133 | 0.9338 | 0.9162 | 75.473 | 860.0 |
| PubMedBERT | 0.9414 | 0.9057 | 0.9350 | 0.9131 | 152.233 | 428.0 |
| BERT-tiny | 0.7873 | 0.8249 | 0.7923 | 0.8317 | 11.028 | 3465.1 |

The returned models and timing scope produce different results from the submitted artifacts. Updated revision analyses must explicitly use `results/revision/gpu_predictions`; the historical paper values must not be mixed into the new transformer tables. The original submission files remain preserved.

## Statistical integration completed

The returned predictions and canonical board-energy readings have been integrated into `results/revision/revision_analysis.json` and the numerical CSV supplements. All ten grouped splits use the same 15-arm catalogue (five models, three recalibration variants). Each split uses 2,000 whole-review/post bootstrap draws, paired AUROC intervals, the complete 165-cell constraint grid, bin sensitivity and both workload threshold policies. All 1,650 grid cells are evaluated and pass feasibility-monotonicity checks. Prediction and energy-file hashes and all stored draw shapes pass `check_revision_artifacts.py --analysis-only`.

At illustrative tau=0.07 and E=120 J/1,000, the ten-split choices are: DistilBERT + Isotonic: 4/10; DistilBERT + TempScale: 6/10.

Across source/external retrospective 90%-sensitivity diagnostics, 20 of 548 arm-pair/split comparisons within 0.015 AUROC exceed a 2% absolute relative change in flagged-review workload. These are descriptive, dependent comparisons, not clinical margin validation.

The existing revised DOCX and reviewer-response body are earlier working drafts. They have not been regenerated from these returned Colab results. Regeneration and final consistency review are deferred until the CPU extension is integrated; the numerical supplements and this audit record the current completed work.

## Next external step

The complete 45-row operating grid is exported as `results/revision/gpu_operating_grid.csv`. A labelled energy-surface figure is available as `reports/revision/figures/supplement_gpu_energy_surface.png` and `.pdf`.

Share the Drive copy of `revision_checkpoints.zip` with the professor. Follow `PROFESSOR_LINUX_STEPS.md`: use the full repository, extract the checkpoint ZIP at the repository root, run RAPL preflight, then execute `scripts/revision_linux.py`. Return `revision_linux_results.zip` and `revision_linux_run.log`.

The checkpoint ZIP itself is not present in this workspace, so its contents have not been inspected here. Colab does not need to be rerun for the completed outputs. The CPU extension, manuscript integration and final consistency review remain required before resubmission. GPU-board energy excludes host-package/wall energy; the complete-system request remains unmet with the available hardware. No independent clinical validation of the equivalence margin is claimed.
