# Returned Linux results: audit

Both archives pass CRC and all six received prediction hashes match the supplied manifests. Text, label and group arrays match the source partitions and external corpus exactly. All arrays load without pickle. Every pipeline measurement retains seven repeats and passes repeatability; raw joule/count/time identities and coupled-median summaries have been checked. The final paper and selection catalogue include only the four planned DistilBERT/BERT-tiny CPU configurations. The extra PubMedBERT CPU experiment is excluded at the author’s instruction; its raw transfer files are preserved locally. Original PubMedBERT GPU configurations remain included.

| Included CPU configuration | Source AUROC | External AUROC | Package J/1,000 | Sentences/s | CV (%) |
|---|---:|---:|---:|---:|---:|
| distilbert_cpu_fp32 | 0.9332 | 0.9133 | 2945.312 | 22.69 | 1.31 |
| distilbert_cpu_int8 | 0.9259 | 0.9069 | 1721.758 | 35.25 | 0.92 |
| bert_tiny_cpu_fp32 | 0.7873 | 0.8249 | 68.474 | 722.79 | 0.42 |
| bert_tiny_cpu_int8 | 0.7860 | 0.8252 | 57.118 | 841.23 | 0.60 |

The initial PubMedBERT tokenizer measurement has CV 10.81%, exceeding the configured 10% limit. It is retained and flagged, alongside the separately returned follow-up measurement (CV 0.38%). Tokenizer-only measurement for all three GPU encoder families was already part of the planned protocol. The passing follow-up is used descriptively; the two runs are not silently averaged or substituted in the raw artifacts.

Counter wraps whose max-range value cannot be independently reconstructed from the archive: 0.

Checkpoint files not supplied locally; included FP32 probabilities agree with saved GPU FP16 outputs within .01, which is compatibility evidence, not a cryptographic weight-identity proof.

CPU-package and GPU-board measurements are different scopes on separate hosts. They are not summed or described as complete-system energy.
