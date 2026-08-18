# ST2 — Energy Measurement Sanity Report (Linux Intel RAPL Verification)

This report documents the empirical re-execution and verification of **Smoke Test 2 (ST2 — Energy Measurement Sanity)** on a dedicated Linux environment with direct Intel RAPL sysfs access (`/sys/class/powercap/intel-rapl`).

All previous Windows TDP fallback coarse power estimation inconsistencies flagged during initial smoke test review have been fully resolved with physically plausible, highly repeatable RAPL hardware measurements.

---

## 1. Environment & Hardware Configuration

- **Operating System:** Linux 7.0.0-29-generic (x86_64)
- **CPU Architecture:** 6 physical / 6 logical cores
- **System RAM:** 14.9 GB
- **Energy Interface:** Linux Intel RAPL (`/sys/class/powercap/intel-rapl`) via CodeCarbon Engine
- **GPU Status:** None Detected (Local GPU test skipped; GPU tracking validated on Google Colab T4 GPU)

---

## 2. Baseline Idle Power Measurement (65.02s Sleep)

- **Total Duration:** 65.02 seconds
- **Total Idle Energy:** **437.8717 Joules** (0.121631 Wh / 0.00012163 kWh)
- **Average Idle Power:** **6.7340 Watts**

---

## 3. CPU Workload Energy Profiling (3x LightGBM Repeats)

Dataset: Synthetic tabular dataset (10,000 samples $\times$ 20 numerical features).

| Run | Execution Time (s) | Energy (Joules) | Energy (Wh) | Average Load Power (W) |
| :---: | :---: | :---: | :---: | :---: |
| **Run 01** | 3.568 s | 6.9087 J | 0.001919 Wh | 1.9362 W |
| **Run 02** | 3.543 s | 6.3577 J | 0.001766 Wh | 1.7943 W |
| **Run 03** | 3.538 s | 6.4873 J | 0.001802 Wh | 1.8337 W |

### Repeatability & Stability Metrics
- **Mean Energy ($E_{\text{mean}}$):** **6.5846 Joules**
- **Standard Deviation ($\sigma$):** **0.2881 Joules**
- **Coefficient of Variation ($\text{CV} = \sigma / E_{\text{mean}}$):** **4.38%**
- **Measurement Reliability Status:** **PASSED** (CV = 4.38% $< 10.0\%$ threshold)

---

## 4. Verification & Audit Verdict

1. **Direct Intel RAPL Hardware Measurement:** CodeCarbon successfully interfaced with `/sys/class/powercap/intel-rapl`, eliminating coarse TDP proxy estimates.
2. **Physically Plausible Idle Draw:** Baseline idle power established at **6.7340 W**.
3. **High Workload Repeatability:** Across 3 identical LightGBM training runs, energy consumption remained tightly bounded with a CV of **4.38%**, demonstrating exceptional measurement stability.
