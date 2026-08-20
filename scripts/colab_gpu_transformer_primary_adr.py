#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
COLAB T4 GPU TRANSFORMER PRIMARY ADR PIPELINE (Round 3 Corrected)
================================================================================
Repository: Talhaasif7/Energy-Aware-Drug-Review
Script Path: scripts/colab_gpu_transformer_primary_adr.py

Round 3 Fixes:
  - Added missing sklearn.metrics imports (roc_auc_score, average_precision_score)
  - Added nvidia-smi power logging: idle baseline trace + per-arm load traces
  - Pre-tokenization into tensors before GPU transfer (fixes dataloader starvation)
  - Logs fitted Temperature T values + calibration-split NLL pre/post
  - Ensures .npz prediction files are auto-downloaded on Colab
  - F1@t* properly tuned on calibration split (not copied from F1@0.5)
  - 3-repeat GPU energy CV for measurement stability

Colab Quick Run:
  !pip install codecarbon transformers datasets accelerate torch pandas numpy scikit-learn scipy
  !git clone https://github.com/Talhaasif7/Energy-Aware-Drug-Review.git
  %cd Energy-Aware-Drug-Review
  !python scripts/colab_gpu_transformer_primary_adr.py
================================================================================
"""

import os
import sys
import time
import json
import math
import subprocess
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logit, expit, softmax
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    f1_score, brier_score_loss, log_loss,
    roc_auc_score, average_precision_score
)

# Ensure UTF-8 output encoding for console compatibility
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ==============================================================================
# 0. AUTOMATIC ENVIRONMENT & DEPENDENCY SETUP
# ==============================================================================
def setup_colab_environment():
    """Install required packages if running inside Google Colab."""
    if 'google.colab' in sys.modules:
        print("[SETUP] Google Colab environment detected. Checking dependencies...")
        reqs = ["codecarbon", "transformers", "datasets", "accelerate", "torch"]
        for req in reqs:
            try:
                __import__(req)
            except ImportError:
                print(f"[SETUP] Installing missing package: {req}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", req])
        print("[SETUP] Dependencies successfully verified.")

setup_colab_environment()

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("[WARNING] CodeCarbon not installed. Energy tracking fallback enabled.")

# ==============================================================================
# 1. CONFIGURATION & GATING FLAGS
# ==============================================================================
# GATING FLAG:
# Set SMOKE_TEST_MODE = True for fast validation (1 seed, 2,000 PsyTAR subset, 2 epochs).
# Set SMOKE_TEST_MODE = False for full Phase-1 matrix (3 seeds, full PsyTAR, 3 epochs).
SMOKE_TEST_MODE = True

BATCH_SIZE = 64           # Increased from 32 to reduce GPU starvation
MAX_SEQ_LENGTH = 128
LEARNING_RATE = 2e-5
NUM_WORKERS = 2           # DataLoader workers for prefetching
GPU_CV_REPEATS = 3        # Inference energy measurement repeats for CV

if SMOKE_TEST_MODE:
    SEEDS = [42]
    EPOCHS = 2
    SUBSET_SIZE = 2000
    GPU_CV_REPEATS = 1
    print("\n>>> RUNNING IN SMOKE TEST MODE (ST3b Gating: 1 seed, 2,000 PsyTAR subset, 2 epochs) <<<")
else:
    SEEDS = [42, 123, 456]
    EPOCHS = 3
    SUBSET_SIZE = None
    print("\n>>> RUNNING IN FULL PHASE-1 MATRIX MODE (3 seeds, Full PsyTAR dataset, 3 epochs) <<<")

TARGET_MODELS = {
    'Efficient Transformer': 'distilbert-base-uncased',
    'Biomedical Transformer': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
}

# ==============================================================================
# 2. NVIDIA-SMI POWER MEASUREMENT UTILITIES
# ==============================================================================
def query_gpu_power():
    """Query current GPU power draw in Watts via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        return float(result.stdout.strip())
    except Exception:
        return None

def query_gpu_utilization():
    """Query current GPU utilization % via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        return float(result.stdout.strip())
    except Exception:
        return None

def measure_gpu_idle_power(duration_s=60):
    """
    Measure GPU idle power by sampling nvidia-smi every 1 second
    for `duration_s` seconds while the GPU is idle.
    Returns: (mean_idle_watts, std_idle_watts, samples)
    """
    print(f"\n[GPU IDLE] Measuring GPU idle power for {duration_s}s...")
    torch.cuda.empty_cache()
    time.sleep(2)  # let GPU settle

    readings = []
    for i in range(duration_s):
        pw = query_gpu_power()
        if pw is not None:
            readings.append(pw)
        time.sleep(1)

    if readings:
        mean_w = np.mean(readings)
        std_w = np.std(readings)
        print(f"[GPU IDLE] Idle power: {mean_w:.2f} W (std={std_w:.2f} W, N={len(readings)})")
        return mean_w, std_w, readings
    else:
        print("[GPU IDLE] nvidia-smi not available; using fallback 11.0 W")
        return 11.0, 0.0, []

def measure_gpu_load_power_during(func, label="workload"):
    """
    Run `func()` while sampling GPU power every 0.5 seconds.
    Returns: (func_result, mean_load_watts, std_load_watts, mean_util_pct, samples)
    """
    readings_w = []
    readings_util = []
    stop_flag = [False]

    import threading
    def sampler():
        while not stop_flag[0]:
            pw = query_gpu_power()
            ut = query_gpu_utilization()
            if pw is not None:
                readings_w.append(pw)
            if ut is not None:
                readings_util.append(ut)
            time.sleep(0.5)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()

    result = func()

    stop_flag[0] = True
    t.join(timeout=2)

    mean_w = np.mean(readings_w) if readings_w else 0.0
    std_w = np.std(readings_w) if readings_w else 0.0
    mean_util = np.mean(readings_util) if readings_util else 0.0
    print(f"[GPU LOAD] {label}: {mean_w:.2f} W (std={std_w:.2f}), "
          f"Util={mean_util:.1f}%, N={len(readings_w)} samples")
    return result, mean_w, std_w, mean_util, readings_w


# ==============================================================================
# 3. DATA LOADING & PRE-TOKENIZATION
# ==============================================================================
def resolve_data_paths():
    """
    Automatically resolve paths for PsyTAR and CADEC harmonised datasets.
    Supports manually uploaded files in Colab (/content/ or current directory)
    as well as standard repository paths.
    """
    import glob

    # Candidate file names
    psytar_candidates = [
        "psytar_harmonised.csv",
        "/content/psytar_harmonised.csv",
        os.path.join(os.getcwd(), "psytar_harmonised.csv"),
        os.path.join("data", "01_primary_adr_detection", "dev_psytar", "psytar_harmonised.csv"),
        "/content/Energy-Aware-Drug-Review/data/01_primary_adr_detection/dev_psytar/psytar_harmonised.csv",
        r"e:\AI Green\data\01_primary_adr_detection\dev_psytar\psytar_harmonised.csv"
    ]

    cadec_candidates = [
        "cadec_harmonised.csv",
        "/content/cadec_harmonised.csv",
        os.path.join(os.getcwd(), "cadec_harmonised.csv"),
        os.path.join("data", "01_primary_adr_detection", "external_val_cadec", "cadec_harmonised.csv"),
        "/content/Energy-Aware-Drug-Review/data/01_primary_adr_detection/external_val_cadec/cadec_harmonised.csv",
        r"e:\AI Green\data\01_primary_adr_detection\external_val_cadec\cadec_harmonised.csv"
    ]

    psytar_path = None
    cadec_path = None

    # Check direct candidates
    for p in psytar_candidates:
        if os.path.exists(p):
            psytar_path = p
            break

    for c in cadec_candidates:
        if os.path.exists(c):
            cadec_path = c
            break

    # Recursive glob search if direct paths not found
    if not psytar_path:
        matches = glob.glob("/**/psytar_harmonised.csv", recursive=True) + glob.glob("./**/psytar_harmonised.csv", recursive=True)
        if matches:
            psytar_path = matches[0]

    if not cadec_path:
        matches = glob.glob("/**/cadec_harmonised.csv", recursive=True) + glob.glob("./**/cadec_harmonised.csv", recursive=True)
        if matches:
            cadec_path = matches[0]

    if not psytar_path or not cadec_path:
        print("[DATA] Local CSVs not found in standard paths. Attempting git clone fallback...")
        try:
            subprocess.run(["git", "clone", "https://github.com/Talhaasif7/Energy-Aware-Drug-Review.git"], check=True)
            clone_root = os.path.join(os.getcwd(), "Energy-Aware-Drug-Review")
            p_cand = os.path.join(clone_root, "data", "01_primary_adr_detection", "dev_psytar", "psytar_harmonised.csv")
            c_cand = os.path.join(clone_root, "data", "01_primary_adr_detection", "external_val_cadec", "cadec_harmonised.csv")
            if os.path.exists(p_cand):
                psytar_path = p_cand
            if os.path.exists(c_cand):
                cadec_path = c_cand
        except Exception as e:
            print(f"[DATA] Git clone fallback notice: {e}")

    if not psytar_path or not cadec_path:
        raise FileNotFoundError(
            f"Dataset files not found! Please ensure 'psytar_harmonised.csv' and 'cadec_harmonised.csv' "
            f"are uploaded to /content/ or the current working directory.\n"
            f"Found PsyTAR: {psytar_path}\nFound CADEC: {cadec_path}"
        )

    print(f"[DATA] Resolved PsyTAR dataset: {psytar_path}")
    print(f"[DATA] Resolved CADEC dataset : {cadec_path}")
    return psytar_path, cadec_path


def pre_tokenize(texts, labels, tokenizer, max_len=128):
    """
    Pre-tokenize all texts into tensors on CPU BEFORE creating DataLoader.
    This eliminates the dataloader CPU bottleneck that causes GPU starvation.
    Returns a TensorDataset ready for DataLoader.
    """
    encoding = tokenizer(
        list(texts),
        truncation=True,
        max_length=max_len,
        padding='max_length',
        return_tensors='pt'
    )
    labels_tensor = torch.tensor(list(labels), dtype=torch.long)
    return TensorDataset(encoding['input_ids'], encoding['attention_mask'], labels_tensor)


# ==============================================================================
# 4. METRIC HELPERS
# ==============================================================================
def compute_ece_adaptive(y_true, y_probs, n_bins=10):
    """Adaptive ECE with equal-frequency quantile bins."""
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0
    sorted_idx = np.argsort(y_probs)
    y_true_s = y_true[sorted_idx]
    y_probs_s = y_probs[sorted_idx]
    bin_size = max(1, n_samples // n_bins)
    ece = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n_samples
        if start >= n_samples:
            break
        bt = y_true_s[start:end]
        bp = y_probs_s[start:end]
        bn = len(bt)
        if bn > 0:
            ece += (bn / n_samples) * abs(np.mean(bt) - np.mean(bp))
    return float(ece)


def bootstrap_ece_ci(y_true, y_probs, n_bins=10, n_bootstrap=1000):
    """Bootstrap 95% CI for adaptive ECE."""
    rng = np.random.RandomState(42)
    n = len(y_true)
    boot_eces = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_eces.append(compute_ece_adaptive(y_true[idx], y_probs[idx], n_bins))
    return float(np.percentile(boot_eces, 2.5)), float(np.percentile(boot_eces, 97.5))


def find_optimal_threshold(y_true, y_probs):
    """Find threshold maximizing F1 on given set."""
    best_f1 = 0.0
    best_t = 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_true, (y_probs >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


class TemperatureScaler:
    """Post-hoc Temperature Scaling on raw logits."""
    def __init__(self):
        self.T = 1.0
        self.nll_pre = None
        self.nll_post = None

    def fit(self, y_calib, logits_calib):
        """Fit T on calibration split logits. Logs NLL pre/post."""
        log_odds = logits_calib[:, 1] - logits_calib[:, 0]
        # NLL before scaling (T=1)
        p1_pre = expit(log_odds)
        p_pre = np.column_stack([1.0 - p1_pre, p1_pre])
        self.nll_pre = float(log_loss(y_calib, p_pre, labels=[0, 1]))

        def nll_objective(T_val):
            if T_val <= 0:
                return 1e9
            scaled_p1 = expit(log_odds / T_val)
            scaled_p = np.column_stack([1.0 - scaled_p1, scaled_p1])
            return log_loss(y_calib, scaled_p, labels=[0, 1])

        res = minimize_scalar(nll_objective, bounds=(0.01, 10.0), method='bounded')
        self.T = float(res.x)
        self.nll_post = float(res.fun)
        return self

    def transform(self, logits):
        log_odds = logits[:, 1] - logits[:, 0]
        scaled_p1 = expit(log_odds / self.T)
        return np.column_stack([1.0 - scaled_p1, scaled_p1])


def eval_full_metrics(y_true, probs_2d, threshold=0.5):
    """Compute complete metric bundle from probability array."""
    p1 = probs_2d[:, 1]
    auroc = float(roc_auc_score(y_true, p1))
    auprc = float(average_precision_score(y_true, p1))

    y_pred_fixed = (p1 >= 0.5).astype(int)
    f1_fixed = float(f1_score(y_true, y_pred_fixed, pos_label=1, zero_division=0))

    y_pred_tuned = (p1 >= threshold).astype(int)
    f1_tuned = float(f1_score(y_true, y_pred_tuned, pos_label=1, zero_division=0))

    ece_ada = compute_ece_adaptive(y_true, p1)
    ece_ci_lo, ece_ci_hi = bootstrap_ece_ci(y_true, p1)
    brier = float(brier_score_loss(y_true, p1))
    nll = float(log_loss(y_true, probs_2d, labels=[0, 1]))

    return {
        'AUROC': auroc, 'AUPRC': auprc,
        'F1@0.5': f1_fixed, 'F1@t*': f1_tuned, 't*': threshold,
        'ECE_adaptive': ece_ada,
        'ECE_CI_lo': ece_ci_lo, 'ECE_CI_hi': ece_ci_hi,
        'Brier': brier, 'NLL': nll
    }


# ==============================================================================
# 5. FINE-TUNING & INFERENCE ENGINE
# ==============================================================================
def train_and_eval_single_seed(model_name, model_hf_path, train_df, calib_df,
                                test_df, cadec_df, seed, gpu_idle_w):
    """Run fine-tuning, inference, calibration, and evaluation for 1 seed."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fp16 = torch.cuda.is_available()

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n--- Model: {model_name} | Seed: {seed} | Device: {device} (FP16={use_fp16}) ---")

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_hf_path, num_labels=2).to(device)

    # PRE-TOKENIZE into tensors on CPU (fixes GPU starvation)
    print(f"  Pre-tokenizing datasets on CPU...")
    train_dataset = pre_tokenize(train_df['text'], train_df['label'], tokenizer, MAX_SEQ_LENGTH)
    calib_dataset = pre_tokenize(calib_df['text'], calib_df['label'], tokenizer, MAX_SEQ_LENGTH)
    test_dataset  = pre_tokenize(test_df['text'], test_df['label'], tokenizer, MAX_SEQ_LENGTH)
    cadec_dataset = pre_tokenize(cadec_df['text'], cadec_df['label'], tokenizer, MAX_SEQ_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    calib_loader = DataLoader(calib_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    cadec_loader = DataLoader(cadec_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # ---- FINE-TUNING with power measurement ----
    def do_training():
        model.train()
        for epoch in range(EPOCHS):
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

    os.makedirs("./energy_logs", exist_ok=True)
    tracker_train = None
    if CODECARBON_AVAILABLE:
        try:
            tracker_train = EmissionsTracker(save_to_file=True, output_dir="./energy_logs", log_level='error')
            tracker_train.start()
        except Exception:
            tracker_train = None

    t0_train = time.perf_counter()
    _, train_load_w, train_load_std, train_util, _ = measure_gpu_load_power_during(
        do_training, label=f"{model_name} Training")
    t1_train = time.perf_counter()
    train_time_secs = t1_train - t0_train

    train_energy_kwh = 0.0
    if tracker_train:
        try:
            train_energy_kwh = tracker_train.stop() or 0.0
        except Exception:
            train_energy_kwh = 0.0
    if train_energy_kwh == 0.0 and torch.cuda.is_available():
        train_energy_kwh = (train_time_secs / 3600.0) * (train_load_w / 1000.0)
    train_energy_joules = float(train_energy_kwh * 3600000.0)

    # ---- INFERENCE with power measurement ----
    def run_inference(dataloader, label="inference"):
        model.eval()
        logits_list = []
        def do_inf():
            nonlocal logits_list
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch[0].to(device)
                    attention_mask = batch[1].to(device)
                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        outputs = model(input_ids, attention_mask=attention_mask)
                    logits_list.append(outputs.logits.cpu().numpy())
            return np.concatenate(logits_list, axis=0)

        tracker_inf = None
        if CODECARBON_AVAILABLE:
            try:
                tracker_inf = EmissionsTracker(save_to_file=False, log_level='error')
                tracker_inf.start()
            except Exception:
                tracker_inf = None

        t0 = time.perf_counter()
        logits_arr, load_w, load_std, util_pct, _ = measure_gpu_load_power_during(
            do_inf, label=label)
        t1 = time.perf_counter()
        inf_time = t1 - t0

        inf_energy_kwh = 0.0
        if tracker_inf:
            try:
                inf_energy_kwh = tracker_inf.stop() or 0.0
            except Exception:
                inf_energy_kwh = 0.0
        if inf_energy_kwh == 0.0 and torch.cuda.is_available():
            inf_energy_kwh = (inf_time / 3600.0) * (load_w / 1000.0)

        return logits_arr, inf_time, float(inf_energy_kwh * 3600000.0), load_w, util_pct

    # Run inference on calib, test, and CADEC
    logits_calib, _, _, _, _ = run_inference(calib_loader, f"{model_name} Calib Inf")
    logits_test, inf_time_test, inf_j_test, inf_load_w_test, inf_util_test = \
        run_inference(test_loader, f"{model_name} PsyTAR Test Inf")
    logits_cadec, inf_time_cadec, inf_j_cadec, inf_load_w_cadec, inf_util_cadec = \
        run_inference(cadec_loader, f"{model_name} CADEC Inf")

    # GPU energy CV (repeat test inference for stability)
    inf_j_repeats = [inf_j_test]
    if GPU_CV_REPEATS > 1:
        for rep in range(GPU_CV_REPEATS - 1):
            _, _, j_rep, _, _ = run_inference(test_loader, f"{model_name} CV Rep {rep+2}")
            inf_j_repeats.append(j_rep)
    inf_j_cv = np.std(inf_j_repeats) / np.mean(inf_j_repeats) if np.mean(inf_j_repeats) > 0 else 0
    print(f"  GPU Inference Energy CV: {inf_j_cv*100:.2f}% over {len(inf_j_repeats)} repeats")

    # Compute per-1k gross and net energy
    n_test = len(test_df)
    throughput_test = n_test / inf_time_test if inf_time_test > 0 else 0.0
    inf_gross_1k = (inf_j_test / n_test) * 1000.0 if n_test > 0 else 0.0
    # Net = gross * (net_power / load_power)
    net_power_w = inf_load_w_test - gpu_idle_w
    inf_net_1k = inf_gross_1k * (net_power_w / inf_load_w_test) if inf_load_w_test > 0 else inf_gross_1k

    # Probabilities
    probs_calib_uncal = softmax(logits_calib, axis=1)
    probs_test_uncal  = softmax(logits_test, axis=1)
    probs_cadec_uncal = softmax(logits_cadec, axis=1)

    y_calib = calib_df['label'].values
    y_test  = test_df['label'].values
    y_cadec = cadec_df['label'].values

    # ---- RECALIBRATION ----
    # Temperature Scaling (fit on CALIBRATION split logits)
    temp_scaler = TemperatureScaler()
    temp_scaler.fit(y_calib, logits_calib)
    probs_test_temp  = temp_scaler.transform(logits_test)
    probs_cadec_temp = temp_scaler.transform(logits_cadec)
    probs_calib_temp = temp_scaler.transform(logits_calib)

    print(f"  Temperature T = {temp_scaler.T:.4f}")
    print(f"  Calib NLL: pre={temp_scaler.nll_pre:.4f} -> post={temp_scaler.nll_post:.4f}")
    print(f"  NLL change: {temp_scaler.nll_post - temp_scaler.nll_pre:+.4f}")

    # Isotonic Regression (fit on CALIBRATION split probabilities)
    iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso_reg.fit(probs_calib_uncal[:, 1], y_calib)
    iso_p1_test  = iso_reg.transform(probs_test_uncal[:, 1])
    probs_test_iso = np.column_stack([1.0 - iso_p1_test, iso_p1_test])
    iso_p1_cadec = iso_reg.transform(probs_cadec_uncal[:, 1])
    probs_cadec_iso = np.column_stack([1.0 - iso_p1_cadec, iso_p1_cadec])
    iso_p1_calib = iso_reg.transform(probs_calib_uncal[:, 1])
    probs_calib_iso = np.column_stack([1.0 - iso_p1_calib, iso_p1_calib])

    # ---- EVALUATE ALL METHODS ----
    methods = {
        'Uncalibrated': (probs_test_uncal, probs_cadec_uncal, probs_calib_uncal),
        'Temperature Scaling': (probs_test_temp, probs_cadec_temp, probs_calib_temp),
        'Isotonic Regression': (probs_test_iso, probs_cadec_iso, probs_calib_iso),
    }

    eval_results = {}
    for method_name, (p_test, p_cadec, p_calib) in methods.items():
        # Find optimal threshold on CALIBRATION split
        t_star, _ = find_optimal_threshold(y_calib, p_calib[:, 1])

        # Evaluate on TEST set using calib-tuned threshold
        psytar_metrics = eval_full_metrics(y_test, p_test, threshold=t_star)

        # Evaluate on CADEC (zero-shot) using same threshold
        cadec_metrics = eval_full_metrics(y_cadec, p_cadec, threshold=t_star)

        eval_results[method_name] = {
            'psytar': psytar_metrics,
            'cadec': cadec_metrics
        }

        print(f"\n  {method_name}:")
        print(f"    PsyTAR: AUROC={psytar_metrics['AUROC']:.4f} "
              f"AUPRC={psytar_metrics['AUPRC']:.4f} "
              f"F1@t*={psytar_metrics['F1@t*']:.4f} (t*={t_star:.2f}) "
              f"F1@0.5={psytar_metrics['F1@0.5']:.4f} "
              f"ECE={psytar_metrics['ECE_adaptive']:.4f} "
              f"[{psytar_metrics['ECE_CI_lo']:.4f},{psytar_metrics['ECE_CI_hi']:.4f}]")
        print(f"    CADEC:  AUROC={cadec_metrics['AUROC']:.4f} "
              f"AUPRC={cadec_metrics['AUPRC']:.4f} "
              f"F1@t*={cadec_metrics['F1@t*']:.4f} "
              f"F1@0.5={cadec_metrics['F1@0.5']:.4f} "
              f"ECE={cadec_metrics['ECE_adaptive']:.4f} "
              f"[{cadec_metrics['ECE_CI_lo']:.4f},{cadec_metrics['ECE_CI_hi']:.4f}]")

    # ---- SAVE PREDICTION ARTIFACTS ----
    os.makedirs("results", exist_ok=True)
    npz_filename = f"results/{model_name.lower().replace(' ', '_')}_seed{seed}_predictions.npz"
    np.savez_compressed(
        npz_filename,
        logits_calib=logits_calib,
        logits_test=logits_test,
        logits_cadec=logits_cadec,
        probs_test_uncal=probs_test_uncal,
        probs_test_temp=probs_test_temp,
        probs_test_iso=probs_test_iso,
        probs_cadec_uncal=probs_cadec_uncal,
        probs_cadec_temp=probs_cadec_temp,
        probs_cadec_iso=probs_cadec_iso,
        y_calib=y_calib,
        y_test=y_test,
        y_cadec=y_cadec
    )
    print(f"  [ARTIFACT] Predictions saved: {npz_filename}")

    # Auto-download on Colab
    if 'google.colab' in sys.modules:
        try:
            from google.colab import files
            files.download(npz_filename)
            print(f"  [COLAB] Auto-download triggered for {npz_filename}")
        except Exception as e:
            print(f"  [COLAB] Download notice: {e}")

    # Package results
    seed_result = {
        'seed': seed,
        'train_time_sec': float(train_time_secs),
        'train_energy_joules': float(train_energy_joules),
        'train_load_watts': float(train_load_w),
        'train_util_pct': float(train_util),
        'inf_throughput_sps': float(throughput_test),
        'inf_gross_energy_1k_j': float(inf_gross_1k),
        'inf_net_energy_1k_j': float(inf_net_1k),
        'inf_load_watts_psytar': float(inf_load_w_test),
        'inf_load_watts_cadec': float(inf_load_w_cadec),
        'inf_util_pct_psytar': float(inf_util_test),
        'inf_util_pct_cadec': float(inf_util_cadec),
        'gpu_idle_watts': float(gpu_idle_w),
        'inf_energy_cv_pct': float(inf_j_cv * 100),
        'temperature_T': float(temp_scaler.T),
        'calib_nll_pre': float(temp_scaler.nll_pre),
        'calib_nll_post': float(temp_scaler.nll_post),
        'eval_results': eval_results,
    }

    # Clean memory
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return seed_result


# ==============================================================================
# 6. MAIN PIPELINE
# ==============================================================================
def main():
    psytar_path, cadec_path = resolve_data_paths()

    print(f"\nLoading PsyTAR from: {psytar_path}")
    df_psytar_full = pd.read_csv(psytar_path)
    print(f"Loading CADEC from:  {cadec_path}")
    df_cadec_full  = pd.read_csv(cadec_path)

    # Subset PsyTAR if in Smoke Test mode
    if SUBSET_SIZE and len(df_psytar_full) > SUBSET_SIZE:
        df_psytar, _ = train_test_split(
            df_psytar_full, train_size=SUBSET_SIZE,
            stratify=df_psytar_full['label'], random_state=42)
        print(f"Stratified subset: {len(df_psytar)} PsyTAR units.")
    else:
        df_psytar = df_psytar_full
        print(f"Full PsyTAR: {len(df_psytar)} units.")

    print(f"CADEC zero-shot target: {len(df_cadec_full)} units.")

    # ---- MEASURE GPU IDLE POWER (60-second trace) ----
    if torch.cuda.is_available():
        gpu_idle_w, gpu_idle_std, _ = measure_gpu_idle_power(duration_s=60)
    else:
        gpu_idle_w = 0.0

    # ---- FORCE DEVICE PINNING ----
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Idle Power: {gpu_idle_w:.2f} W")

    all_model_results = {}

    for model_name, model_hf_path in TARGET_MODELS.items():
        print(f"\n{'='*80}")
        print(f" STARTING: {model_name} ({model_hf_path})")
        print(f"{'='*80}")

        seed_results = []

        for seed in SEEDS:
            # 3-Way Stratified Split: 60% Train, 20% Calib, 20% Test
            train_df, calib_test_df = train_test_split(
                df_psytar, train_size=0.6,
                stratify=df_psytar['label'], random_state=seed)
            calib_df, test_df = train_test_split(
                calib_test_df, test_size=0.5,
                stratify=calib_test_df['label'], random_state=seed)

            print(f"  Split (seed={seed}): Train={len(train_df)} "
                  f"Calib={len(calib_df)} Test={len(test_df)}")

            res = train_and_eval_single_seed(
                model_name, model_hf_path,
                train_df, calib_df, test_df, df_cadec_full,
                seed, gpu_idle_w)
            seed_results.append(res)

        all_model_results[model_name] = seed_results

    # ---- SUMMARY TABLE ----
    print("\n" + "=" * 110)
    print("        EMPIRICAL GPU TRANSFORMER BENCHMARK SUMMARY (COLAB T4)")
    print("=" * 110)

    for model_name, seeds_res in all_model_results.items():
        print(f"\n  === {model_name} ===")
        avg_T = np.mean([r['temperature_T'] for r in seeds_res])
        avg_nll_pre = np.mean([r['calib_nll_pre'] for r in seeds_res])
        avg_nll_post = np.mean([r['calib_nll_post'] for r in seeds_res])
        avg_throughput = np.mean([r['inf_throughput_sps'] for r in seeds_res])
        avg_gross = np.mean([r['inf_gross_energy_1k_j'] for r in seeds_res])
        avg_net = np.mean([r['inf_net_energy_1k_j'] for r in seeds_res])
        avg_load = np.mean([r['inf_load_watts_psytar'] for r in seeds_res])
        avg_util = np.mean([r['inf_util_pct_psytar'] for r in seeds_res])
        avg_cv = np.mean([r['inf_energy_cv_pct'] for r in seeds_res])

        print(f"    Temperature T = {avg_T:.4f}")
        print(f"    Calib NLL: pre={avg_nll_pre:.4f} -> post={avg_nll_post:.4f} "
              f"(delta={avg_nll_post - avg_nll_pre:+.4f})")
        print(f"    Throughput: {avg_throughput:.1f} sents/s")
        print(f"    GPU Load: {avg_load:.1f} W | Util: {avg_util:.1f}%")
        print(f"    Gross Energy: {avg_gross:.2f} J/1k | Net Energy: {avg_net:.2f} J/1k")
        print(f"    Energy CV: {avg_cv:.2f}%")

        for method in ['Uncalibrated', 'Temperature Scaling', 'Isotonic Regression']:
            ps = [r['eval_results'][method]['psytar'] for r in seeds_res]
            cd = [r['eval_results'][method]['cadec'] for r in seeds_res]

            print(f"\n    {method}:")
            print(f"      PsyTAR: AUROC={np.mean([m['AUROC'] for m in ps]):.4f} "
                  f"AUPRC={np.mean([m['AUPRC'] for m in ps]):.4f} "
                  f"F1@t*={np.mean([m['F1@t*'] for m in ps]):.4f} "
                  f"F1@0.5={np.mean([m['F1@0.5'] for m in ps]):.4f} "
                  f"ECE={np.mean([m['ECE_adaptive'] for m in ps]):.4f} "
                  f"[{np.mean([m['ECE_CI_lo'] for m in ps]):.4f},"
                  f"{np.mean([m['ECE_CI_hi'] for m in ps]):.4f}]")
            print(f"      CADEC:  AUROC={np.mean([m['AUROC'] for m in cd]):.4f} "
                  f"AUPRC={np.mean([m['AUPRC'] for m in cd]):.4f} "
                  f"F1@t*={np.mean([m['F1@t*'] for m in cd]):.4f} "
                  f"F1@0.5={np.mean([m['F1@0.5'] for m in cd]):.4f} "
                  f"ECE={np.mean([m['ECE_adaptive'] for m in cd]):.4f} "
                  f"[{np.mean([m['ECE_CI_lo'] for m in cd]):.4f},"
                  f"{np.mean([m['ECE_CI_hi'] for m in cd]):.4f}]")

    # ---- EXPORT TO JSON ----
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "colab_transformer_gpu_results.json")

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    export = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "smoke_test_mode": SMOKE_TEST_MODE,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "gpu_idle_watts": float(gpu_idle_w),
        "results": {k: [{kk: convert(vv) for kk, vv in r.items()} for r in v]
                    for k, v in all_model_results.items()}
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, default=convert)
    print(f"\n[ARTIFACT] Results JSON: {os.path.abspath(json_path)}")

    # Auto-download JSON on Colab
    if 'google.colab' in sys.modules:
        try:
            from google.colab import files
            files.download(json_path)
        except Exception as e:
            print(f"[COLAB] Download notice: {e}")

    print("\n" + "=" * 80)
    print("  GPU validation complete. Download .npz and .json artifacts.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
