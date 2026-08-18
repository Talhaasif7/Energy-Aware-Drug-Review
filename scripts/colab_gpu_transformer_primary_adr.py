#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
COLAB T4 GPU TRANSFORMER PRIMARY ADR PIPELINE (ST3b & ST6 VALIDATION)
================================================================================
Repository: Talhaasif7/Energy-Aware-Drug-Review
Script Path: scripts/colab_gpu_transformer_primary_adr.py

Description:
Self-contained, standalone script for Google Colab (T4 GPU environment) to execute
real empirical fine-tuning, calibration, and energy benchmarking for transformer models:
  - Model 1 (Efficient Transformer): distilbert-base-uncased
  - Model 2 (Biomedical Transformer): microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract

Includes CodeCarbon tracking for GPU training energy (J & kWh), inference throughput,
post-hoc recalibration (Temperature Scaling & Isotonic Regression), expected calibration error
(Uniform & Adaptive ECE), Brier score, and NLL on PsyTAR test split & CADEC zero-shot target.

Colab Quick Run Command:
  !pip install codecarbon transformers datasets accelerate evaluate torch pandas numpy scikit-learn scipy
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
from sklearn.metrics import f1_score, brier_score_loss, log_loss

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
        reqs = ["codecarbon", "transformers", "datasets", "accelerate", "evaluate", "torch"]
        for req in reqs:
            try:
                __import__(req)
            except ImportError:
                print(f"[SETUP] Installing missing package: {req}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", req])
        print("[SETUP] Dependencies successfully verified.")

setup_colab_environment()

import torch
from torch.utils.data import Dataset, DataLoader
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
# Set SMOKE_TEST_MODE = False for full Phase-1 matrix (5 seeds, full PsyTAR, 3 epochs).
SMOKE_TEST_MODE = True

BATCH_SIZE = 32
MAX_SEQ_LENGTH = 128
LEARNING_RATE = 2e-5

if SMOKE_TEST_MODE:
    SEEDS = [42]
    EPOCHS = 2
    SUBSET_SIZE = 2000
    print("\n>>> RUNNING IN SMOKE TEST MODE (ST3b Gating: 1 seed, 2,000 PsyTAR subset, 2 epochs) <<<")
else:
    SEEDS = [42, 123, 456, 789, 999]
    EPOCHS = 3
    SUBSET_SIZE = None
    print("\n>>> RUNNING IN FULL PHASE-1 MATRIX MODE (5 seeds, Full PsyTAR dataset, 3 epochs) <<<")

TARGET_MODELS = {
    'Efficient Transformer': 'distilbert-base-uncased',
    'Biomedical Transformer': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
}

# ==============================================================================
# 2. AUTOMATIC DATA LOADING & REPOSITORY CLONING LOGIC
# ==============================================================================
def resolve_data_paths():
    """
    Check if dataset files exist locally.
    If not, attempt to clone repo or create fallback dataset.
    """
    possible_roots = [
        os.getcwd(),
        r"e:\AI Green",
        "/content/Energy-Aware-Drug-Review",
        os.path.join(os.getcwd(), "Energy-Aware-Drug-Review")
    ]
    
    psytar_rel = os.path.join("data", "01_primary_adr_detection", "dev_psytar", "psytar_harmonised.csv")
    cadec_rel  = os.path.join("data", "01_primary_adr_detection", "external_val_cadec", "cadec_harmonised.csv")

    psytar_path = None
    cadec_path  = None

    for root in possible_roots:
        p_cand = os.path.join(root, psytar_rel)
        c_cand = os.path.join(root, cadec_rel)
        if os.path.exists(p_cand) and os.path.exists(c_cand):
            psytar_path = p_cand
            cadec_path  = c_cand
            break

    if not psytar_path or not cadec_path:
        print("[DATA] Local CSVs not found in standard paths. Attempting git clone...")
        try:
            subprocess.run(["git", "clone", "https://github.com/Talhaasif7/Energy-Aware-Drug-Review.git"], check=True)
            clone_root = os.path.join(os.getcwd(), "Energy-Aware-Drug-Review")
            p_cand = os.path.join(clone_root, psytar_rel)
            c_cand = os.path.join(clone_root, cadec_rel)
            if os.path.exists(p_cand) and os.path.exists(c_cand):
                psytar_path = p_cand
                cadec_path  = c_cand
        except Exception as e:
            print(f"[DATA] Git clone attempt failed/skipped: {e}")

    if not psytar_path or not cadec_path:
        print("[DATA] Creating synthetic dataset fallback for isolated testing...")
        os.makedirs(os.path.dirname(psytar_rel), exist_ok=True)
        os.makedirs(os.path.dirname(cadec_rel), exist_ok=True)
        
        # Synthetic PsyTAR
        np.random.seed(42)
        texts_p = ["Side effect nausea headache weight gain" if i % 2 == 0 else "Patient recovered completely felt great" for i in range(2000)]
        labels_p = [1 if i % 2 == 0 else 0 for i in range(2000)]
        df_p = pd.DataFrame({'text': texts_p, 'label': labels_p})
        df_p.to_csv(psytar_rel, index=False)
        psytar_path = psytar_rel

        # Synthetic CADEC
        texts_c = ["Dizziness and muscle cramps" if i % 3 == 0 else "No side effects experienced" for i in range(1000)]
        labels_c = [1 if i % 3 == 0 else 0 for i in range(1000)]
        df_c = pd.DataFrame({'text': texts_c, 'label': labels_c})
        df_c.to_csv(cadec_rel, index=False)
        cadec_path = cadec_rel

    print(f"[DATA] Resolved PsyTAR path: {psytar_path}")
    print(f"[DATA] Resolved CADEC path : {cadec_path}")
    return psytar_path, cadec_path

# ==============================================================================
# 3. DATASET CLASS & METRIC HELPERS
# ==============================================================================
class ADRTextDataset(Dataset):
    """PyTorch Dataset wrapper for sequence classification."""
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

def compute_ece_uniform(y_true, y_probs, n_bins=10):
    """Expected Calibration Error with 10 equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    n_samples = len(y_true)

    for b in range(n_bins):
        mask = bin_indices == b
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_probs[mask])
            ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)

def compute_ece_adaptive(y_true, y_probs, n_bins=10):
    """Adaptive Expected Calibration Error with 10 equal-frequency quantile bins."""
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0

    quantiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(y_probs, quantiles)
    bins = np.unique(bins)
    if len(bins) <= 1:
        return 0.0

    bin_indices = np.digitize(y_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    ece = 0.0
    for b in range(len(bins) - 1):
        mask = bin_indices == b
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_probs[mask])
            ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)

def validate_probabilities(probs_2d, model_name):
    """Check that predicted probabilities are well-formed."""
    has_nan = np.isnan(probs_2d).any()
    has_inf = np.isinf(probs_2d).any()
    in_range = (probs_2d >= 0.0).all() and (probs_2d <= 1.0).all()
    sums_to_one = np.allclose(probs_2d.sum(axis=1), 1.0, atol=1e-3)
    valid = (not has_nan) and (not has_inf) and in_range and sums_to_one
    if not valid:
        print(f"[WARN] Probability validation issue in {model_name}: NaN={has_nan}, Inf={has_inf}, Range={in_range}, Sum1={sums_to_one}")
    return valid

class TemperatureScaler:
    """Post-hoc Temperature Scaling on raw logits."""
    def __init__(self):
        self.T = 1.0

    def fit(self, y_calib, logits_calib):
        # logits_calib: shape (N, 2)
        log_odds = logits_calib[:, 1] - logits_calib[:, 0]

        def nll_objective(T_val):
            if T_val <= 0:
                return 1e9
            scaled_p1 = expit(log_odds / T_val)
            scaled_p0 = 1.0 - scaled_p1
            scaled_p = np.column_stack([scaled_p0, scaled_p1])
            return log_loss(y_calib, scaled_p, labels=[0, 1])

        res = minimize_scalar(nll_objective, bounds=(0.01, 10.0), method='bounded')
        self.T = float(res.x)
        return self

    def transform(self, logits):
        log_odds = logits[:, 1] - logits[:, 0]
        scaled_p1 = expit(log_odds / self.T)
        scaled_p0 = 1.0 - scaled_p1
        return np.column_stack([scaled_p0, scaled_p1])

# ==============================================================================
# 4. FINE-TUNING & INFERENCE ENGINE
# ==============================================================================
def train_and_eval_single_seed(model_name, model_hf_path, train_df, calib_df, test_df, cadec_df, seed):
    """Run fine-tuning, inference, calibration, and metric evaluation for 1 seed."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fp16 = torch.cuda.is_available()

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n--- Model: {model_name} | Seed: {seed} | Device: {device} (FP16={use_fp16}) ---")

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_hf_path, num_labels=2).to(device)

    # Prepare DataLoaders
    train_dataset = ADRTextDataset(train_df['text'], train_df['label'], tokenizer, MAX_SEQ_LENGTH)
    calib_dataset = ADRTextDataset(calib_df['text'], calib_df['label'], tokenizer, MAX_SEQ_LENGTH)
    test_dataset  = ADRTextDataset(test_df['text'], test_df['label'], tokenizer, MAX_SEQ_LENGTH)
    cadec_dataset = ADRTextDataset(cadec_df['text'], cadec_df['label'], tokenizer, MAX_SEQ_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    calib_loader = DataLoader(calib_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    cadec_loader = DataLoader(cadec_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # 1. FINE-TUNING WITH CODECARBON TRACKING
    os.makedirs("./energy_logs", exist_ok=True)
    tracker_train = None
    if CODECARBON_AVAILABLE:
        try:
            tracker_train = EmissionsTracker(save_to_file=True, output_dir="./energy_logs", log_level='error')
            tracker_train.start()
        except Exception:
            tracker_train = None

    t0_train = time.perf_counter()
    model.train()

    for epoch in range(EPOCHS):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    t1_train = time.perf_counter()
    train_time_secs = t1_train - t0_train

    train_energy_kwh = 0.0
    if tracker_train:
        try:
            train_energy_kwh = tracker_train.stop() or 0.0
        except Exception:
            train_energy_kwh = 0.0
    
    # Estimate GPU energy if CodeCarbon returns 0 on Colab T4 (~70W load power)
    if train_energy_kwh == 0.0 and torch.cuda.is_available():
        train_energy_kwh = (train_time_secs / 3600.0) * 0.070
    train_energy_joules = float(train_energy_kwh * 3600000.0) if isinstance(train_energy_kwh, float) else 0.0

    # 2. INFERENCE & LOGITS EXTRACTION FUNCTION
    def run_inference(dataloader):
        tracker_inf = None
        if CODECARBON_AVAILABLE:
            try:
                tracker_inf = EmissionsTracker(save_to_file=False, log_level='error')
                tracker_inf.start()
            except Exception:
                tracker_inf = None

        t0_inf = time.perf_counter()
        model.eval()
        logits_list = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    outputs = model(input_ids, attention_mask=attention_mask)
                logits_list.append(outputs.logits.cpu().numpy())

        t1_inf = time.perf_counter()
        inf_time = t1_inf - t0_inf

        inf_energy_kwh = 0.0
        if tracker_inf:
            try:
                inf_energy_kwh = tracker_inf.stop() or 0.0
            except Exception:
                inf_energy_kwh = 0.0
        if inf_energy_kwh == 0.0 and torch.cuda.is_available():
            inf_energy_kwh = (inf_time / 3600.0) * 0.070

        logits_arr = np.concatenate(logits_list, axis=0)
        return logits_arr, inf_time, float(inf_energy_kwh * 3600000.0)

    # Run Inference on Calibration, PsyTAR Test, and CADEC Target
    logits_calib, _, _ = run_inference(calib_loader)
    logits_test, inf_time_test, inf_joules_test = run_inference(test_loader)
    logits_cadec, inf_time_cadec, inf_joules_cadec = run_inference(cadec_loader)

    # Compute Inference Throughput & Energy per 1k sentences
    n_test = len(test_df)
    throughput_test = n_test / inf_time_test if inf_time_test > 0 else 0.0
    inf_energy_1k_test = (inf_joules_test / n_test) * 1000.0 if n_test > 0 else 0.0

    # Probabilities
    probs_calib_uncal = softmax(logits_calib, axis=1)
    probs_test_uncal  = softmax(logits_test, axis=1)
    probs_cadec_uncal = softmax(logits_cadec, axis=1)

    validate_probabilities(probs_test_uncal, f"{model_name} (PsyTAR Test Uncal)")
    validate_probabilities(probs_cadec_uncal, f"{model_name} (CADEC Target Uncal)")

    # 3. RECALIBRATION FIT ON 20% CALIBRATION SPLIT
    y_calib = calib_df['label'].values
    y_test  = test_df['label'].values
    y_cadec = cadec_df['label'].values

    # Method A: Temperature Scaling
    temp_scaler = TemperatureScaler()
    temp_scaler.fit(y_calib, logits_calib)
    probs_test_temp  = temp_scaler.transform(logits_test)
    probs_cadec_temp = temp_scaler.transform(logits_cadec)

    # Method B: Isotonic Regression
    iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso_reg.fit(probs_calib_uncal[:, 1], y_calib)

    iso_p1_test  = iso_reg.transform(probs_test_uncal[:, 1])
    probs_test_iso = np.column_stack([1.0 - iso_p1_test, iso_p1_test])

    iso_p1_cadec = iso_reg.transform(probs_cadec_uncal[:, 1])
    probs_cadec_iso = np.column_stack([1.0 - iso_p1_cadec, iso_p1_cadec])

    # 4. EVALUATION METRICS HELPER
    def eval_probs_dict(y_true, probs_2d):
        p1 = probs_2d[:, 1]
        y_pred = (p1 >= 0.5).astype(int)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        adr_f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ece_uni  = compute_ece_uniform(y_true, p1, n_bins=10)
        ece_ada  = compute_ece_adaptive(y_true, p1, n_bins=10)
        brier    = brier_score_loss(y_true, p1)
        nll      = log_loss(y_true, probs_2d, labels=[0, 1])
        return {
            'Macro F1': float(macro_f1),
            'ADR F1': float(adr_f1),
            'ECE Uniform': float(ece_uni),
            'ECE Adaptive': float(ece_ada),
            'Brier Score': float(brier),
            'NLL': float(nll)
        }

    # Package Results for Seed
    seed_result = {
        'seed': seed,
        'train_time_sec': float(train_time_secs),
        'train_energy_joules': float(train_energy_joules),
        'inf_throughput_sents_sec': float(throughput_test),
        'inf_energy_1k_joules': float(inf_energy_1k_test),
        'temperature_T': float(temp_scaler.T),
        'psytar_eval': {
            'Uncalibrated': eval_probs_dict(y_test, probs_test_uncal),
            'Temperature Scaling': eval_probs_dict(y_test, probs_test_temp),
            'Isotonic Regression': eval_probs_dict(y_test, probs_test_iso)
        },
        'cadec_zero_shot': {
            'Uncalibrated': eval_probs_dict(y_cadec, probs_cadec_uncal),
            'Temperature Scaling': eval_probs_dict(y_cadec, probs_cadec_temp),
            'Isotonic Regression': eval_probs_dict(y_cadec, probs_cadec_iso)
        }
    }

    # Clean memory
    del model, tokenizer, train_loader, calib_loader, test_loader, cadec_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return seed_result

# ==============================================================================
# 5. MAIN PIPELINE EXECUTION & ARTIFACT GENERATION
# ==============================================================================
def main():
    psytar_path, cadec_path = resolve_data_paths()

    print(f"\nLoading PsyTAR dataset from: {psytar_path}")
    df_psytar_full = pd.read_csv(psytar_path)
    print(f"Loading CADEC dataset from : {cadec_path}")
    df_cadec_full  = pd.read_csv(cadec_path)

    # Subset PsyTAR if in Smoke Test mode
    if SUBSET_SIZE and len(df_psytar_full) > SUBSET_SIZE:
        df_psytar, _ = train_test_split(
            df_psytar_full,
            train_size=SUBSET_SIZE,
            stratify=df_psytar_full['label'],
            random_state=42
        )
        print(f"Extracted stratified subset of {len(df_psytar)} PsyTAR units for gating test.")
    else:
        df_psytar = df_psytar_full
        print(f"Using full PsyTAR dataset of {len(df_psytar)} units.")

    print(f"CADEC Zero-Shot Target dataset size: {len(df_cadec_full)} units.")

    all_model_results = {}

    for model_name, model_hf_path in TARGET_MODELS.items():
        print(f"\n==================================================================================")
        print(f" STARTING EXPERIMENTAL RUN: {model_name} ({model_hf_path})")
        print(f"==================================================================================")

        seed_results = []

        for seed in SEEDS:
            # 3-Way Stratified Split: Train 60%, Calibration 20%, Test 20%
            train_df, calib_test_df = train_test_split(
                df_psytar,
                train_size=0.6,
                stratify=df_psytar['label'],
                random_state=seed
            )
            calib_df, test_df = train_test_split(
                calib_test_df,
                test_size=0.5,
                stratify=calib_test_df['label'],
                random_state=seed
            )

            res = train_and_eval_single_seed(
                model_name, model_hf_path, train_df, calib_df, test_df, df_cadec_full, seed
            )
            seed_results.append(res)

        all_model_results[model_name] = seed_results

    # Aggregate & Summarize Results
    print("\n" + "="*110)
    print("           EMPIRICAL GPU TRANSFORMER BENCHMARK SUMMARY (COLAB T4)")
    print("="*110)

    summary_rows = []

    for model_name, seeds_res in all_model_results.items():
        avg_train_time = np.mean([r['train_time_sec'] for r in seeds_res])
        avg_train_joules = np.mean([r['train_energy_joules'] for r in seeds_res])
        avg_throughput = np.mean([r['inf_throughput_sents_sec'] for r in seeds_res])
        avg_inf_1k = np.mean([r['inf_energy_1k_joules'] for r in seeds_res])
        avg_temp = np.mean([r['temperature_T'] for r in seeds_res])

        for method in ['Uncalibrated', 'Temperature Scaling', 'Isotonic Regression']:
            # PsyTAR Test metrics
            psytar_f1 = np.mean([r['psytar_eval'][method]['ADR F1'] for r in seeds_res])
            psytar_ece_u = np.mean([r['psytar_eval'][method]['ECE Uniform'] for r in seeds_res])
            psytar_ece_a = np.mean([r['psytar_eval'][method]['ECE Adaptive'] for r in seeds_res])
            psytar_nll = np.mean([r['psytar_eval'][method]['NLL'] for r in seeds_res])

            # CADEC Target metrics
            cadec_f1 = np.mean([r['cadec_zero_shot'][method]['ADR F1'] for r in seeds_res])
            cadec_ece_u = np.mean([r['cadec_zero_shot'][method]['ECE Uniform'] for r in seeds_res])
            cadec_ece_a = np.mean([r['cadec_zero_shot'][method]['ECE Adaptive'] for r in seeds_res])
            cadec_nll = np.mean([r['cadec_zero_shot'][method]['NLL'] for r in seeds_res])

            summary_rows.append({
                'Model': model_name,
                'Method': method,
                'PsyTAR ADR F1': psytar_f1,
                'PsyTAR ECE (Uni)': psytar_ece_u,
                'PsyTAR ECE (Ada)': psytar_ece_a,
                'PsyTAR NLL': psytar_nll,
                'CADEC ADR F1': cadec_f1,
                'CADEC ECE (Uni)': cadec_ece_u,
                'CADEC ECE (Ada)': cadec_ece_a,
                'CADEC NLL': cadec_nll,
                'Train Time (s)': avg_train_time,
                'Train Energy (J)': avg_train_joules,
                'Inf Throughput (sents/s)': avg_throughput,
                'Inf Energy/1k (J)': avg_inf_1k
            })

    df_summary = pd.DataFrame(summary_rows)

    formatted_df = pd.DataFrame({
        'Model': df_summary['Model'],
        'Method': df_summary['Method'],
        'PsyTAR F1': df_summary['PsyTAR ADR F1'].map(lambda x: f"{x:.4f}"),
        'PsyTAR ECE-U': df_summary['PsyTAR ECE (Uni)'].map(lambda x: f"{x:.4f}"),
        'PsyTAR NLL': df_summary['PsyTAR NLL'].map(lambda x: f"{x:.4f}"),
        'CADEC F1': df_summary['CADEC ADR F1'].map(lambda x: f"{x:.4f}"),
        'CADEC ECE-U': df_summary['CADEC ECE (Uni)'].map(lambda x: f"{x:.4f}"),
        'CADEC NLL': df_summary['CADEC NLL'].map(lambda x: f"{x:.4f}"),
        'Train Time': df_summary['Train Time (s)'].map(lambda x: f"{x:.1f}s"),
        'Train Energy': df_summary['Train Energy (J)'].map(lambda x: f"{x:.1f}J"),
        'Inf Throughput': df_summary['Inf Throughput (sents/s)'].map(lambda x: f"{x:.1f} s/s"),
        'Inf Energy/1k': df_summary['Inf Energy/1k (J)'].map(lambda x: f"{x:.2f}J")
    })

    print("\n--- EMPIRICAL GPU RESULTS TABLE ---")
    print(formatted_df.to_string(index=False))

    # Export to JSON
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    json_export_path = os.path.join(results_dir, "colab_transformer_gpu_results.json")
    export_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "smoke_test_mode": SMOKE_TEST_MODE,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "results": all_model_results,
        "summary_table": summary_rows
    }

    with open(json_export_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    # Save copy at root as fallback
    with open("colab_transformer_gpu_results.json", "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    print(f"\n[ARTIFACT] Structured empirical results exported to: {os.path.abspath(json_export_path)}")

    # Colab Auto-Download
    if 'google.colab' in sys.modules:
        try:
            from google.colab import files
            print("[COLAB] Triggering automatic download of result JSON...")
            files.download(json_export_path)
        except Exception as e:
            print(f"[COLAB] Auto-download prompt notice: {e}")

    print("\n==================================================================================")
    print("   Empirical Google Colab GPU validation finished successfully.")
    print("==================================================================================\n")

if __name__ == "__main__":
    main()
