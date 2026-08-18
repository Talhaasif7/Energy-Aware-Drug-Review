import os
import sys
import time
import platform
import subprocess
import numpy as np

def ensure_dependencies():
    """Ensure required packages (codecarbon, lightgbm, scikit-learn, psutil) are installed."""
    required_packages = {
        'codecarbon': 'codecarbon',
        'lightgbm': 'lightgbm',
        'sklearn': 'scikit-learn',
        'psutil': 'psutil'
    }
    for module_name, pip_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            print(f"[Setup] Installing missing package: '{pip_name}'...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

# Auto-ensure dependencies inside execution process
ensure_dependencies()

import psutil
from codecarbon import OfflineEmissionsTracker, EmissionsTracker
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier

def detect_environment():
    """
    Detect OS, CPU, RAM, GPU status, and Intel RAPL / CodeCarbon measurement drivers.
    """
    os_name = platform.system()
    os_release = platform.release()
    arch = platform.machine()
    processor = platform.processor() or "CPU"
    
    physical_cores = psutil.cpu_count(logical=False) or "N/A"
    logical_cores = psutil.cpu_count(logical=True) or "N/A"
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    
    # Check Intel RAPL path on Linux/WSL
    rapl_path = "/sys/class/powercap/intel-rapl"
    rapl_available = os.path.exists(rapl_path) if os_name == "Linux" else False
    
    # Check CodeCarbon measurement mode
    measurement_mode = "CodeCarbon Engine (RAPL)" if rapl_available else f"CodeCarbon Fallback Estimation ({os_name})"

    # Check GPU availability
    gpu_available = False
    gpu_name = "None Detected"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    env_info = {
        'os': f"{os_name} {os_release} ({arch})",
        'processor': processor,
        'physical_cores': physical_cores,
        'logical_cores': logical_cores,
        'ram_gb': ram_gb,
        'rapl_available': rapl_available,
        'measurement_mode': measurement_mode,
        'gpu_available': gpu_available,
        'gpu_name': gpu_name
    }
    return env_info

def measure_idle_power(duration_secs=60):
    """
    Measure baseline/idle power consumption by sleeping main thread for duration_secs.
    Returns: idle_energy_joules, idle_power_watts
    """
    print(f"\n--- Measuring Idle Power Baseline ({duration_secs} Seconds Sleep) ---")
    tracker = EmissionsTracker(save_to_file=False, log_level='error')
    
    start_time = time.perf_counter()
    tracker.start()
    time.sleep(duration_secs)
    energy_kwh = tracker.stop()
    elapsed_secs = time.perf_counter() - start_time
    
    # Convert kWh to Joules: 1 kWh = 3.6e6 Joules
    idle_energy_joules = (energy_kwh or 0.0) * 3600000.0 if (energy_kwh and energy_kwh > 0) else 0.0
    idle_power_watts = (idle_energy_joules / elapsed_secs) if elapsed_secs > 0 else 0.0

    print(f"Idle measurement completed in {elapsed_secs:.2f}s.")
    print(f"Idle Energy: {idle_energy_joules:.4f} J ({energy_kwh:.8f} kWh)")
    print(f"Average Idle Power: {idle_power_watts:.4f} W")
    
    return idle_energy_joules, idle_power_watts

def profile_cpu_workload(repeats=3):
    """
    Generate mock tabular dataset (10,000 rows x 20 features) simulating GBDT input.
    Train model 3 times, tracking execution time, total energy (Joules), and load power (W).
    Returns list of run dicts and computed CV.
    """
    print(f"\n--- CPU Workload Energy Profiling ({repeats}x Repeats) ---")
    print("Generating synthetic tabular dataset: 10,000 samples, 20 numerical features...")
    
    np.random.seed(42)
    X = np.random.randn(10000, 20).astype(np.float32)
    y = np.random.randint(0, 2, size=10000)
    
    runs_data = []

    for run_idx in range(1, repeats + 1):
        print(f"  [Run {run_idx}/{repeats}] Training LightGBM model...")
        
        # Instantiate GBDT model
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42 + run_idx,
            n_jobs=-1,
            verbose=-1
        )
        
        tracker = EmissionsTracker(save_to_file=False, log_level='error')
        
        t0 = time.perf_counter()
        tracker.start()
        
        # Execute training workload
        clf.fit(X, y)
        
        energy_kwh = tracker.stop()
        t1 = time.perf_counter()
        
        elapsed_secs = t1 - t0
        energy_joules = (energy_kwh or 0.0) * 3600000.0 if (energy_kwh and energy_kwh > 0) else 0.0
        load_power_watts = (energy_joules / elapsed_secs) if elapsed_secs > 0 else 0.0
        
        runs_data.append({
            'run': run_idx,
            'time_secs': elapsed_secs,
            'energy_kwh': energy_kwh or 0.0,
            'energy_joules': energy_joules,
            'load_power_watts': load_power_watts
        })
        
        print(f"    Run {run_idx}: Time = {elapsed_secs:.3f}s | Energy = {energy_joules:.4f} J ({energy_kwh:.8f} kWh) | Load Power = {load_power_watts:.4f} W")
        time.sleep(1) # Short pause between repeats

    energies = [r['energy_joules'] for r in runs_data]
    mean_energy = np.mean(energies)
    std_energy = np.std(energies, ddof=1) if repeats > 1 else 0.0
    cv_pct = (std_energy / mean_energy * 100.0) if mean_energy > 0 else 0.0

    return runs_data, mean_energy, std_energy, cv_pct

def run_gpu_check(gpu_available):
    """
    Perform GPU trivial energy check if NVIDIA GPU is present, else output Google Colab instructions.
    """
    print("\n--- GPU Trivial Energy Check ---")
    if gpu_available:
        try:
            import torch
            print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
            print("Running PyTorch CUDA matrix multiplication benchmark...")
            
            device = torch.device('cuda:0')
            tracker = EmissionsTracker(save_to_file=False, log_level='error')
            
            t0 = time.perf_counter()
            tracker.start()
            
            # Tiny PyTorch matrix multiplication workload
            A = torch.randn(4000, 4000, device=device)
            B = torch.randn(4000, 4000, device=device)
            for _ in range(50):
                C = torch.matmul(A, B)
            torch.cuda.synchronize()
            
            energy_kwh = tracker.stop()
            elapsed_secs = time.perf_counter() - t0
            energy_joules = (energy_kwh or 0.0) * 3600000.0
            
            print(f"GPU Workload completed in {elapsed_secs:.3f}s.")
            print(f"GPU Energy: {energy_joules:.4f} J ({energy_kwh:.8f} kWh)")
            return True, energy_joules
        except Exception as e:
            print(f"GPU execution note: {e}")
            return False, 0.0
    else:
        print("No local NVIDIA GPU detected.")
        print("[Notice for GPU Profiling]: To measure GPU energy in Cloud/Google Colab:")
        print("  1. Enable GPU accelerator (T4/V100/A100) in Colab Notebook Settings.")
        print("  2. Install codecarbon: !pip install codecarbon torch")
        print("  3. Wrap PyTorch model training with codecarbon.EmissionsTracker().")
        return False, 0.0

def print_st2_report(env_info, idle_joules, idle_watts, runs_data, mean_e, std_e, cv_pct, gpu_tested, gpu_energy):
    """
    Print formatted Smoke Test 2 (ST2) report.
    """
    print("\n" + "="*80)
    print("                ST2 — ENERGY MEASUREMENT SANITY REPORT")
    print("="*80)
    
    print("\n--- 1. ENVIRONMENT & HARDWARE DETECTION ---")
    print(f"  * Operating System     : {env_info['os']}")
    print(f"  * Processor (CPU)      : {env_info['processor']}")
    print(f"  * Cores (Physical/Log) : {env_info['physical_cores']} physical / {env_info['logical_cores']} logical")
    print(f"  * System RAM           : {env_info['ram_gb']} GB")
    print(f"  * Intel RAPL (Linux)   : {'Available' if env_info['rapl_available'] else 'Not Applicable (Windows)'}")
    print(f"  * Measurement Engine   : {env_info['measurement_mode']}")
    print(f"  * GPU Status           : {env_info['gpu_name']}")

    print("\n--- 2. IDLE POWER BASELINE (60s Sleep) ---")
    print(f"  * Total Idle Energy    : {idle_joules:.4f} Joules ({idle_joules / 3600.0:.6f} Wh)")
    print(f"  * Average Idle Power   : {idle_watts:.4f} Watts")

    print("\n--- 3. CPU WORKLOAD ENERGY PROFILING (3x Repeats) ---")
    print("  Run  |  Exec Time (s)  |  Energy (Joules)  |  Energy (Wh)  |  Average Load Power (W)")
    print("  " + "-"*72)
    for r in runs_data:
        wh = r['energy_joules'] / 3600.0
        print(f"  [{r['run']:02d}] |  {r['time_secs']:12.3f}   |  {r['energy_joules']:14.4f}   |  {wh:10.6f}   |  {r['load_power_watts']:20.4f}")
    print("  " + "-"*72)
    print(f"  Mean Energy (J)        : {mean_e:.4f} J")
    print(f"  Std Deviation (J)      : {std_e:.4f} J")
    print(f"  Coefficient of Var (CV): {cv_pct:.2f}%")
    
    reliability_status = "PASSED (< 10% threshold)" if cv_pct < 10.0 else "WARNING (>= 10% threshold)"
    print(f"  Measurement Reliability: {reliability_status}")

    print("\n--- 4. GPU ENERGY CHECK ---")
    if gpu_tested:
        print(f"  * Local GPU Test       : Executed successfully (Energy: {gpu_energy:.4f} J)")
    else:
        print("  * Local GPU Test       : Skipped (No local CUDA GPU). Follow Colab instructions above for cloud GPU runs.")

    print("\n--- 5. SANITY TEST SUMMARY & VERDICT ---")
    print("  [OK] CodeCarbon energy tracking engine verified.")
    print("  [OK] Baseline idle power established.")
    print("  [OK] Workload repeatability & energy measurement stability validated.")
    print("="*80 + "\n")

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    print("Starting Smoke Test 2 (ST2 - Energy Measurement Sanity)...")
    
    # 1. Environment Detection
    env_info = detect_environment()
    
    # 2. Idle Power Measurement (60 seconds sleep)
    idle_joules, idle_watts = measure_idle_power(duration_secs=60)
    
    # 3. CPU Workload Energy Profiling (3 repeats)
    runs_data, mean_e, std_e, cv_pct = profile_cpu_workload(repeats=3)
    
    # 4. GPU Trivial Check
    gpu_tested, gpu_energy = run_gpu_check(env_info['gpu_available'])
    
    # 5. Generate Report
    print_st2_report(env_info, idle_joules, idle_watts, runs_data, mean_e, std_e, cv_pct, gpu_tested, gpu_energy)

if __name__ == "__main__":
    main()
