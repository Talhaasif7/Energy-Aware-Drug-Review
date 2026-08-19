# GPU Energy Measurement Protocol (Colab T4/V100)

## Problem

CodeCarbon on Google Colab frequently mis-attributes or misses GPU power domains,
reporting ~9.7W average load when the actual T4 GPU draws 60–70W under fp16 training.
This 7× under-reporting biases the Pareto frontier toward transformer models and
undermines the paper's core energy–accuracy trade-off analysis.

## Ground Truth: Manual nvidia-smi Power Trace

### Background Thread Setup

Run `nvidia-smi` power sampling in a background thread during all GPU workloads:

```python
import subprocess
import threading
import time
import csv
import io

class NvidiaSmiPowerTracer:
    """
    Sample GPU power draw at 100ms intervals using nvidia-smi.
    Integrate the trace post-hoc to get total energy (Joules).
    """
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.power_readings = []  # (timestamp_s, power_watts)
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._stop_event.clear()
        self.power_readings = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self._integrate()

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    ['nvidia-smi',
                     '--query-gpu=power.draw',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    power_w = float(result.stdout.strip())
                    self.power_readings.append((time.time(), power_w))
            except Exception:
                pass
            time.sleep(self.interval_ms / 1000.0)

    def _integrate(self):
        """Trapezoidal integration of power trace → energy in Joules."""
        if len(self.power_readings) < 2:
            return 0.0, 0.0, []

        total_energy_j = 0.0
        for i in range(1, len(self.power_readings)):
            dt = self.power_readings[i][0] - self.power_readings[i-1][0]
            avg_power = (self.power_readings[i][1] +
                        self.power_readings[i-1][1]) / 2.0
            total_energy_j += avg_power * dt

        total_time = (self.power_readings[-1][0] -
                     self.power_readings[0][0])
        avg_power = total_energy_j / total_time if total_time > 0 else 0.0

        return total_energy_j, avg_power, self.power_readings
```

### Usage in Training Loop

```python
# Ground truth measurement
power_tracer = NvidiaSmiPowerTracer(interval_ms=100)

# CodeCarbon as cross-check
from codecarbon import EmissionsTracker
cc_tracker = EmissionsTracker(save_to_file=False, log_level='error')

power_tracer.start()
cc_tracker.start()

# ... training loop ...

cc_kwh = cc_tracker.stop()
manual_energy_j, avg_power_w, trace = power_tracer.stop()

print(f"Manual (nvidia-smi): {manual_energy_j:.2f} J, {avg_power_w:.1f} W avg")
print(f"CodeCarbon:          {cc_kwh * 3_600_000:.2f} J")
print(f"Ratio (Manual/CC):   {manual_energy_j / (cc_kwh * 3_600_000):.1f}x")
```

## Reporting Standard

1. **Ground truth**: Manual nvidia-smi integration (report as primary measurement)
2. **Cross-check**: CodeCarbon reading (report as secondary, note the ratio)
3. **Expected T4 ranges**:
   - Idle: 10–12 W
   - fp16 training load: 55–70 W
   - fp32 inference: 40–55 W
4. If Manual/CodeCarbon ratio exceeds 2×, flag CodeCarbon as unreliable for
   that workload and use manual trace exclusively.
