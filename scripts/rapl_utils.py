#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rapl_utils.py  —  Shared Intel RAPL package-energy reader + host provenance probe.

WHY THIS EXISTS
---------------
Energy claims in this project must be traceable to a *measurement* on a *named
host*, or explicitly labelled as an estimate. Two things are needed for that:

  1. A way to read integrated package energy from Intel RAPL when the host
     actually exposes it (Linux, bare metal, readable sysfs).
  2. A record of the host, so a reviewer can tell whether a number came from a
     real power sensor or from a TDP-based software estimate, and on what CPU.

Both are provided here so `minimal_pipeline_st3.py` (training energy) and the
pre-flight checker report the same facts in the same way.

RAPL AVAILABILITY — READ THIS BEFORE TRUSTING A NUMBER
------------------------------------------------------
`/sys/class/powercap/intel-rapl:*/energy_uj` is readable only when ALL of:
  * the OS is Linux (Windows and macOS have no powercap sysfs at all);
  * the CPU is Intel, or AMD with the powercap RAPL driver bound;
  * the kernel exposes it to this user. Since CVE-2020-8694 ("PLATYPUS") most
    distributions ship `energy_uj` as mode 0400 root-only, so an unprivileged
    process sees a PermissionError even though the sensor exists. Fix with
    `sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj`.
  * the kernel is running on the metal. WSL2, Docker/Podman, and ordinary
    cloud VMs do not pass the MSRs through, so powercap is absent or frozen.

`probe_environment()` reports each of those conditions separately so a failure
is diagnosable instead of just "RAPL not available".

NOTE ON DUPLICATION (technical debt, deliberate)
------------------------------------------------
`measure_cpu_energy.py` carries its own equivalent inline RAPL reader. It is a
verified, already-run script, so it is intentionally left untouched rather than
refactored on the eve of an external run. Consolidate the two afterwards; if you
change the wraparound or domain-selection logic here, mirror it there.
"""
from __future__ import annotations

import os
import re
import glob
import platform
import subprocess

POWERCAP_BASE = "/sys/class/powercap"


# ---------------------------------------------------------------------------
# RAPL
# ---------------------------------------------------------------------------
class RAPLReader:
    """Sums integrated energy over all *top-level* Intel RAPL package domains.

    Top-level only (``intel-rapl:N``, not ``intel-rapl:N:M``) — subzones such as
    ``core`` and ``uncore`` are components of the package total, so including
    them would double-count.

    ``self.ok`` is False on any host where the counters are not readable, and
    ``self.reason`` says why. Never raises from the constructor.

    ``self.domain_names`` records which sysfs paths were actually summed, so a
    reviewer can verify no subdomain was accidentally included.
    """

    def __init__(self, base: str = POWERCAP_BASE):
        self.domains: list[tuple[str, int]] = []   # (energy_uj path, max_range_uj)
        self.domain_names: list[str] = []           # sysfs basenames for audit trail
        self.ok = False
        self.reason = "unknown"

        if platform.system() != "Linux":
            self.reason = f"not Linux (this host is {platform.system()})"
            return
        if not os.path.isdir(base):
            self.reason = (f"{base} does not exist — no powercap driver. Typical on "
                           "WSL2, containers and virtualised cloud instances.")
            return

        pkgs = sorted(p for p in glob.glob(os.path.join(base, "intel-rapl:*"))
                      if os.path.basename(p).count(":") == 1)
        if not pkgs:
            self.reason = (f"{base} exists but exposes no intel-rapl:N package "
                           "domain (non-Intel CPU, or driver not bound)")
            return

        denied = []
        for p in pkgs:
            epath = os.path.join(p, "energy_uj")
            try:
                with open(epath, "r") as fh:
                    fh.read()          # probe readability, not just existence
            except PermissionError:
                denied.append(epath)
                continue
            except OSError as exc:
                denied.append(f"{epath} ({exc.__class__.__name__})")
                continue
            try:
                with open(os.path.join(p, "max_energy_range_uj"), "r") as fh:
                    maxr = int(fh.read().strip())
            except (OSError, ValueError):
                maxr = 0               # 0 disables wraparound correction
            self.domains.append((epath, maxr))
            self.domain_names.append(os.path.basename(p))

        if self.domains:
            self.ok = True
            self.reason = f"{len(self.domains)} readable package domain(s): {self.domain_names}"
            if denied:
                self.reason += f"; {len(denied)} domain(s) unreadable, total is partial"
        else:
            self.reason = ("counters exist but are not readable by this user "
                           "(root-only since CVE-2020-8694). Run: sudo chmod a+r "
                           f"{base}/intel-rapl:*/energy_uj")

    # -- reading -----------------------------------------------------------
    def read(self):
        """Snapshot of raw microjoule counters, or None if unavailable."""
        if not self.ok:
            return None
        try:
            out = []
            for path, _ in self.domains:
                with open(path, "r") as fh:
                    out.append(int(fh.read().strip()))
            return out
        except (OSError, ValueError):
            return None

    def delta_j(self, before, after):
        """Joules between two snapshots, correcting counter wraparound.

        Returns None if either snapshot is missing, so a failed read can never
        be silently reported as 0.0 J of consumption.
        """
        if before is None or after is None:
            return None
        if len(before) != len(self.domains) or len(after) != len(self.domains):
            return None
        total_uj = 0
        for (_, maxr), b, a in zip(self.domains, before, after):
            d = a - b
            if d < 0:
                if maxr <= 0:
                    return None        # wrapped, but range unknown -> unusable
                d += maxr
            total_uj += d
        return total_uj / 1e6


# ---------------------------------------------------------------------------
# Host provenance
# ---------------------------------------------------------------------------
def _cpu_model() -> str:
    """Best-effort CPU model string, cross-platform."""
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if platform.system() == "Windows":
        # platform.processor() already returns a descriptive string on Windows.
        return platform.processor() or "unknown"
    if platform.system() == "Darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True,
                                  timeout=5).stdout.strip() or "unknown"
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or "unknown"


def _is_wsl() -> bool:
    try:
        with open("/proc/version", "r") as fh:
            v = fh.read().lower()
        return "microsoft" in v or "wsl" in v
    except OSError:
        return False


def _is_container() -> bool:
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r") as fh:
            c = fh.read()
        return bool(re.search(r"docker|lxc|kubepods|containerd", c))
    except OSError:
        return False


def _physical_cores() -> int | None:
    """Physical core count (not hyper-threaded logical count)."""
    try:
        if platform.system() == "Linux":
            # Count unique (physical id, core id) pairs from /proc/cpuinfo
            cores = set()
            phys_id, core_id = None, None
            with open("/proc/cpuinfo", "r") as fh:
                for line in fh:
                    low = line.lower().strip()
                    if low.startswith("physical id"):
                        phys_id = low.split(":", 1)[1].strip()
                    elif low.startswith("core id"):
                        core_id = low.split(":", 1)[1].strip()
                    if phys_id is not None and core_id is not None:
                        cores.add((phys_id, core_id))
                        phys_id, core_id = None, None
            if cores:
                return len(cores)
    except OSError:
        pass
    # Fallback: psutil if available, else os.cpu_count()
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except Exception:
        pass
    return os.cpu_count()


def _socket_count() -> int | None:
    """Number of physical CPU sockets from /proc/cpuinfo."""
    try:
        if platform.system() == "Linux":
            ids = set()
            with open("/proc/cpuinfo", "r") as fh:
                for line in fh:
                    if line.lower().strip().startswith("physical id"):
                        ids.add(line.split(":", 1)[1].strip())
            if ids:
                return len(ids)
    except OSError:
        pass
    return None


def _rapl_tdp_watts() -> float | None:
    """Best-effort TDP from RAPL constraint_0 (long-term power limit), if readable."""
    try:
        pkgs = sorted(glob.glob(os.path.join(POWERCAP_BASE, "intel-rapl:*")))
        pkgs = [p for p in pkgs if os.path.basename(p).count(":") == 1]
        for p in pkgs:
            cpath = os.path.join(p, "constraint_0_power_limit_uw")
            if os.path.exists(cpath):
                with open(cpath, "r") as fh:
                    return int(fh.read().strip()) / 1e6  # µW → W
    except (OSError, ValueError):
        pass
    return None


def probe_environment() -> dict:
    """Facts about the host that determine whether energy is measured or estimated.

    Safe to call anywhere; every probe is individually guarded.
    """
    rapl = RAPLReader()
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "physical_cores": _physical_cores(),
        "logical_cpus": os.cpu_count(),
        "socket_count": _socket_count(),
        "tdp_watts": _rapl_tdp_watts(),
        "is_wsl": _is_wsl(),
        "is_container": _is_container(),
        "rapl_available": rapl.ok,
        "rapl_domains": len(rapl.domains),
        "rapl_domain_names": rapl.domain_names,
        "rapl_reason": rapl.reason,
        "energy_measurement_class": (
            "measured_rapl" if rapl.ok else "estimated_software_model"),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(probe_environment(), indent=2))
