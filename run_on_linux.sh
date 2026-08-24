#!/usr/bin/env bash
# =============================================================================
#  run_on_linux.sh  —  ONE command. Runs the whole CPU pipeline, captures every
#  artifact and the raw console log, and packs a single archive to send back.
#
#      bash run_on_linux.sh
#
#  Nothing here needs a GPU. Nothing here needs network access. It takes roughly
#  15-25 minutes depending on the machine.
#
#  Options (all optional):
#      --measure-s N   energy measurement window, seconds        (default 30)
#      --repeats N     repeats per model for the CV estimate     (default 5)
#      --skip-preflight   run even if the pre-flight check fails (not advised)
#      --install       pip install -r requirements.txt first
#
#  WHY THE DEFAULTS ARE LONGER THAN THE SCRIPT DEFAULTS
#  Two runs of the identical benchmark on the same Windows host disagreed by
#  ~30% on throughput, because background load moved while it measured. A 30 s
#  window with 5 repeats gives a coefficient of variation we can actually quote.
#  Please keep the machine otherwise IDLE while this runs — no browser, no IDE,
#  no other jobs. That matters more than any flag here.
# =============================================================================

# Re-exec under bash if this was invoked as `sh run_on_linux.sh`. The script uses
# process substitution and `pipefail`, neither of which POSIX sh / dash provides,
# and dash aborts on `set -o pipefail` outright. Cheaper to handle than to explain.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -uo pipefail

MEASURE_S=30
REPEATS=5
SKIP_PREFLIGHT=0
DO_INSTALL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --measure-s)      MEASURE_S="${2:-30}"; shift 2 ;;
    --repeats)        REPEATS="${2:-5}";    shift 2 ;;
    --skip-preflight) SKIP_PREFLIGHT=1;     shift ;;
    --install)        DO_INSTALL=1;         shift ;;
    -h|--help)        sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

# --- locate the repo root from this script's own location, not the cwd --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
ROOT="$SCRIPT_DIR"

OUT_DIR="$ROOT/handoff_out"
LOG="$OUT_DIR/console_log.txt"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$OUT_DIR" || { echo "cannot create $OUT_DIR"; exit 1; }
: > "$LOG"

# Everything from here on is tee'd to $LOG, so the log we get back is the raw
# stream and not a retyped summary. fd 3/4 keep the real terminal, so the tail of
# the script can close the tee and then still print — otherwise the archive would
# be built before tee had flushed the last lines into the log.
exec 3>&1 4>&2
exec > >(tee -a "$LOG") 2>&1

say() { printf '%s\n' "$*"; }
rule() { printf '%s\n' "=============================================================================="; }

rule
say "  ENERGY-AWARE DRUG-REVIEW  —  CPU PIPELINE (Linux run)"
rule
say "  started (UTC) : $STAMP"
say "  repo root     : $ROOT"
say "  measure window: ${MEASURE_S}s x ${REPEATS} repeats"
say ""

# --- pick an interpreter -----------------------------------------------------
PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  say "[FATAL] no python3 on PATH. Install Python 3.10-3.12 and re-run."
  exit 1
fi
say "[env] interpreter : $PY ($($PY -V 2>&1))"

if [ "$DO_INSTALL" = "1" ]; then
  say ""
  say "--- installing requirements ---"
  "$PY" -m pip install -r "$ROOT/requirements.txt" || {
    say "[FATAL] pip install failed. Fix the errors above, or install the packages"
    say "        listed in requirements.txt by hand, then re-run without --install."
    exit 1
  }
fi

# --- host provenance ---------------------------------------------------------
# Captured BEFORE the run, into separate files, so every number we publish can be
# attributed to a named machine. Each probe is best-effort: a missing tool must
# not stop the pipeline.
say ""
say "--- capturing host provenance ---"
{
  echo "### date -u";            date -u
  echo; echo "### uname -a";     uname -a
  echo; echo "### python";       "$PY" -VV
  echo; echo "### nproc";        nproc 2>/dev/null || echo "n/a"
  echo; echo "### lscpu";        lscpu 2>/dev/null || echo "n/a"
  echo; echo "### /proc/cpuinfo model name";
                                 grep -m1 "model name" /proc/cpuinfo 2>/dev/null || echo "n/a"
  echo; echo "### free -h";      free -h 2>/dev/null || echo "n/a"
  echo; echo "### powercap tree";
                                 ls -l /sys/class/powercap/ 2>/dev/null || echo "no /sys/class/powercap"
  echo; echo "### powercap energy_uj permissions";
                                 ls -l /sys/class/powercap/intel-rapl:*/energy_uj 2>/dev/null || echo "none readable"
  echo; echo "### cpu governor";
                                 cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "n/a"
  echo; echo "### load average"; cat /proc/loadavg 2>/dev/null || echo "n/a"
} > "$OUT_DIR/env_host.txt" 2>&1
say "[env] wrote handoff_out/env_host.txt"

"$PY" -m pip freeze > "$OUT_DIR/env_pip_freeze.txt" 2>&1 \
  && say "[env] wrote handoff_out/env_pip_freeze.txt" \
  || say "[env] pip freeze unavailable (non-fatal)"

# A machine-readable version of the same facts, from the code's own probe.
# stderr goes to a SEPARATE file on purpose: this one is parsed as JSON later (and
# grepped for rapl_available below), so a single Python DeprecationWarning merged
# into it would make the artifact invalid.
"$PY" "$ROOT/scripts/rapl_utils.py" > "$OUT_DIR/env_probe.json" \
                                    2> "$OUT_DIR/env_probe.stderr.txt" \
  && say "[env] wrote handoff_out/env_probe.json" \
  || say "[env] rapl_utils probe failed (non-fatal; see env_probe.stderr.txt)"

# Report the load average plainly — a busy machine invalidates the energy numbers
# more thoroughly than any wrong flag.
if [ -r /proc/loadavg ]; then
  LOAD1="$(cut -d' ' -f1 /proc/loadavg)"
  say "[env] 1-min load average: $LOAD1  (want this near 0 before starting)"
fi

# --- pre-flight --------------------------------------------------------------
say ""
say "--- pre-flight check ---"
if [ "$SKIP_PREFLIGHT" = "1" ]; then
  say "[skip] pre-flight skipped by flag."
else
  if ! "$PY" "$ROOT/scripts/preflight_linux.py"; then
    say ""
    say "[FATAL] pre-flight found blocking problems (listed above). Nothing was run."
    say "        Fix them and re-run:  bash run_on_linux.sh"
    say "        To override anyway:   bash run_on_linux.sh --skip-preflight"
    exit 1
  fi
fi

# --- the pipeline ------------------------------------------------------------
# run_all_cpu.py internally runs, in order:
#   1 measure_cpu_energy.py   -> results/cpu_energy_measured.json
#   2 run_frozen_split_analysis.py -> results/frozen_split_reconciled.json
#   3 eccms_regime_st8.py     -> results/st8_regime_reconciled.json
#   4 budget_and_subgroup_st6_st7.py -> results/st6_st7_reconciled.json
#
# STALENESS MARKER. Every artifact below is also present in the sender's own
# checkout from earlier Windows runs. If a step fails here and an old file is
# lying in results/, a naive copy would return it labelled as this run's output —
# i.e. a Windows software estimate presented as a Linux sensor measurement. The
# marker's timestamp is compared against each artifact, so only files actually
# written by THIS run are collected.
MARKER="$OUT_DIR/.run_started"
: > "$MARKER"
sleep 1                     # keep the marker strictly older than any new output

RC_MAIN=0
say ""
rule
say "  STEP 1/2  —  run_all_cpu.py  (CPU energy, frozen split, ST8, ST6/ST7)"
rule
"$PY" "$ROOT/scripts/run_all_cpu.py" --measure-s "$MEASURE_S" --repeats "$REPEATS"
RC_MAIN=$?
say ""
say "[rc] run_all_cpu.py exit code = $RC_MAIN"

# ST3 is separate: it is the stage that measures TRAINING time and energy, which
# the ST6 budget extrapolation depends on. It was print-only until now, so its
# numbers had been hand-copied into the code as literals; it now writes
# results/st3_cpu_energy.json.
RC_ST3=0
say ""
rule
say "  STEP 2/2  —  minimal_pipeline_st3.py  (training time + training energy)"
rule
"$PY" "$ROOT/scripts/minimal_pipeline_st3.py"
RC_ST3=$?
say ""
say "[rc] minimal_pipeline_st3.py exit code = $RC_ST3"

# --- collect -----------------------------------------------------------------
say ""
rule
say "  COLLECTING ARTIFACTS"
rule

# Artifacts THIS run must produce. Each is staleness-checked against $MARKER.
WANT_JSON="
results/frozen_split_reconciled.json
results/st8_regime_reconciled.json
results/st6_st7_reconciled.json
results/cpu_energy_measured.json
results/st3_cpu_energy.json
"
# Inputs, returned unchanged for provenance. NOT staleness-checked: these came
# from the one-off Colab GPU run and are not expected to be rewritten here.
WANT_INPUT="
results/colab_transformer_gpu_results.json
"
WANT_EXTRA="
results/cpu_arms_seed42_predictions.npz
reports/st8_regime_map.png
"

MISSING=0
STALE=0
for rel in $WANT_JSON; do
  if [ ! -s "$ROOT/$rel" ]; then
    say "  [MISS] $rel   <-- not produced; check the log above for why"
    MISSING=$((MISSING + 1))
  elif [ ! "$ROOT/$rel" -nt "$MARKER" ]; then
    # Exists, but predates this run: a leftover from an earlier run on another
    # machine. Deliberately NOT copied — returning it would misattribute it.
    say "  [STALE] $rel  <-- left over from an EARLIER run, not written by this"
    say "          one. Not collected, because returning it would present old"
    say "          numbers as this machine's. The step that should have written"
    say "          it failed — see the log above."
    STALE=$((STALE + 1))
    MISSING=$((MISSING + 1))
  else
    mkdir -p "$OUT_DIR/$(dirname "$rel")"
    cp -f "$ROOT/$rel" "$OUT_DIR/$rel"
    say "  [ OK ] $rel"
  fi
done
for rel in $WANT_INPUT; do
  if [ -s "$ROOT/$rel" ]; then
    mkdir -p "$OUT_DIR/$(dirname "$rel")"
    cp -f "$ROOT/$rel" "$OUT_DIR/$rel"
    say "  [ OK ] $rel  (input, returned for provenance)"
  else
    say "  [MISS] $rel   <-- required INPUT is absent; the run cannot have worked"
    MISSING=$((MISSING + 1))
  fi
done
for rel in $WANT_EXTRA; do
  if [ -s "$ROOT/$rel" ] && [ "$ROOT/$rel" -nt "$MARKER" ]; then
    mkdir -p "$OUT_DIR/$(dirname "$rel")"
    cp -f "$ROOT/$rel" "$OUT_DIR/$rel"
    say "  [ OK ] $rel  (optional)"
  else
    say "  [  - ] $rel  (optional, not produced by this run)"
  fi
done
rm -f "$MARKER"

# --- verdict -----------------------------------------------------------------
say ""
rule
if [ "$RC_MAIN" -ne 0 ] || [ "$RC_ST3" -ne 0 ] || [ "$MISSING" -gt 0 ]; then
  say "  FINISHED WITH PROBLEMS"
  rule
  say "  run_all_cpu.py exit=$RC_MAIN, minimal_pipeline_st3.py exit=$RC_ST3,"
  say "  $MISSING expected artifact(s) missing ($STALE of them stale leftovers)."
  say ""
  say "  Please send the archive ANYWAY — the console log inside it is exactly"
  say "  what is needed to diagnose this, and any artifact that did get written"
  say "  is still usable."
  EXIT=1
else
  say "  FINISHED CLEANLY"
  rule
  say "  Every expected artifact was produced, and each one was verified to have"
  say "  been written by THIS run rather than left over from an earlier one."
  say ""
  say "  Please do not retype, summarise, or reformat the console log — send the"
  say "  file. A retyped log has already caused one round of confusion on this"
  say "  project, where a summary contained metrics that no run had produced."
  EXIT=0
fi

# Report whether the run achieved a real measurement or only an estimate, since
# that is the entire reason for running on Linux.
if grep -q '"rapl_available": true' "$OUT_DIR/env_probe.json" 2>/dev/null; then
  say ""
  say "  RAPL: available on this host — CPU energy is a real sensor reading."
else
  say ""
  say "  RAPL: NOT available on this host — CPU energy in this run is a software"
  say "        estimate, same as before. See handoff_out/env_probe.json for the"
  say "        reason. If the machine is bare-metal Intel Linux, this usually"
  say "        just needs:"
  say "            sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj"
  say "        then re-run this script."
fi
rule

# --- finalise: close the log, THEN digest and archive it ---------------------
# Restoring fd 1/2 closes tee's stdin so it flushes and exits; the short sleep
# gives it time to do so. Only after that is console_log.txt complete and safe to
# put in the archive.
exec 1>&3 2>&4
sleep 1

# console_log.txt and SHA256SUMS.txt are excluded from the digest list: the log is
# itself the audit trail, and a file cannot contain its own hash.
{
  echo "# SHA-256 of every returned artifact, computed on the run machine."
  echo "# Excludes console_log.txt (the audit trail itself) and this file."
  ( cd "$OUT_DIR" && find . -type f \
      ! -name 'SHA256SUMS.txt' ! -name 'console_log.txt' -print0 \
      | sort -z | xargs -0 sha256sum )
} > "$OUT_DIR/SHA256SUMS.txt" 2>/dev/null \
  && echo "  [ OK ] handoff_out/SHA256SUMS.txt" \
  || echo "  [  - ] SHA256SUMS.txt (sha256sum unavailable)"

ARCHIVE="$ROOT/handoff_results_${STAMP}.tar.gz"
if tar -czf "$ARCHIVE" -C "$ROOT" "$(basename "$OUT_DIR")" 2>/dev/null; then
  echo ""
  echo "  =========================================================================="
  echo "   SEND THIS ONE FILE BACK:"
  echo "     $ARCHIVE"
  echo "     ($(du -h "$ARCHIVE" 2>/dev/null | cut -f1))"
  echo "  =========================================================================="
else
  echo ""
  echo "  [warn] could not create the tar archive; send the handoff_out/ folder"
  echo "         itself instead. Everything needed is inside it:"
  echo "         $OUT_DIR"
fi
echo ""

exit "$EXIT"
