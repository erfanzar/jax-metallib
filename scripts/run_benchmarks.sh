#!/bin/bash
# Run the jax-metallib vs jax-mps comparison under conditions that give usable
# numbers, and keep a log of every run.
#
# Two things skewed runs during development and are checked before starting:
#   - battery power: on an M5 laptop it roughly halved loop-heavy rows
#     (scan_rnn 5.1 -> 11.7 ms) while jax-mps slowed less;
#   - other GPU users: Chrome's GPU helper at ~170% CPU took matmul_f16 from
#     2.38x to 1.28x; games and Metal shader compiles do the same.
# Power state and load are recorded again at the end, so a run that was
# disturbed half-way can be spotted and discarded.
#
# Usage: scripts/run_benchmarks.sh [options]
#   --filter SUBSTR      only workloads whose name contains SUBSTR (plain
#                        substring, not a regex), e.g. fusion. or loop.
#   --cycles N           ABBA cycles per workload (default 3; use 3 before
#                        calling a row a win or a loss)
#   --order ab|abba      run order (default abba)
#   --jax-mps-python P   interpreter with jax-mps installed (default: first of
#                        .venv-jax-mps-0.11/bin/python, .venv-jax-mps/bin/python)
#   --python P           interpreter with jax-metallib (default .venv/bin/python)
#   --max-load X         refuse to start above this 1-minute load (default 6)
#   --skip-build         do not rebuild the native library first
#   --allow-battery      run on battery power anyway (results are not comparable)
#   --force              skip the busy-machine checks
#   -h, --help           show this help
#
# Output: build/bench-runs/<UTC time>-<commit>[-dirty].{log,jsonl,md}
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

FILTER=""
CYCLES=3
ORDER="abba"
MPS_PY=""
PY=".venv/bin/python"
MAX_LOAD=6
SKIP_BUILD=0
ALLOW_BATTERY=0
FORCE=0

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
    --filter) FILTER="${2:?--filter needs a value}"; shift 2 ;;
    --cycles) CYCLES="${2:?--cycles needs a value}"; shift 2 ;;
    --order) ORDER="${2:?--order needs a value}"; shift 2 ;;
    --jax-mps-python) MPS_PY="${2:?--jax-mps-python needs a value}"; shift 2 ;;
    --python) PY="${2:?--python needs a value}"; shift 2 ;;
    --max-load) MAX_LOAD="${2:?--max-load needs a value}"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --allow-battery) ALLOW_BATTERY=1; shift ;;
    --force) FORCE=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

case "$ORDER" in ab | abba) ;; *) echo "--order must be ab or abba" >&2; exit 2 ;; esac
case "$CYCLES" in '' | *[!0-9]* | 0) echo "--cycles must be a positive integer" >&2; exit 2 ;; esac

fail() {
    echo "run_benchmarks: $*" >&2
    exit 1
}

# --- interpreters ---------------------------------------------------------
[ -x "$PY" ] || fail "jax-metallib interpreter not found: $PY"
if [ -z "$MPS_PY" ]; then
    for candidate in .venv-jax-mps-0.11/bin/python .venv-jax-mps/bin/python; do
        if [ -x "$candidate" ]; then
            MPS_PY="$candidate"
            break
        fi
    done
fi
[ -n "$MPS_PY" ] && [ -x "$MPS_PY" ] ||
    fail "no jax-mps interpreter; create a venv with jax-mps or pass --jax-mps-python"

# --- machine state --------------------------------------------------------
power_source() {
    pmset -g batt 2>/dev/null | head -n 1 | sed -n "s/.*'\(.*\)'.*/\1/p"
}

load1() {
    sysctl -n vm.loadavg | awk '{print $2}'
}

POWER="$(power_source)"
if [ "$POWER" != "AC Power" ] && [ -n "$POWER" ]; then
    if [ "$ALLOW_BATTERY" -eq 1 ]; then
        echo "warning: running on '$POWER'; numbers are not comparable with AC runs" >&2
    else
        fail "on '$POWER'. Plug in, or pass --allow-battery (results will not be comparable)"
    fi
fi

if [ "$FORCE" -eq 0 ]; then
    if pgrep -f 'benchmarks/vs_jax_mps.py|pytest' >/dev/null 2>&1; then
        fail "another benchmark or pytest run is active; wait for it or pass --force"
    fi
    LOAD="$(load1)"
    if awk -v l="$LOAD" -v m="$MAX_LOAD" 'BEGIN { exit !(l > m) }'; then
        fail "1-minute load is $LOAD (limit $MAX_LOAD); wait or raise --max-load"
    fi
    # Processes known to use the GPU heavily, if any of them is busy.
    busy="$(ps -Ao pcpu=,comm= | awk '$1 > 30' |
        grep -iE 'Helper \(GPU\)|MTLCompilerService|dota2|steam|unity|blender|obs' || true)"
    if [ -n "$busy" ]; then
        echo "GPU-heavy processes are busy:" >&2
        echo "$busy" >&2
        fail "close them or pass --force"
    fi
fi

# --- build ----------------------------------------------------------------
if [ "$SKIP_BUILD" -eq 0 ]; then
    BUILD_DIR=""
    for d in build/cp*-macosx_*_arm64; do
        if [ -f "$d/build.ninja" ]; then
            BUILD_DIR="$d"
            break
        fi
    done
    if [ -n "$BUILD_DIR" ]; then
        command -v cmake >/dev/null || fail "cmake not on PATH"
        echo "building $BUILD_DIR ..."
        cmake --build "$BUILD_DIR" --target pjrt_plugin_silicon >/dev/null ||
            fail "native build failed; run: cmake --build $BUILD_DIR --target pjrt_plugin_silicon"
    else
        echo "warning: no build/cp*-macosx_*_arm64 directory; using the installed library as is" >&2
    fi
fi

# --- run ------------------------------------------------------------------
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    SHA="$SHA-dirty"
fi
OUT_DIR="build/bench-runs"
mkdir -p "$OUT_DIR"
STEM="$OUT_DIR/$(date -u +%Y%m%dT%H%M%SZ)-$SHA"
LOG="$STEM.log"

args=(--jax-metallib-python "$PY" --jax-mps-python "$MPS_PY" --order "$ORDER" --cycles "$CYCLES")
[ -n "$FILTER" ] && args+=(--filter "$FILTER")

{
    echo "commit:   $SHA"
    echo "host:     $(sysctl -n machdep.cpu.brand_string 2>/dev/null), macOS $(sw_vers -productVersion 2>/dev/null)"
    echo "start:    $(date '+%Y-%m-%d %H:%M:%S'), load $(load1), power '$POWER'"
    echo "settings: order=$ORDER cycles=$CYCLES filter='${FILTER}'"
    env | grep '^JAX_METALLIB_' | sed 's/^/env:      /' || true
    echo
} | tee "$LOG"

set +e
"$PY" benchmarks/vs_jax_mps.py "${args[@]}" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e

END_POWER="$(power_source)"
{
    echo
    echo "end:      $(date '+%Y-%m-%d %H:%M:%S'), load $(load1), power '$END_POWER'"
} | tee -a "$LOG"

for ext in jsonl md; do
    if [ -f "tmp-files/vs_jax_mps.$ext" ]; then
        cp "tmp-files/vs_jax_mps.$ext" "$STEM.$ext"
    fi
done

if [ "$END_POWER" != "$POWER" ]; then
    echo "warning: power source changed during the run ('$POWER' -> '$END_POWER'); discard it" >&2
fi
echo "log: $LOG"
exit "$status"
