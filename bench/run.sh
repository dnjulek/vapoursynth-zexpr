#!/usr/bin/env bash
# Sequential vspipe benchmark runner: zexpr vs akarin (and cranexpr with CRANEXPR=1) across formats.
# Every script exposes 9 outputs:
#   0..2  YUV420P8   (0=zexpr, 1=akarin, 2=cranexpr)
#   3..5  YUV420P16  (3=zexpr, 4=akarin, 5=cranexpr)
#   6..8  YUV420PS   (6=zexpr, 7=akarin, 8=cranexpr)
#
# Always runs one case, format, and output at a time (never parallel, never under Intel SDE).
#
# Usage:
#   bash bench/run.sh                           # all scripts x all formats (p8, p16, ps)
#   FORMATS="p8" bash bench/run.sh              # only YUV420P8 (-o 0..2)
#   FORMATS="p8 ps" bash bench/run.sh gamma.py  # subset of formats and scripts
#   THREADS=1 bash bench/run.sh                 # force one thread everywhere (default: p8 auto, p16/ps 1T)
#   THREADS=auto bash bench/run.sh              # VapourSynth default threads everywhere
#   CRANEXPR=1 bash bench/run.sh               # also time cranexpr (outputs 2/5/8)
#   FRAMES=500 bash bench/run.sh                # fixed frame count for every run (default: ~1.5 s each)
#   BENCH_SECONDS=5 bash bench/run.sh           # target seconds per plugin run
#   BENCH_LOG=runs.txt bash bench/run.sh        # log frames/seconds/fps of every measured run

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

THREADS="${THREADS:-}"
FRAMES_OVERRIDE="${FRAMES:-}"
FORMATS_STR="${FORMATS:-p8 p16 ps}"
read -r -a FORMAT_LIST <<< "$FORMATS_STR"


# Output index base per format tag (a function, not an associative array: macOS /bin/bash is 3.2).
fmt_base() {
    case "$1" in
        p8) echo 0 ;;
        p16) echo 3 ;;
        ps) echo 6 ;;
        *) return 1 ;;
    esac
}

# Threads per cell: p8 runs at VapourSynth's default thread count, while p16 and ps run single-threaded,
# because at full thread counts their 2-4x larger frames saturate DRAM bandwidth and all three plugins
# converge on the same memory-bound fps. THREADS=N forces N threads for every cell (THREADS=0 or auto =
# VapourSynth's default everywhere).
cell_threads() {
    if [[ -n "$THREADS" ]]; then
        [[ "$THREADS" == "auto" ]] && echo 0 || echo "$THREADS"
        return
    fi
    case "$1" in
        p8) echo 0 ;;
        *) echo 1 ;;
    esac
}
thread_label() {
    [[ "$1" == "0" ]] && echo "auto" || echo "${1}T"
}

# Run length: bench/probe.py first times every output of a script in one Python process (~0.25 s
# each), then each plugin gets ONE vspipe launch sized to last about BENCH_SECONDS (default 1.5 s).
# run.sh passes `-a fmt=` so the script builds only that format and generates its source frames before
# vspipe starts timing; vspipe's own fps then covers only the expression.
TARGET_SECONDS="${BENCH_SECONDS:-1.5}"
MAX_FRAMES=100000   # random_source loops 8 frames up to 100000
PROBE_SECONDS="${PROBE_SECONDS:-0.25}"

# Ordered by zexpr's speedup over akarin (geometric mean of p8/p16/ps), biggest first.
ALL_CASES=(
    deblock.py
    gamma.py
    kirsch.py
    morpho.py
    median3x3.py
    sobel.py
    exp.py
    rotate.py
    box3x3.py
    wide-reach.py
)

if [[ $# -gt 0 ]]; then
    CASES=("$@")
else
    CASES=("${ALL_CASES[@]}")
fi

VSPIPE_BIN="$(command -v vspipe || true)"
if [[ -z "$VSPIPE_BIN" ]]; then
    echo "ERROR: vspipe not found in PATH (activate the Python environment that has VapourSynth installed)" >&2
    exit 1
fi
PYTHON_BIN="${PYTHON:-$(dirname "$VSPIPE_BIN")/python3}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3 || command -v python)"

# zexpr vs akarin by default; CRANEXPR=1 adds cranexpr (outputs 2/5/8, pip install vapoursynth-cranexpr).
WITH_CRANEXPR="${CRANEXPR:-0}"
[[ "$WITH_CRANEXPR" == "0" || "$WITH_CRANEXPR" == "1" ]] || { echo "ERROR: CRANEXPR must be 0 or 1" >&2; exit 1; }

# Fail fast with the real error instead of a table of FAIL cells: every case needs this
# environment's vspipe to import numpy and load zexpr and akarin (and cranexpr with CRANEXPR=1).
PREFLIGHT_OUT=1
[[ "$WITH_CRANEXPR" == "1" ]] && PREFLIGHT_OUT=2
# Render one frame (not --info: vspipe --info accepts a nonexistent -o index).
if ! preflight="$(vspipe --end 0 -o "$PREFLIGHT_OUT" -a fmt=p8 "bench/${CASES[0]#bench/}" -- 2>&1 >/dev/null)"; then
    echo "ERROR: bench/${CASES[0]#bench/} output $PREFLIGHT_OUT does not load with $VSPIPE_BIN:" >&2
    echo "$preflight" | grep -v '^[[:space:]]*$' | tail -n 3 | sed 's/^/    /' >&2
    echo "Activate the environment with VapourSynth, or install what is missing:" >&2
    echo "    pip install VapourSynth numpy vapoursynth-akarin   # + vapoursynth-cranexpr for CRANEXPR=1" >&2
    echo "    zig build -Doptimize=ReleaseFast   # zexpr -> zig-out/lib/libzexpr.so (.dylib on macOS)" >&2
    exit 1
fi
# Show common.py's warnings (such as a pip-installed zexpr being timed) once, not per cell.
echo "$preflight" | grep '^WARNING:' | sort -u >&2 || true

if [[ -n "$FRAMES_OVERRIDE" ]]; then RUN_DESC="$FRAMES_OVERRIDE frames"; else RUN_DESC="~${TARGET_SECONDS}s"; fi
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ -r /proc/cpuinfo ]]; then
    CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs || true)"
else
    CPU_MODEL="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"   # macOS
fi
CPU_MODEL="${CPU_MODEL:-$(uname -m)}"
echo "# commit=$COMMIT cpu=\"$CPU_MODEL\" threads=\"${THREADS:-p8:auto p16,ps:1T}\" formats=\"${FORMAT_LIST[*]}\" run=\"$RUN_DESC\" vspipe=$VSPIPE_BIN"
if [[ "$WITH_CRANEXPR" == "1" ]]; then
    printf "%-22s %13s %13s %13s %10s %10s\n" "case" "zexpr" "akarin" "cranexpr" "vs_akarin" "vs_crane"
    printf "%-22s %13s %13s %13s %10s %10s\n" "----------------------" "-------------" "-------------" "-------------" "----------" "----------"
else
    printf "%-22s %13s %13s %10s\n" "case" "zexpr" "akarin" "vs_akarin"
    printf "%-22s %13s %13s %10s\n" "----------------------" "-------------" "-------------" "----------"
fi

# vs_run SCRIPT FMT OUTPUT FRAMES THREADS -> "fps seconds", or FAIL
vs_run() {
    local err
    if ! err="$(vspipe -p -o "$3" --end "$(($4 - 1))" -a "fmt=$2" -a "threads=$5" "bench/$1" -- 2>&1 >/dev/null)"; then
        echo "FAIL"
        return 0
    fi
    err="$(echo "$err" | sed -n 's/.*Output [0-9]* frames in \([0-9.]*\) seconds (\([0-9.]*\) fps).*/\2 \1/p' | tail -n1)"
    echo "${err:-FAIL}"
}

# probe_script SCRIPT -> sets PROBE_RESULT ("OUT FPS" lines) for every output this run will time,
# measured by bench/probe.py instead of extra vspipe launches. One Python process per thread count:
# a VapourSynth core's thread pool keeps the size it was created with, so p8 (auto) and p16/ps (1T)
# cannot share a process.
probe_script() {
    local fmt base outs t group_threads specs
    PROBE_RESULT=""
    for group_threads in $(for fmt in "${FORMAT_LIST[@]}"; do cell_threads "$fmt"; done | sort -u); do
        specs=()
        for fmt in "${FORMAT_LIST[@]}"; do
            base="$(fmt_base "$fmt")" || continue
            t="$(cell_threads "$fmt")"
            [[ "$t" == "$group_threads" ]] || continue
            outs="$base,$((base + 1))"
            [[ "$WITH_CRANEXPR" == "1" ]] && outs="$outs,$((base + 2))"
            specs+=("$fmt:$t:$outs")
        done
        PROBE_RESULT="$PROBE_RESULT
$("$PYTHON_BIN" -B bench/probe.py "bench/$1" "$PROBE_SECONDS" "${specs[@]}" 2>/dev/null || true)"
    done
}

# run_one SCRIPT FMT OUTPUT THREADS -> fps, or FAIL. One vspipe launch, sized from the probe to last
# about TARGET_SECONDS; with `-a fmt=` the sources are generated before vspipe's timer starts, so its
# own fps measures only the expression.
run_one() {
    local script="$1" fmt="$2" out_idx="$3" threads="$4" n res fps secs pfps
    if [[ -n "$FRAMES_OVERRIDE" ]]; then
        n="$FRAMES_OVERRIDE"
    else
        pfps="$(echo "$PROBE_RESULT" | awk -v o="$out_idx" '$1 == o { print $2 }')"
        if [[ -z "$pfps" || "$pfps" == "FAIL" ]]; then
            echo "FAIL"
            return 0
        fi
        n="$(awk -v f="$pfps" -v t="$TARGET_SECONDS" -v hi="$MAX_FRAMES" \
            'BEGIN { m = int(f * t); if (m < 8) m = 8; if (m > hi) m = hi; print m }')"
    fi
    res="$(vs_run "$script" "$fmt" "$out_idx" "$n" "$threads")"
    if [[ "$res" == "FAIL" ]]; then
        echo "FAIL"
        return 0
    fi
    fps="${res%% *}"
    secs="${res#* }"
    if [[ -n "${BENCH_LOG:-}" ]]; then
        echo "$script $out_idx $threads $n $secs $fps" >>"$BENCH_LOG"
    fi
    echo "$fps"
}

fmt_cell() {
    if [[ "$1" == "FAIL" ]]; then
        echo "FAIL"
    else
        echo "${1} fps"
    fi
}

for script in "${CASES[@]}"; do
    script="${script#bench/}"
    PROBE_RESULT=""
    [[ -z "$FRAMES_OVERRIDE" ]] && probe_script "$script"
    for fmt in "${FORMAT_LIST[@]}"; do
        if ! base="$(fmt_base "$fmt")"; then
            echo "ERROR: unknown format '$fmt' (expected p8, p16, or ps)" >&2
            exit 1
        fi
        threads="$(cell_threads "$fmt")"
        f0="$(run_one "$script" "$fmt" "$((base + 0))" "$threads")"
        f1="$(run_one "$script" "$fmt" "$((base + 1))" "$threads")"
        r1="$(awk -v a="$f0" -v b="$f1" 'BEGIN { if (a != "FAIL" && b != "FAIL" && b+0 > 0) printf "%.2fx", a/b; else print "-" }')"
        label="${script%.py} ($fmt, $(thread_label "$threads"))"
        if [[ "$WITH_CRANEXPR" == "1" ]]; then
            f2="$(run_one "$script" "$fmt" "$((base + 2))" "$threads")"
            r2="$(awk -v a="$f0" -v b="$f2" 'BEGIN { if (a != "FAIL" && b != "FAIL" && b+0 > 0) printf "%.2fx", a/b; else print "-" }')"
            printf "%-22s %13s %13s %13s %10s %10s\n" "$label" "$(fmt_cell "$f0")" "$(fmt_cell "$f1")" "$(fmt_cell "$f2")" "$r1" "$r2"
        else
            printf "%-22s %13s %13s %10s\n" "$label" "$(fmt_cell "$f0")" "$(fmt_cell "$f1")" "$r1"
        fi
    done
done
