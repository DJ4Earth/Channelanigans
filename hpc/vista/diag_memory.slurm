#!/bin/bash
# Diagnostic (not a table case): how much memory does enzyme_naive need at N=128 in the
# aug2026 image, and does collecting garbage earlier make it fit on a gg node?
#
# The August 2026 run fit with a 222 GiB peak on a 237 GiB node; the aug2026 sweep's run
# was killed at >203 GiB (fresh gg nodes show ~217 GiB available). This job runs
# enzyme_naive at N=$NSTEPS (default 128) once per variant, on one node, without perf
# (so an OOM kill is seen and classified "m"), with the aug2026 profile's flags and the
# full protocol (1 warmup + 12 measured):
#   default        as in the sweep
#   heaphint200G   julia --heap-size-hint=200G (the GC collects more eagerly near 200 GiB)
# A host-side sampler logs the driver's RSS and the node's MemAvailable every 5 s to
# memory.csv in each variant's case directory.
#
#   sbatch -A Transfer-of-ECCO-glo -o $WORK/channelanigans_results/slurm_logs/%x.%j.out \
#          hpc/vista/diag_memory.slurm                    (from the repo root)
#   optional: --export=ALL,VARIANTS=heaphint200G,NSTEPS=64
#
# Results: $WORK/channelanigans_results/diag_memory/<variant>/enzyme_naive/N<nnnn>/
#
#SBATCH -J ch-aug-diag-memory
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00

set -uo pipefail
export CH_PROFILE=aug2026
source "${HPC_DIR:-$SLURM_SUBMIT_DIR/hpc/vista}/env.sh"
load_apptainer
NSTEPS=${NSTEPS:-128}
VARIANTS=${VARIANTS:-default heaphint200G}
TIMEOUT_S=${TIMEOUT_S:-9000}     # per variant: ~35 min compile + 13 calls of ~5 min
SAMPLE_S=5

echo "host: $(hostname)   job: $SLURM_JOB_ID   profile: $CH_PROFILE   N=$NSTEPS   variants: $VARIANTS"
echo "kernel: $(uname -r)"
free -g | head -2
grep -E 'MemTotal|MemAvailable' /proc/meminfo

# RSS (KiB) summed over the driver process(es); MemAvailable (KiB) of the node.
sample_memory() {   # sample_memory <csv>; runs until killed
    echo "utc,elapsed_s,driver_rss_kib,mem_available_kib" > "$1"
    local t0=$SECONDS rss pid
    while :; do
        rss=0
        for pid in $(pgrep -f 'bench/driver_enzyme.jl' 2>/dev/null); do
            rss=$(( rss + $(awk '/^VmRSS:/ {print $2}' /proc/$pid/status 2>/dev/null || echo 0) ))
        done
        echo "$(date -u +%FT%TZ),$((SECONDS - t0)),$rss,$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$1"
        sleep $SAMPLE_S
    done
}

rc_all=0
for variant in $VARIANTS; do
    case $variant in
        default)      jflags=$RUN_JULIA_FLAGS ;;
        heaphint200G) jflags="$RUN_JULIA_FLAGS --heap-size-hint=200G" ;;
        *) echo "unknown variant $variant" >&2; exit 2 ;;
    esac
    SWEEP=diag_memory/$variant
    CASE=enzyme_naive/$(printf 'N%04d' "$NSTEPS")
    SCR=$SCRATCH_ROOT/$SWEEP/$CASE
    DST=$WORK_ROOT/$SWEEP/$CASE
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    for d in "$SCR" "$DST"; do [[ -d $d ]] && mv "$d" "$d.$stamp"; done
    mkdir -p "$SCRATCH_ROOT/$SWEEP" "$WORK_ROOT/$SWEEP"
    CSV=$SCRATCH_ROOT/$SWEEP/memory_$(printf 'N%04d' "$NSTEPS").csv

    echo; echo "===== $variant (julia flags: '${jflags# }')  [$(date -u +%FT%TZ)]"
    sample_memory "$CSV" & sampler=$!
    apptainer_exec julia /opt/channelanigans/bench/run_cases.jl \
        --configs enzyme_naive --nsteps "$NSTEPS" --out "/results/$SWEEP" \
        --timeout "$TIMEOUT_S" --perf off --numactl on --core "$CORE" \
        --julia-flags "${jflags# }" \
        -- $DRIVER_ARGS
    rc=$?; (( rc )) && rc_all=$rc
    kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
    mkdir -p "$DST"; cp -a "$SCR/." "$DST/" 2>/dev/null; cp "$CSV" "$DST/memory.csv"

    # Summary
    grep -hE '"(status|reason|termsignal|elapsed_s)"' "$DST/status.json" 2>/dev/null
    awk -F, 'NR > 1 && $3 > max { max = $3; t = $2 }
             NR > 1 && (min == "" || $4 < min) { min = $4 }
             END { printf "peak driver RSS %.1f GiB at %d s; lowest MemAvailable %.1f GiB\n",
                          max / 2^20, t, min / 2^20 }' "$DST/memory.csv"
    if [[ -f $DST/result.json ]]; then
        echo "result: $(grep -m1 -o '"runtime_s": [0-9.]*' "$DST/result.json")   $(grep -m1 -o '"peak_rss_bytes": [0-9.e+]*' "$DST/result.json")"
    else
        echo "no result.json; calls finished: $(grep -c '"wall_s"' "$DST/progress.json" 2>/dev/null || echo 0)"
    fi
done
echo
echo "August 2026, enzyme_naive N=128: 477.7 s per call, 222.0 GiB peak RSS (237 GiB node)"
exit $rc_all
