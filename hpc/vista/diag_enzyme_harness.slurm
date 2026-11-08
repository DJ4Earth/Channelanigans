#!/bin/bash
# Diagnostic (not a table case): why does the August 2026 Enzyme-only script compute the
# same gradient 8-18% slower than bench/driver_enzyme.jl, with the same Julia 1.11.9 and
# Manifest (envs/aug2026) on the same nodes?
#
# Runs enzyme_naive at N=$NSTEPS (default 64) in the aug2026 image, on one node, as four
# variants, one after the other:
#   driver   bench/diag_driver_enzyme_harness.jl with both switches off (= our driver)
#   cuda     ... --load-cuda on    (`using CUDA`, as the August script does)
#   reuse    ... --reuse-model on  (one model + shadows for all calls, as August)
#   august   the August script itself, run inside the image like its August job script
#            (numactl -C 0, perf stat with FIFO control, JULIA_NUM_THREADS=1)
# driver/cuda/reuse: bench/run_cases.jl with the sweep's flags, 1 warmup + $REPEATS
# measured calls, perf on (FLOPs). The repo's diag driver is mounted over the image's
# bench/driver_enzyme.jl.
# august: a copy of $AUG_SCRIPT with three edits, needed to run in the image:
#   `using BenchmarkTools` commented out (not in the image's project);
#   `@btime f(...)` -> three `@time f(...)` calls, each after GC.gc() (for a call this
#     long, @btime runs it ~3 times and prints one sample taken after a GC);
#   `jldsave(filename; ...)` -> `jldsave(filename, false, IOStream; ...)` (JLD2's mmap
#     fails under the image's Julia 1.11 on 64 KiB pages; see bench/common.jl).
# The script must have its loop as a plain `for` (no active @ad_checkpoint line).
#
#   sbatch -A Transfer-of-ECCO-glo -o $WORK/channelanigans_results/slurm_logs/%x.%j.out \
#          hpc/vista/diag_enzyme_harness.slurm             (from the repo root)
#   optional: --export=ALL,VARIANTS="driver august",NSTEPS=16,AUG_SCRIPT=/path/to/script.jl
#
# Results: $WORK/channelanigans_results/diag_enzyme_harness/<variant>/...
#
#SBATCH -J ch-aug-diag-enzyme-harness
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00

set -uo pipefail
export CH_PROFILE=aug2026
source "${HPC_DIR:-$SLURM_SUBMIT_DIR/hpc/vista}/env.sh"
load_apptainer
NSTEPS=${NSTEPS:-64}
VARIANTS=${VARIANTS:-driver cuda reuse august}
REPEATS=${REPEATS:-3}
TIMEOUT_S=${TIMEOUT_S:-4500}     # per variant: ~35 min Enzyme compile + 4-6 calls of ~2.5-3 min
AUG_SCRIPT=${AUG_SCRIPT:-/work/08955/jlk3675/vista/cpu-only/Channelanigans/scripts/abernathey_channel_no_reactant.jl}
HDRIVER=$REPO_DIR/bench/diag_driver_enzyme_harness.jl
[[ -f $HDRIVER ]] || { echo "$HDRIVER not found; git pull first" >&2; exit 2; }

echo "host: $(hostname)   job: $SLURM_JOB_ID   profile: $CH_PROFILE   N=$NSTEPS   variants: $VARIANTS"
echo "repo: $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null)   kernel: $(uname -r)   page size: $(getconf PAGESIZE)"
free -g | head -2

depot=${TMPDIR:-/tmp}/channelanigans_depot_${SLURM_JOB_ID:-$$}
mkdir -p "$depot"
container() {   # apptainer_exec (env.sh) plus extra --bind arguments given before "--"
    local binds=()
    while [[ $# -gt 0 && $1 != -- ]]; do binds+=(--bind "$1"); shift; done
    shift
    apptainer exec --cleanenv \
        --bind "$SCRATCH_ROOT:/results" \
        --bind "$WORK_ROOT:/work_results" \
        ${binds[@]+"${binds[@]}"} \
        --env "JULIA_DEPOT_PATH=$depot:/opt/julia_depot" \
        --env "SLURM_JOB_ID=${SLURM_JOB_ID:-},SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}" \
        --env "CHANNELANIGANS_IMAGE_DIGEST=$IMAGE_DIGEST" \
        "$SIF" "$@"
}

fresh_dirs() {   # fresh_dirs <subdir of the results roots>; moves old results aside
    local stamp d
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    for d in "$SCRATCH_ROOT/$1" "$WORK_ROOT/$1"; do [[ -d $d ]] && mv "$d" "$d.$stamp"; done
    mkdir -p "$SCRATCH_ROOT/$1" "$WORK_ROOT/$1"
}

rc_all=0
for variant in $VARIANTS; do
    echo; echo "===== $variant  [$(date -u +%FT%TZ)]"
    SWEEP=diag_enzyme_harness/$variant
    case $variant in
        driver|cuda|reuse)
            cuda=off; reuse=off
            [[ $variant == cuda ]] && cuda=on
            [[ $variant == reuse ]] && reuse=on
            CASE=enzyme_naive/$(printf 'N%04d' "$NSTEPS")
            fresh_dirs "$SWEEP"
            orchestrator_flag_args "$SWEEP/$CASE"
            container "$HDRIVER:/opt/channelanigans/bench/driver_enzyme.jl:ro" -- \
                julia /opt/channelanigans/bench/run_cases.jl \
                    --configs enzyme_naive --nsteps "$NSTEPS" --out "/results/$SWEEP" \
                    --timeout "$TIMEOUT_S" --perf on --numactl on --core "$CORE" "${ORCH_FLAGS[@]}" \
                    -- $DRIVER_ARGS --repeats "$REPEATS" --load-cuda "$cuda" --reuse-model "$reuse"
            rc=$?; (( rc )) && rc_all=$rc
            SCR=$SCRATCH_ROOT/$SWEEP; DST=$WORK_ROOT/$SWEEP
            cp -a "$SCR/." "$DST/" 2>/dev/null
            # Summary
            grep -h -o 'cuda_loaded = [a-z]*' "$DST/$CASE/log.txt" 2>/dev/null | head -1
            grep -h -o 'harness call [0-9]*: .*' "$DST/$CASE/log.txt" 2>/dev/null
            grep -hE '"(status|reason)"' "$DST/$CASE/status.json" 2>/dev/null
            if [[ -f $DST/$CASE/result.json ]]; then
                echo "result: $(grep -m1 -o '"runtime_s": [0-9.e+-]*' "$DST/$CASE/result.json")   $(grep -m1 -o '"flops_per_call": [0-9.e+-]*' "$DST/$CASE/result.json")   $(grep -m1 -o '"peak_rss_bytes": [0-9.e+-]*' "$DST/$CASE/result.json")"
            fi
            ;;
        august)
            [[ -f $AUG_SCRIPT ]] || { echo "AUG_SCRIPT not found: $AUG_SCRIPT" >&2; rc_all=2; continue; }
            if grep -qE '^[[:space:]]*@ad_checkpoint' "$AUG_SCRIPT"; then
                echo "$AUG_SCRIPT has an active @ad_checkpoint line; the August runs used a plain for loop" >&2
                rc_all=2; continue
            fi
            nb=$(grep -cE '^[[:space:]]*@btime ' "$AUG_SCRIPT")
            (( nb == 1 )) || { echo "expected one @btime line in $AUG_SCRIPT, found $nb" >&2; rc_all=2; continue; }
            if grep -E '^[[:space:]]*@btime ' "$AUG_SCRIPT" | grep -q '\$'; then
                echo "the @btime line in $AUG_SCRIPT interpolates (\$); edit the sed below to handle it" >&2
                rc_all=2; continue
            fi
            fresh_dirs "$SWEEP"
            WD=$SCRATCH_ROOT/$SWEEP
            cp "$AUG_SCRIPT" "$WD/august_original.jl"
            sed -e 's/^\([[:space:]]*\)using BenchmarkTools/\1# using BenchmarkTools   # diag: not in the image; @btime replaced below/' \
                -e 's/^\([[:space:]]*\)@btime \(.*\)$/\1for _k in 1:3; GC.gc(); GC.gc(); print("august-script timed call $_k: "); @time \2; end/' \
                -e 's/jldsave(filename;/jldsave(filename, false, IOStream;/' \
                "$AUG_SCRIPT" > "$WD/august_script.jl"
            echo "edits to the August script:"
            diff "$WD/august_original.jl" "$WD/august_script.jl"
            # As the August job script: JULIA_NUM_THREADS=1, its XLA_FLAGS (unused: no
            # Reactant), numactl -C 0, perf stat counting only between the script's
            # enable/disable (its counted() call).
            container -- bash -c '
                cd "/results/'"$SWEEP"'" || exit 2
                export JULIA_NUM_THREADS=1
                export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_backend_extra_options=xla_cpu_use_multi_output_fusion"
                export PERF_CTL_FIFO=/tmp/perf_ctl.$SLURM_JOB_ID PERF_ACK_FIFO=/tmp/perf_ack.$SLURM_JOB_ID
                rm -f "$PERF_CTL_FIFO" "$PERF_ACK_FIFO"; mkfifo "$PERF_CTL_FIFO" "$PERF_ACK_FIFO"
                timeout '"$TIMEOUT_S"' numactl -C '"$CORE"' perf stat -D -1 -e r80C0,r80C1 \
                    --control fifo:$PERF_CTL_FIFO,$PERF_ACK_FIFO \
                    -- julia --project=/opt/channelanigans august_script.jl '"$NSTEPS"' 0
                rc=$?; rm -f "$PERF_CTL_FIFO" "$PERF_ACK_FIFO"; exit $rc' \
                > "$WD/log.txt" 2>&1
            rc=$?; (( rc )) && rc_all=$rc
            echo "exit code $rc"
            cp -a "$WD/." "$WORK_ROOT/$SWEEP/"
            # Summary
            grep -hE 'august-script timed call|peak_rss|r80C0|r80C1|time elapsed|ERROR' "$WD/log.txt"
            ;;
        *) echo "unknown variant $variant" >&2; rc_all=2 ;;
    esac
done
echo
echo "August script today (your rerun), N=8 17.40 s, N=64 171.06 s; aug2026 sweep (our driver) N=8 16.0 s, N=64 140.6 s"
exit $rc_all
