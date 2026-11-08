#!/bin/bash
# Diagnostic (not a table case): why does the August 2026 Enzyme-only script compute the
# same gradient 8-18% slower than bench/driver_enzyme.jl, with the same Julia 1.11.9 and
# Manifest (envs/aug2026) on the same nodes?
#
# Runs enzyme_naive at N=$NSTEPS (default 64) in the aug2026 image, on one node, as
# variants run one after the other:
#   driver      bench/diag_driver_enzyme_harness.jl with all switches off (= our driver)
#   cuda        ... --load-cuda on    (`using CUDA`, as the August script does)
#   reuse       ... --reuse-model on  (one model + shadows for all calls, as August)
#   zero_ics    ... --zero-ics on     (T = S = 0 initial state, as the August script
#               had by accident: its commented-out spinup left `set!(Tᵢ, model.tracers.T)`
#               and `set!(Sᵢ, model.tracers.S)` copying never-set, zero tracers)
#   august      the August script itself, run inside the image like its August job script
#               (numactl -C 0, perf stat with FIFO control, JULIA_NUM_THREADS=1)
#   august_ics  the same, with those two set! lines commented out, so it differentiates
#               the stratified initial state (T profile + noise, S = 35) like every row
# Default: driver zero_ics august august_ics (the first run, job 1129118, used
# driver cuda reuse august; rerunning a variant moves its old results aside).
# driver/cuda/reuse/zero_ics: bench/run_cases.jl with the sweep's flags, 1 warmup +
# $REPEATS measured calls, perf on (FLOPs). The repo's diag driver is mounted over the
# image's bench/driver_enzyme.jl.
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
VARIANTS=${VARIANTS:-driver zero_ics august august_ics}
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
        driver|cuda|reuse|zero_ics)
            cuda=off; reuse=off; zero=off
            [[ $variant == cuda ]] && cuda=on
            [[ $variant == reuse ]] && reuse=on
            [[ $variant == zero_ics ]] && zero=on
            CASE=enzyme_naive/$(printf 'N%04d' "$NSTEPS")
            fresh_dirs "$SWEEP"
            orchestrator_flag_args "$SWEEP/$CASE"
            container "$HDRIVER:/opt/channelanigans/bench/driver_enzyme.jl:ro" -- \
                julia /opt/channelanigans/bench/run_cases.jl \
                    --configs enzyme_naive --nsteps "$NSTEPS" --out "/results/$SWEEP" \
                    --timeout "$TIMEOUT_S" --perf on --numactl on --core "$CORE" "${ORCH_FLAGS[@]}" \
                    -- $DRIVER_ARGS --repeats "$REPEATS" --load-cuda "$cuda" --reuse-model "$reuse" --zero-ics "$zero"
            rc=$?; (( rc )) && rc_all=$rc
            SCR=$SCRATCH_ROOT/$SWEEP; DST=$WORK_ROOT/$SWEEP
            cp -a "$SCR/." "$DST/" 2>/dev/null
            # Summary
            grep -h -o -E '(cuda_loaded|zero_ics) = [a-z]*' "$DST/$CASE/log.txt" 2>/dev/null | head -2
            grep -h -o 'harness call [0-9]*: .*' "$DST/$CASE/log.txt" 2>/dev/null
            grep -hE '"(status|reason)"' "$DST/$CASE/status.json" 2>/dev/null
            if [[ -f $DST/$CASE/result.json ]]; then
                echo "result: $(grep -m1 -o '"runtime_s": [0-9.e+-]*' "$DST/$CASE/result.json")   $(grep -m1 -o '"flops_per_call": [0-9.e+-]*' "$DST/$CASE/result.json")   $(grep -m1 -o '"peak_rss_bytes": [0-9.e+-]*' "$DST/$CASE/result.json")"
            fi
            ;;
        august|august_ics)
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
            if [[ $variant == august_ics ]]; then
                # Keep the stratified initial state: comment out the two lines that copy
                # the never-set (zero) model tracers into Tᵢ and Sᵢ.
                sed -i -e 's/^\([[:space:]]*\)\(set!(\([TS]\)ᵢ, model\.tracers\.\3)\)/\1# \2   # diag august_ics: keep the stratified initial state/' \
                    "$WD/august_script.jl"
                nics=$(grep -c 'diag august_ics' "$WD/august_script.jl")
                (( nics == 2 )) || { echo "august_ics: expected to comment out 2 set! lines, did $nics" >&2; rc_all=2; continue; }
            fi
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
echo "N=64 so far: August script, Vista setup today 171.06 s; job 1129118 (one node): driver 139.7 s, cuda 139.2 s, reuse 138.0 s, august 152.0 s"
exit $rc_all
