#!/bin/bash
# Diagnostic (not a table case): does the environment of the August 2026 Enzyme-only runs
# (host Julia 1.11.9 + the August depot on $WORK, TACC modules, XALT preloaded) make the
# same gradient slower than the aug2026 image does, when both run on the same node?
#
# Runs the August Enzyme-only script at N=$NSTEPS (default 64), one variant after the other:
#   host        as the August job script (evidence/enzyme_only_job_script.sh): the job's
#               inherited environment, `module remove cuda nvidia`, its exports, then
#               numactl -C 0 perf stat ... -- julia --project=$AUG_PROJECT
#   host_clean  the same, with LD_PRELOAD (TACC's XALT) and LD_LIBRARY_PATH removed from
#               the julia process and XALT_EXECUTABLE_TRACKING=no
#   image       the same script in the aug2026 image (Julia 1.11.9 + envs/aug2026),
#               as the `august` variant of diag_enzyme_harness.slurm
# All three run one identical copy of $AUG_SCRIPT with the edits the image needs (and
# which change nothing on the host), so only the environment differs:
#   `using BenchmarkTools` commented out (not in the image's project);
#   `@btime f(...)` -> three `@time f(...)` calls, each after GC.gc();
#   `jldsave(filename; ...)` -> `jldsave(filename, false, IOStream; ...)` (JLD2's mmap
#     fails under the image's Julia 1.11 on 64 KiB pages; see bench/common.jl).
# The script runs from a scratch directory (it writes its JLD2 files to the working
# directory), not from $AUG_PROJECT.
#
# Submit from the repo root, from your normal login shell: the job inherits that
# environment (sbatch --export=ALL), as the August jobs did.
#   sbatch -A Transfer-of-ECCO-glo -o $WORK/channelanigans_results/slurm_logs/%x.%j.out \
#          hpc/vista/diag_enzyme_env.slurm
#   optional: --export=ALL,VARIANTS="host image",NSTEPS=16,AUG_PROJECT=/path,AUG_SCRIPT=/path
#
# Results: $WORK/channelanigans_results/diag_enzyme_env/<variant>/log.txt
#
#SBATCH -J ch-aug-diag-enzyme-env
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

set -uo pipefail
export CH_PROFILE=aug2026
source "${HPC_DIR:-$SLURM_SUBMIT_DIR/hpc/vista}/env.sh"
NSTEPS=${NSTEPS:-64}
VARIANTS=${VARIANTS:-host host_clean image}
TIMEOUT_S=${TIMEOUT_S:-4500}     # per variant: ~35 min Enzyme compile + 6 calls of ~2.5-3 min
AUG_PROJECT=${AUG_PROJECT:-/work/08955/jlk3675/vista/cpu-only/Channelanigans}
AUG_SCRIPT=${AUG_SCRIPT:-$AUG_PROJECT/scripts/abernathey_channel_no_reactant.jl}
ROOT=diag_enzyme_env

echo "host: $(hostname)   job: $SLURM_JOB_ID   N=$NSTEPS   variants: $VARIANTS"
echo "kernel: $(uname -r)   page size: $(getconf PAGESIZE)   host glibc: $(ldd --version 2>&1 | head -1)"
free -g | head -2

# Lmod's `module` is not safe under `set -u` (see load_apptainer in env.sh).
module_quiet() { local saved=$-; set +eu; module "$@"; [[ $saved == *u* ]] && set -u; return 0; }

# ---- The script, edited once; every variant runs this same copy ---------------------
[[ -f $AUG_SCRIPT ]] || { echo "AUG_SCRIPT not found: $AUG_SCRIPT" >&2; exit 2; }
if grep -qE '^[[:space:]]*@ad_checkpoint' "$AUG_SCRIPT"; then
    echo "$AUG_SCRIPT has an active @ad_checkpoint line; the August runs used a plain for loop" >&2; exit 2
fi
nb=$(grep -cE '^[[:space:]]*@btime ' "$AUG_SCRIPT")
(( nb == 1 )) || { echo "expected one @btime line in $AUG_SCRIPT, found $nb" >&2; exit 2; }
if grep -E '^[[:space:]]*@btime ' "$AUG_SCRIPT" | grep -q '\$'; then
    echo "the @btime line in $AUG_SCRIPT interpolates (\$); edit the sed below to handle it" >&2; exit 2
fi
stamp=$(date -u +%Y%m%dT%H%M%SZ)
for d in "$SCRATCH_ROOT/$ROOT" "$WORK_ROOT/$ROOT"; do [[ -d $d ]] && mv "$d" "$d.$stamp"; done
mkdir -p "$SCRATCH_ROOT/$ROOT" "$WORK_ROOT/$ROOT"
cp "$AUG_SCRIPT" "$SCRATCH_ROOT/$ROOT/august_original.jl"
sed -e 's/^\([[:space:]]*\)using BenchmarkTools/\1# using BenchmarkTools   # diag: not in the image; @btime replaced below/' \
    -e 's/^\([[:space:]]*\)@btime \(.*\)$/\1for _k in 1:3; GC.gc(); GC.gc(); print("august-script timed call $_k: "); @time \2; end/' \
    -e 's/jldsave(filename;/jldsave(filename, false, IOStream;/' \
    "$AUG_SCRIPT" > "$SCRATCH_ROOT/$ROOT/august_script.jl"
echo; echo "script: $AUG_SCRIPT   project (host variants): $AUG_PROJECT"
echo "edits (same for all variants):"
diff "$SCRATCH_ROOT/$ROOT/august_original.jl" "$SCRATCH_ROOT/$ROOT/august_script.jl"

rc_all=0
for variant in $VARIANTS; do
    echo; echo "===== $variant  [$(date -u +%FT%TZ)]"
    WD=$SCRATCH_ROOT/$ROOT/$variant
    mkdir -p "$WD"
    cp "$SCRATCH_ROOT/$ROOT/august_script.jl" "$WD/"
    case $variant in
        host|host_clean)
            # A subshell, so the module changes and exports stay within this variant.
            (
                cd "$WD" || exit 2
                # As evidence/enzyme_only_job_script.sh
                module_quiet remove cuda nvidia
                export XLA_FLAGS="--xla_backend_extra_options=xla_cpu_use_multi_output_fusion"
                export JULIA_NUM_THREADS=1
                export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_dump_to=$SCRATCH/hlo_$SLURM_JOB_ID --xla_backend_extra_options=xla_cpu_use_multi_output_fusion"
                export PERF_CTL_FIFO=/tmp/perf_ctl.$SLURM_JOB_ID PERF_ACK_FIFO=/tmp/perf_ack.$SLURM_JOB_ID
                rm -f "$PERF_CTL_FIFO" "$PERF_ACK_FIFO"; mkfifo "$PERF_CTL_FIFO" "$PERF_ACK_FIFO"
                clean=()
                [[ $variant == host_clean ]] && clean=(env -u LD_PRELOAD -u LD_LIBRARY_PATH XALT_EXECUTABLE_TRACKING=no)
                echo "julia: $(command -v julia)"
                echo "JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-<unset>}"
                if [[ $variant == host_clean ]]; then
                    echo "julia process: LD_PRELOAD and LD_LIBRARY_PATH unset, XALT_EXECUTABLE_TRACKING=no"
                else
                    echo "LD_PRELOAD=${LD_PRELOAD:-<unset>}"
                    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
                fi
                ${clean[@]+"${clean[@]}"} timeout "$TIMEOUT_S" numactl -C "$CORE" perf stat -D -1 -e r80C0,r80C1 \
                    --control fifo:$PERF_CTL_FIFO,$PERF_ACK_FIFO \
                    -- julia --project="$AUG_PROJECT" august_script.jl "$NSTEPS" 0
                rc=$?; rm -f "$PERF_CTL_FIFO" "$PERF_ACK_FIFO"; exit $rc
            ) > "$WD/log.txt" 2>&1
            rc=$?
            grep -hE '^(julia:|JULIA_DEPOT_PATH=|LD_PRELOAD=|julia process:)' "$WD/log.txt"
            ;;
        image)
            ( load_apptainer
              depot=${TMPDIR:-/tmp}/channelanigans_depot_${SLURM_JOB_ID:-$$}
              mkdir -p "$depot"
              apptainer exec --cleanenv \
                --bind "$SCRATCH_ROOT:/results" \
                --env "JULIA_DEPOT_PATH=$depot:/opt/julia_depot" \
                --env "SLURM_JOB_ID=${SLURM_JOB_ID:-}" \
                "$SIF" bash -c '
                    cd "/results/'"$ROOT/$variant"'" || exit 2
                    echo "image glibc: $(ldd --version 2>&1 | head -1)"
                    export JULIA_NUM_THREADS=1
                    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_backend_extra_options=xla_cpu_use_multi_output_fusion"
                    export PERF_CTL_FIFO=/tmp/perf_ctl.$SLURM_JOB_ID PERF_ACK_FIFO=/tmp/perf_ack.$SLURM_JOB_ID
                    rm -f "$PERF_CTL_FIFO" "$PERF_ACK_FIFO"; mkfifo "$PERF_CTL_FIFO" "$PERF_ACK_FIFO"
                    timeout '"$TIMEOUT_S"' numactl -C '"$CORE"' perf stat -D -1 -e r80C0,r80C1 \
                        --control fifo:$PERF_CTL_FIFO,$PERF_ACK_FIFO \
                        -- julia --project=/opt/channelanigans august_script.jl '"$NSTEPS"' 0
                    rc=$?; rm -f "$PERF_CTL_FIFO" "$PERF_ACK_FIFO"; exit $rc'
            ) > "$WD/log.txt" 2>&1
            rc=$?
            grep -h '^image glibc:' "$WD/log.txt"
            ;;
        *) echo "unknown variant $variant" >&2; rc_all=2; continue ;;
    esac
    (( rc )) && rc_all=$rc
    echo "exit code $rc"
    mkdir -p "$WORK_ROOT/$ROOT/$variant"
    cp -a "$WD/." "$WORK_ROOT/$ROOT/$variant/"
    # Summary
    grep -hE 'august-script timed call|peak_rss|r80C0|r80C1|time elapsed|ERROR' "$WD/log.txt"
done
cp "$SCRATCH_ROOT/$ROOT"/august_*.jl "$WORK_ROOT/$ROOT/"
echo
echo "N=64 so far: August script, Vista setup 171.06 s (separate job); in the image: 152.0 s (job 1129118), 150-159 s (job 1129580)"
exit $rc_all
