#!/bin/bash
# Diagnostic (not a table case): is the explicit first-step init (initialize! +
# update_state! before the loop, see bench/driver_reactant.jl) what makes the aug2026
# Reactant runs slower than the original July/August 2026 runs?
#
# Runs reactant_full at N=$NSTEPS (default 8) in the aug2026 image, on one node, once
# with the init off (the program of the August scripts) and once with it on (as in the
# sweep), with the sweep's flags and protocol (1 warmup + 12 measured, perf, numactl).
# The repo's bench/driver_reactant.jl is mounted over the image's copy, because only
# the repo's version has --first-step-init; the rest of the image is unchanged.
#
#   sbatch -A Transfer-of-ECCO-glo -o $WORK/channelanigans_results/slurm_logs/%x.%j.out \
#          hpc/vista/diag_firststep.slurm                 (from the repo root)
#   optional: --export=ALL,NSTEPS=16,MODES=off
#
# Results: $WORK/channelanigans_results/diag_firststep/init_<mode>/reactant_full/N<nnnn>/
#
#SBATCH -J ch-aug-diag-firststep
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00

set -uo pipefail
export CH_PROFILE=aug2026
source "${HPC_DIR:-$SLURM_SUBMIT_DIR/hpc/vista}/env.sh"
load_apptainer
NSTEPS=${NSTEPS:-8}
MODES=${MODES:-off on}
TIMEOUT_S=${TIMEOUT_S:-2400}     # per mode; reactant_full N=8 needs ~20 min
DRIVER=$REPO_DIR/bench/driver_reactant.jl
grep -q first-step-init "$DRIVER" || { echo "$DRIVER has no --first-step-init; git pull first" >&2; exit 2; }

echo "host: $(hostname)   job: $SLURM_JOB_ID   profile: $CH_PROFILE   N=$NSTEPS   modes: $MODES"
echo "driver: $DRIVER ($(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null))"
rc_all=0
for mode in $MODES; do
    SWEEP=diag_firststep/init_$mode
    CASE=reactant_full/$(printf 'N%04d' "$NSTEPS")
    SCR=$SCRATCH_ROOT/$SWEEP/$CASE
    DST=$WORK_ROOT/$SWEEP/$CASE
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    for d in "$SCR" "$DST"; do [[ -d $d ]] && mv "$d" "$d.$stamp"; done
    mkdir -p "$SCRATCH_ROOT/$SWEEP" "$WORK_ROOT/$SWEEP"

    orchestrator_flag_args "$SWEEP/$CASE"
    HLO=$SCRATCH_ROOT/_hlo/$SWEEP/$CASE
    [[ ${XLA_DUMP:-0} == 1 ]] && { rm -rf "$HLO"; mkdir -p "$HLO"; }

    echo; echo "===== first-step init $mode  [$(date -u +%FT%TZ)]"
    # apptainer_exec (env.sh) plus one bind for the driver
    depot=${TMPDIR:-/tmp}/channelanigans_depot_${SLURM_JOB_ID:-$$}
    mkdir -p "$depot"
    apptainer exec --cleanenv \
        --bind "$SCRATCH_ROOT:/results" \
        --bind "$WORK_ROOT:/work_results" \
        --bind "$DRIVER:/opt/channelanigans/bench/driver_reactant.jl:ro" \
        --env "JULIA_DEPOT_PATH=$depot:/opt/julia_depot" \
        --env "SLURM_JOB_ID=${SLURM_JOB_ID:-},SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}" \
        --env "CHANNELANIGANS_IMAGE_DIGEST=$IMAGE_DIGEST" \
        "$SIF" julia /opt/channelanigans/bench/run_cases.jl \
            --configs reactant_full --nsteps "$NSTEPS" --out "/results/$SWEEP" \
            --timeout "$TIMEOUT_S" --perf on --numactl on --core "$CORE" "${ORCH_FLAGS[@]}" \
            -- $DRIVER_ARGS --first-step-init "$mode"
    rc=$?; (( rc )) && rc_all=$rc
    [[ ${XLA_DUMP:-0} == 1 && -d "$SCR" ]] && xla_total_bytes "$HLO" > "$SCR/xla_buffer_assignment.txt"
    mkdir -p "$DST"; cp -a "$SCR/." "$DST/"

    # Summary: per-call times, FLOPs, objective, XLA cost analysis
    apptainer_exec julia -e '
        using JSON
        r = JSON.parsefile(ARGS[1])
        meas = [c for c in r["calls"] if c["kind"] == "measured"]
        println("first_step_init: ", r["reactant_options"]["first_step_init"],
                "   runtime ", round(r["runtime_s"], digits=2), " ± ",
                round(r["runtime_ci95_halfwidth_s"], digits=2), " s (n=", length(meas), ")",
                "   peak RSS ", round(r["peak_rss_bytes"] / 2^30, digits=2), " GiB")
        println("compile ", round(r["compile_s"], digits=1), " s   XLA cost-analysis FLOPs ",
                r["xla_cost_analysis_flops"], "   objective ", r["objective"])
        println("perf FLOPs per call: ", [get(c, "flops", nothing) for c in meas])' \
        "/work_results/$SWEEP/$CASE/result.json" 2>&1 | grep -v gocryptfs
done
echo
echo "August (original logs), reactant_full: N=8 21.3 s, N=16 43.9 s; XLA cost-analysis FLOPs 7.99e9"
exit $rc_all
