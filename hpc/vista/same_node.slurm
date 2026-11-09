#!/bin/bash
# Same-node runtime comparison (not a table sweep): several configurations at several N,
# one after the other in ONE job, so they all run on the same gg node.
#
# Why: the table's cases each ran on whichever node was free, and runtimes differ by up
# to ~20% between Vista nodes (data/vista_2026-11_aug2026/README.md, "Reading the
# table"). Runtime differences between rows of that size cannot be attributed to the
# configurations. Running the rows to be compared on one node removes that spread.
#
# Each case is measured exactly as in the sweep (case.slurm): the image's orchestrator,
# the profile's Julia/XLA flags, numactl -C $CORE, perf-counted FLOPs, a fresh process
# per case, 1 warmup + $REPEATS measured calls. Cases run N by N, configs in the order
# given, so same-N cases run close together in time.
#
# Submit from the repo root on a login node, e.g.
#   N = 16 and 64 (about 6.5 h with 12 calls):
#     sbatch -A Transfer-of-ECCO-glo -t 09:00:00 \
#            -o $WORK/channelanigans_results/slurm_logs/%x.%j.out \
#            hpc/vista/same_node.slurm
#   N = 128 alone (about 7.5 h with 12 calls):
#     sbatch -A Transfer-of-ECCO-glo -t 10:00:00 --export=ALL,NSTEPS=128 \
#            -o $WORK/channelanigans_results/slurm_logs/%x.%j.out \
#            hpc/vista/same_node.slurm
#
# Settings (override with --export=ALL,NAME=value,...):
#   CH_PROFILE  image profile from env.sh                     (default aug2026)
#   CONFIGS     space-separated config names, run in this order
#               (default "reactant_full reactant_after_enzyme reactant_binomial_log2 enzyme_naive")
#   NSTEPS      space-separated timestep counts                (default "16 64")
#   REPEATS     measured calls per case                        (default 12, as the table)
#   PERF        on | off: count FLOPs with perf                (default on)
#               enzyme_naive at N >= 128 always runs with perf off, as its table entry
#               did: it needs ~221 GiB, and under perf an out-of-memory kill is
#               recorded as "e" instead of "m" (see case.slurm).
#   MARGIN_S    time kept back at the end of the job for copying results (default 900)
#   MIN_CASE_S  a case is skipped if less than this remains     (default 1800)
#
# Each case gets the time left in the job (minus MARGIN_S) as its timeout, so a slow
# case is recorded as "t" and copied back before SLURM ends the job; later cases that
# no longer fit are skipped and listed in the summary.
#
# Results: $WORK/channelanigans_results/same_node_<profile>/<jobid>/
#   <config>/N<nnnn>/      as in the sweep, plus job_info.json (node, order, free memory
#                          before the case)
#   node_info.txt          the node at job start
#   summary.tsv            one line per case: order, config, N, status, mean runtime, start/end
#   table.tex, table.csv   make_table.jl over this job's cases (runtime, memory, FLOPs, CIs)
#
#SBATCH -J ch-aug-same-node
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 09:00:00

set -uo pipefail
export CH_PROFILE=${CH_PROFILE:-aug2026}
source "${HPC_DIR:-$SLURM_SUBMIT_DIR/hpc/vista}/env.sh"
load_apptainer

CONFIGS=${CONFIGS:-reactant_full reactant_after_enzyme reactant_binomial_log2 enzyme_naive}
NSTEPS=${NSTEPS:-16 64}
REPEATS=${REPEATS:-12}
PERF=${PERF:-on}
MARGIN_S=${MARGIN_S:-900}
MIN_CASE_S=${MIN_CASE_S:-1800}
SWEEP=same_node_${CH_PROFILE}/$SLURM_JOB_ID
OUT_SCR=$SCRATCH_ROOT/$SWEEP
OUT_WORK=$WORK_ROOT/$SWEEP
mkdir -p "$OUT_SCR" "$OUT_WORK"

# Seconds left in this job, from squeue's %L ([D-]HH:MM:SS, MM:SS or SS).
remaining_s() {
    local left d=0 h=0 m=0 s=0
    left=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null)
    [[ -z $left || $left == INVALID || $left == UNLIMITED ]] && { echo 172800; return; }
    if [[ $left == *-* ]]; then d=${left%%-*}; left=${left#*-}; fi
    local IFS=:; set -- $left
    case $# in
        3) h=$1; m=$2; s=$3 ;;
        2) m=$1; s=$2 ;;
        1) s=$1 ;;
    esac
    echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}
mem_available_kib() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }

{
    echo "host: $(hostname)   job: $SLURM_JOB_ID   profile: $CH_PROFILE   image: $IMAGE_DIGEST"
    echo "configs: $CONFIGS"
    echo "nsteps: $NSTEPS   repeats: $REPEATS   perf: $PERF   core: $CORE"
    echo "kernel: $(uname -r)   page size: $(getconf PAGESIZE)"
    echo "start: $(date -u +%FT%TZ)   time left: $(remaining_s) s"
    free -g | head -2
    grep -E 'MemTotal|MemAvailable' /proc/meminfo
} | tee "$OUT_WORK/node_info.txt"

hpc_commit=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)
[[ $hpc_commit != unknown ]] && ! git -C "$REPO_DIR" diff --quiet HEAD -- hpc && hpc_commit="$hpc_commit-dirty"

printf 'order\tconfig\tnsteps\tstatus\twall_s_mean\tstart_utc\tend_utc\n' > "$OUT_WORK/summary.tsv"
rc_all=0
order=0
for N in $NSTEPS; do
    for CONFIG in $CONFIGS; do
        order=$(( order + 1 ))
        CASE=$CONFIG/$(printf 'N%04d' "$N")
        SCR=$OUT_SCR/$CASE
        DST=$OUT_WORK/$CASE

        left=$(remaining_s)
        timeout_s=$(( left - MARGIN_S ))
        if (( timeout_s < MIN_CASE_S )); then
            echo; echo "===== [$order] $CASE skipped: ${left}s left in the job"
            printf '%s\t%s\t%s\tskipped\t\t\t\n' "$order" "$CONFIG" "$N" >> "$OUT_WORK/summary.tsv"
            continue
        fi
        perf=$PERF
        [[ $CONFIG == enzyme_naive ]] && (( N >= 128 )) && perf=off

        orchestrator_flag_args "$SWEEP/$CASE"
        HLO=$SCRATCH_ROOT/_hlo/$SWEEP/$CASE
        [[ ${XLA_DUMP:-0} == 1 ]] && { rm -rf "$HLO"; mkdir -p "$HLO"; }

        mem_before=$(mem_available_kib)
        start=$(date -u +%FT%TZ)
        echo; echo "===== [$order] $CASE  [$start]  timeout ${timeout_s}s  perf $perf  MemAvailable $(( mem_before / 1048576 )) GiB"
        apptainer_exec julia /opt/channelanigans/bench/run_cases.jl \
            --configs "$CONFIG" --nsteps "$N" --out "/results/$SWEEP" \
            --timeout "$timeout_s" --perf "$perf" --numactl on --core "$CORE" "${ORCH_FLAGS[@]}" \
            -- $DRIVER_ARGS --repeats "$REPEATS"
        rc=$?; (( rc )) && rc_all=$rc
        end=$(date -u +%FT%TZ)

        [[ ${XLA_DUMP:-0} == 1 && -d "$SCR" ]] && xla_total_bytes "$HLO" > "$SCR/xla_buffer_assignment.txt"
        mkdir -p "$DST"
        cp -a "$SCR/." "$DST/" 2>/dev/null
        cat > "$DST/job_info.json" <<EOF
{
  "slurm_job_id": "$SLURM_JOB_ID",
  "slurm_job_name": "${SLURM_JOB_NAME:-}",
  "partition": "${SLURM_JOB_PARTITION:-}",
  "account": "${SLURM_JOB_ACCOUNT:-$ACCOUNT}",
  "node": "$(hostname)",
  "start_utc": "$start",
  "end_utc": "$end",
  "same_node_order": $order,
  "same_node_configs": "$CONFIGS",
  "same_node_nsteps": "$NSTEPS",
  "mem_available_before_kib": $mem_before,
  "orchestrator_timeout_s": $timeout_s,
  "orchestrator_exit_code": $rc,
  "core": $CORE,
  "driver_args": "$DRIVER_ARGS --repeats $REPEATS",
  "perf_mode": "$perf",
  "image_profile": "$CH_PROFILE",
  "sif": "$SIF",
  "image_ref": "$IMAGE_REF",
  "image_digest": "$IMAGE_DIGEST",
  "apptainer_version": "$(apptainer --version)",
  "kernel": "$(uname -r)",
  "perf_event_paranoid": "$(cat /proc/sys/kernel/perf_event_paranoid)",
  "hpc_scripts_commit": "$hpc_commit"
}
EOF
        status=$(grep -o '"status": "[a-z]*"' "$DST/status.json" 2>/dev/null | cut -d'"' -f4)
        wall=$(grep -m1 -o '"runtime_s": [0-9.e+-]*' "$DST/result.json" 2>/dev/null | awk '{print $2}')
        echo "===== [$order] $CASE finished: status=${status:-none}  mean ${wall:-?} s per call  [$end]"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$order" "$CONFIG" "$N" "${status:-none}" "${wall:-}" "$start" "$end" \
            >> "$OUT_WORK/summary.tsv"
    done
done

# This job's table (the image's make_table.jl; reads only the JSON copied to $WORK).
echo; echo "===== table"
cfg_list=$(echo $CONFIGS | tr ' ' ',')
n_list=$(echo $NSTEPS | tr ' ' ',')
apptainer_exec julia /opt/channelanigans/bench/make_table.jl --out "/work_results/$SWEEP" \
    --ci --flops --delta --configs "$cfg_list" --nsteps "$n_list" \
    || echo "make_table failed (the case results are still in $OUT_WORK)"

echo; echo "===== summary ($OUT_WORK/summary.tsv)"
column -t -s $'\t' "$OUT_WORK/summary.tsv" 2>/dev/null || cat "$OUT_WORK/summary.tsv"
exit $rc_all
