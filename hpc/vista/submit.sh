#!/bin/bash
# Submit the Vista jobs. Run from a login node, from anywhere in the repo checkout.
#
#   hpc/vista/submit.sh --gate                          # gate job (run first)
#   hpc/vista/submit.sh --nsteps 4,8 --walltime 04:00:00   # wave 1: all table configs
#   hpc/vista/submit.sh --walltimes hpc/vista/walltimes.tsv  # later waves, per-case limits
#
# Options:
#   --configs LIST     comma-separated config names, or "table" (default: table rows)
#   --nsteps LIST      comma-separated N (default: the table's N list)
#   --walltime T       HH:MM:SS for every selected case
#   --walltimes FILE   per-case limits, lines "config<TAB>N<TAB>HH:MM:SS" (written by
#                      plan.jl); cases not listed there are skipped
#   --force            rerun cases that already have an ok result in $WORK
#   --dry-run          print the sbatch commands only
#
# One job per (config, N), each on a whole gg node. Cases that are already ok in
# $WORK, or already queued/running under the same job name, are skipped. The
# orchestrator timeout is the walltime minus TIMEOUT_MARGIN_S (env.sh), capped at
# MAX_WALLTIME_H.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

configs=table; nsteps=""; walltime=""; walltimes=""; force=0; dry=0; gate=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --gate) gate=1 ;;
        --configs) configs=$2; shift ;;
        --nsteps) nsteps=$2; shift ;;
        --walltime) walltime=$2; shift ;;
        --walltimes) walltimes=$2; shift ;;
        --force) force=1 ;;
        --dry-run) dry=1 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

load_apptainer
mkdir -p "$SCRATCH_ROOT" "$WORK_ROOT" "$LOG_DIR"
[[ -f "$SIF" ]] || { echo "SIF not found: $SIF" >&2; exit 2; }

submit() {   # submit <job name> <walltime> <script> [extra --export vars]
    local name=$1 wt=$2 script=$3 vars=${4:-}
    local cmd=(sbatch -A "$ACCOUNT" -p "$PARTITION" -J "$name" -t "$wt"
               -o "$LOG_DIR/%x.%j.out"
               ${MAIL_USER:+--mail-user="$MAIL_USER" --mail-type="${MAIL_TYPE:-END,FAIL}"}
               --export="ALL,HPC_DIR=$HPC_DIR${vars:+,$vars}" "$HPC_DIR/$script")
    if [[ $dry -eq 1 ]]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
}

if [[ $gate -eq 1 ]]; then
    submit ch-gate 01:30:00 gate.slurm
    echo "gate log: $LOG_DIR/ch-gate.<jobid>.out"
    exit 0
fi

[[ -n "$walltime" || -n "$walltimes" ]] || { echo "give --walltime or --walltimes" >&2; exit 2; }

# Config names and N list come from the configs.toml inside the image (the code that runs).
case_list=$(apptainer_exec julia -e '
    using TOML
    all = TOML.parsefile("/opt/channelanigans/bench/configs.toml")
    names = ARGS[1] == "table" ? all["table"]["rows"] : split(ARGS[1], ",")
    ns = isempty(ARGS[2]) ? all["table"]["nsteps"] : parse.(Int, split(ARGS[2], ","))
    for n in names; haskey(all["configs"], n) || error("unknown config $n"); end
    for n in names, N in ns; println("CASE $n $N"); end' "$configs" "$nsteps")
# Keep only the CASE lines: anything else the container or environment prints to
# stdout must not be mistaken for (or hide) the case list.
mapfile -t cases < <(awk '$1 == "CASE" && NF == 3 { print $2 ":" $3 }' <<< "$case_list")
if [[ ${#cases[@]} -eq 0 ]]; then
    echo "no cases found; container output was:" >&2; printf '%s\n' "$case_list" >&2; exit 2
fi
echo "${#cases[@]} case(s) selected"

queued=$(squeue -h -u "$USER" -o %j 2>/dev/null || true)
max_s=$((MAX_WALLTIME_H * 3600))
nsub=0
for c in "${cases[@]}"; do
    cfg=${c%%:*}; N=${c##*:}
    name=ch-$cfg-N$N
    dst=$WORK_ROOT/$SWEEP/$cfg/$(printf 'N%04d' "$N")
    if [[ $force -eq 0 && -f $dst/status.json ]] && grep -q '"status": "ok"' "$dst/status.json"; then
        echo "skip $name: already ok"; continue
    fi
    if grep -qx "$name" <<< "$queued"; then echo "skip $name: already queued"; continue; fi
    wt=$walltime
    if [[ -n "$walltimes" ]]; then
        wt=$(awk -F'\t' -v c="$cfg" -v n="$N" '$1 == c && $2 == n { print $3 }' "$walltimes")
        [[ -n "$wt" ]] || { echo "skip $name: not in $walltimes"; continue; }
    fi
    wt_s=$(hms_to_s "$wt")
    if (( wt_s > max_s )); then wt_s=$max_s; wt=$(printf '%02d:00:00' "$MAX_WALLTIME_H"); fi
    timeout_s=$(( wt_s - TIMEOUT_MARGIN_S ))
    (( timeout_s > 600 )) || { echo "walltime $wt too short for $name" >&2; exit 2; }
    submit "$name" "$wt" case.slurm "CONFIG=$cfg,NSTEPS=$N,TIMEOUT_S=$timeout_s,FORCE=$force"
    nsub=$((nsub + 1))
done
echo "submitted $nsub job(s); logs in $LOG_DIR"
