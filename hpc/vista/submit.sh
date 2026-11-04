#!/bin/bash
# Submit the Vista jobs. Run from a login node, from anywhere in the repo checkout.
#
#   hpc/vista/submit.sh --gate                          # gate job (run first)
#   hpc/vista/submit.sh --nsteps 4,8 --walltime 04:00:00   # wave 1: all table configs
#   hpc/vista/submit.sh --walltimes hpc/vista/walltimes.tsv  # later waves, per-case limits
#   hpc/vista/submit.sh --report status                 # run report.sh as a short job
#
# Options:
#   --configs LIST     comma-separated config names, or "table" (default: table rows)
#   --nsteps LIST      comma-separated N (default: the table's N list)
#   --walltime T       HH:MM:SS for every selected case
#   --walltimes FILE   per-case limits, lines "config<TAB>N<TAB>HH:MM:SS" (written by
#                      plan.jl); cases not listed there are skipped
#   --force            rerun cases that already have an ok result in $WORK
#   --perf on|off      FLOP counting (default on). Use off only to diagnose a case that
#                      perf hides a kill for (see case.slurm); off-cases have no FLOPs
#   --report MODE      run `report.sh MODE` (status | final) as a short batch job
#                      instead of in an idev session; output in the log directory
#   --dry-run          print the sbatch commands only
#
# One job per (config, N), each on a whole gg node. Cases that are already ok in
# $WORK, or already queued/running under the same job name, are skipped. The
# orchestrator timeout is the walltime minus TIMEOUT_MARGIN_S (env.sh), capped at
# MAX_WALLTIME_H.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

configs=table; nsteps=""; walltime=""; walltimes=""; force=0; dry=0; gate=0; report=""; perf=on
while [[ $# -gt 0 ]]; do
    case $1 in
        --gate) gate=1 ;;
        --report) report=$2; shift ;;
        --configs) configs=$2; shift ;;
        --nsteps) nsteps=$2; shift ;;
        --walltime) walltime=$2; shift ;;
        --walltimes) walltimes=$2; shift ;;
        --force) force=1 ;;
        --perf) perf=$2; shift
                [[ $perf == on || $perf == off ]] || { echo "--perf takes on or off" >&2; exit 2; } ;;
        --dry-run) dry=1 ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

# No apptainer here: on Vista login nodes it is a stub that only prints a banner.
mkdir -p "$SCRATCH_ROOT" "$WORK_ROOT" "$LOG_DIR"
[[ -f "$SIF" ]] || { echo "SIF not found: $SIF" >&2; exit 2; }

submit() {   # submit <job name> <walltime> <script | --wrap CMD> [extra --export vars]
    local name=$1 wt=$2 vars=${4:-}
    local target=("$HPC_DIR/$3")
    [[ $3 == --wrap=* ]] && target=("$3")
    # -N/-n repeat the scripts' #SBATCH lines; a --wrap job has none, and TACC
    # rejects any submission without an explicit node count.
    local cmd=(sbatch -A "$ACCOUNT" -p "$PARTITION" -N 1 -n 1 -J "$name" -t "$wt"
               -o "$LOG_DIR/%x.%j.out"
               ${MAIL_USER:+--mail-user="$MAIL_USER" --mail-type="${MAIL_TYPE:-END,FAIL}"}
               --export="ALL,HPC_DIR=$HPC_DIR${vars:+,$vars}" "${target[@]}")
    if [[ $dry -eq 1 ]]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
}

if [[ -n $report ]]; then
    [[ $report == status || $report == final ]] || { echo "--report takes status or final" >&2; exit 2; }
    # --wrap runs report.sh from the repo (not a spooled copy), so it finds env.sh.
    submit "ch-report-$report" "$([[ $report == final ]] && echo 01:00:00 || echo 00:20:00)" \
        "--wrap=bash $HPC_DIR/report.sh $report"
    echo "report log: $LOG_DIR/ch-report-$report.<jobid>.out"
    exit 0
fi

if [[ $gate -eq 1 ]]; then
    submit ch-gate 01:30:00 gate.slurm
    echo "gate log: $LOG_DIR/ch-gate.<jobid>.out"
    exit 0
fi

[[ -n "$walltime" || -n "$walltimes" ]] || { echo "give --walltime or --walltimes" >&2; exit 2; }

# Config names and the N list come from the repo's bench/configs.toml, which must be
# identical to the one baked into the image (the code that actually runs).
TOML=$REPO_DIR/bench/configs.toml
if ! git -C "$REPO_DIR" diff --quiet "$IMAGE_COMMIT" -- bench/configs.toml 2>/dev/null; then
    echo "bench/configs.toml differs from the image's (commit ${IMAGE_COMMIT:0:7}), or that" >&2
    echo "commit is not in this checkout; the case list would not match the code that runs." >&2
    exit 2
fi
# [table] section with comments stripped, as one line; then pull out an array value.
table=$(awk '/^\[/ { t = ($0 ~ /^\[table\]/); next } t { sub(/#.*/, ""); printf "%s ", $0 }' "$TOML")
toml_array() { sed -n "s/.*$1 *= *\[\([^]]*\)\].*/\1/p" <<< "$table" | tr -d '" ' | tr ',' ' '; }
known=" $(sed -n 's/^\[configs\.\([A-Za-z0-9_]*\)\].*/\1/p' "$TOML" | tr '\n' ' ') "

if [[ $configs == table ]]; then read -r -a names <<< "$(toml_array rows)"
else IFS=, read -r -a names <<< "$configs"; fi
if [[ -z $nsteps ]]; then read -r -a ns <<< "$(toml_array nsteps)"
else IFS=, read -r -a ns <<< "$nsteps"; fi
for n in "${names[@]}"; do
    [[ $known == *" $n "* ]] || { echo "unknown config '$n'; known:$known" >&2; exit 2; }
done
for N in "${ns[@]}"; do
    [[ $N =~ ^[0-9]+$ ]] || { echo "bad N '$N'" >&2; exit 2; }
done
cases=()
for n in "${names[@]}"; do for N in "${ns[@]}"; do cases+=("$n:$N"); done; done
(( ${#cases[@]} > 0 )) || { echo "no cases selected (configs '$configs', nsteps '$nsteps')" >&2; exit 2; }
echo "${#cases[@]} case(s) selected: ${#names[@]} config(s) × N = ${ns[*]}"

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
    submit "$name" "$wt" case.slurm "CONFIG=$cfg,NSTEPS=$N,TIMEOUT_S=$timeout_s,FORCE=$force,PERF=$perf"
    nsub=$((nsub + 1))
done
echo "submitted $nsub job(s); logs in $LOG_DIR"
