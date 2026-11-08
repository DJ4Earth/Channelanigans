#!/bin/bash
# Submit the Vista jobs. Run from a login node, from anywhere in the repo checkout.
#
#   hpc/vista/submit.sh --gate                          # gate job (run first)
#   hpc/vista/submit.sh --nsteps 4,8 --walltime 04:00:00   # wave 1: all table configs
#   hpc/vista/submit.sh --walltimes hpc/vista/walltimes.tsv  # later waves, per-case limits
#   hpc/vista/submit.sh --report status                 # run report.sh as a short job
#   hpc/vista/submit.sh --profile aug2026 ...           # same, for the August-environment image
#
# Options:
#   --profile NAME     image profile from env.sh: current (default) or aug2026. Sets the
#                      image, the results folder, the table's configs and job names
#   --configs LIST     comma-separated config names, or "table" (default: the profile's
#                      table configs)
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
# The image profile must be known before env.sh is read. Only --profile selects it (an
# inherited CH_PROFILE is ignored); it is exported so the jobs (--export=ALL) use it too.
CH_PROFILE=current
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    [[ ${args[i]} == --profile ]] && CH_PROFILE=${args[i+1]:-}
done
export CH_PROFILE
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
        --profile) shift ;;   # read before env.sh was sourced (above)
        -h|--help) sed -n '2,31p' "$0"; exit 0 ;;
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

# For the case loop: a refused sbatch (e.g. a per-user job limit) must not silently
# drop the rest of the sweep, so record it and carry on.
failed=()
try_submit() {   # same arguments as submit; the job name is $1
    local out
    if out=$(submit "$@" 2>&1); then
        echo "$out"; return 0
    fi
    echo "$out" >&2
    echo "NOT SUBMITTED: $1" >&2
    failed+=("$1")
    return 0
}

if [[ -n $report ]]; then
    [[ $report == status || $report == final ]] || { echo "--report takes status or final" >&2; exit 2; }
    # --wrap runs report.sh from the repo (not a spooled copy), so it finds env.sh.
    submit "$JOB_PREFIX-report-$report" "$([[ $report == final ]] && echo 01:00:00 || echo 00:20:00)" \
        "--wrap=bash $HPC_DIR/report.sh $report"
    echo "report log: $LOG_DIR/$JOB_PREFIX-report-$report.<jobid>.out"
    exit 0
fi

if [[ $gate -eq 1 ]]; then
    submit "$JOB_PREFIX-gate" "$GATE_WALLTIME" gate.slurm
    echo "gate log: $LOG_DIR/$JOB_PREFIX-gate.<jobid>.out"
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

[[ $configs == table && $TABLE_CONFIGS != table ]] && configs=$TABLE_CONFIGS
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
echo "profile $CH_PROFILE: ${#cases[@]} case(s) selected: ${#names[@]} config(s) × N = ${ns[*]}"

queued=$(squeue -h -u "$USER" -o %j 2>/dev/null || true)
max_s=$((MAX_WALLTIME_H * 3600))
nsub=0
for c in "${cases[@]}"; do
    cfg=${c%%:*}; N=${c##*:}
    name=$JOB_PREFIX-$cfg-N$N
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
    nfail=${#failed[@]}
    try_submit "$name" "$wt" case.slurm "CONFIG=$cfg,NSTEPS=$N,TIMEOUT_S=$timeout_s,FORCE=$force,PERF=$perf"
    (( ${#failed[@]} == nfail )) && nsub=$((nsub + 1))
done
echo "submitted $nsub job(s); logs in $LOG_DIR"
if (( ${#failed[@]} > 0 )); then
    echo >&2
    echo "${#failed[@]} case(s) were NOT submitted (sbatch refused them; error above):" >&2
    printf '  %s\n' "${failed[@]}" >&2
    echo "Rerun the same command later: it skips cases that are done or queued." >&2
    exit 1
fi
