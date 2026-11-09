#!/bin/bash
# Keep the queue topped up: run submit.sh every INTERVAL seconds until every selected
# case has been submitted (or is done, or is skipped), then exit. Run it on a login
# node inside tmux, so it survives logging out:
#
#   tmux new -s fill        # reattach later with `tmux attach -t fill`, on the SAME login node
#   hpc/vista/fill.sh --profile jll443 --batches 1,2,3 \
#       --walltimes hpc/vista/walltimes_jll443_seed.tsv [--max-jobs 40] [--interval 600]
#
# Every option except --interval is passed to submit.sh, which also gets --new-only:
# a case that already has a result in $WORK is never resubmitted by this loop, whatever
# its outcome (rerun those by hand with submit.sh). Each round's full submit.sh output
# is appended to $LOG_DIR/fill.<profile>.log; the terminal shows one line per round.
# Stops on its own when nothing is left (PENDING=0), or when submit.sh reports a
# configuration error (exit code 2). Stop it any time with Ctrl-C; nothing is lost.

set -uo pipefail
HPC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
interval=600; profile=current; pass=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --interval) interval=$2; shift ;;
        --profile)  profile=$2; pass+=("$1" "$2"); shift ;;
        -h|--help)  sed -n '2,15p' "$0"; exit 0 ;;
        *)          pass+=("$1") ;;
    esac
    shift
done
[[ $interval =~ ^[0-9]+$ ]] || { echo "--interval takes seconds" >&2; exit 2; }

CH_PROFILE=$profile
source "$HPC_DIR/env.sh"
mkdir -p "$LOG_DIR"
log=$LOG_DIR/fill.$profile.log
echo "fill: profile $profile, every ${interval}s; full output in $log"

while true; do
    out=$("$HPC_DIR/submit.sh" "${pass[@]}" --new-only 2>&1)
    rc=$?
    { echo "===== $(date '+%F %T')  (exit $rc)"; echo "$out"; } >> "$log"
    pending=$(sed -n 's/^PENDING=\([0-9]*\)$/\1/p' <<< "$out" | tail -1)
    nsub=$(sed -n 's/^submitted \([0-9]*\) job.*/\1/p' <<< "$out" | tail -1)
    room=$(sed -n 's/.*room for \([0-9]*\)$/\1/p' <<< "$out" | tail -1)
    echo "$(date '+%F %T')  submitted ${nsub:-?}, still to submit ${pending:-?}, queue room was ${room:-?}"
    if [[ $rc -ne 0 && $rc -ne 1 ]] || [[ -z $pending ]]; then
        echo "submit.sh failed (exit $rc); last output:"; echo "$out" | tail -20
        exit 2
    fi
    if [[ $pending -eq 0 ]]; then
        echo "everything is submitted, done, or skipped; fill.sh exits"
        exit 0
    fi
    sleep "$interval"
done
