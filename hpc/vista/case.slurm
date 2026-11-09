#!/bin/bash
# One table case = one (config, N) = one job on a whole gg node.
# Submitted by submit.sh with CONFIG, NSTEPS and TIMEOUT_S exported; not meant to be
# sbatch'ed by hand.
#
# Flow:
#   * skip if $WORK already holds an ok result for this case (FORCE=1 to rerun);
#   * PERF=off (submit.sh --perf off) runs without perf: no FLOPs, but a process
#     killed by the kernel (e.g. OOM killer) is then seen as SIGKILL and classified
#     "m". Under perf the kill is hidden: perf exits 0 and the case looks like "e".
#   * move any earlier non-ok attempt aside to $WORK_ROOT/_superseded/ (kept, not deleted);
#   * run the orchestrator (1 warmup + 12 measured calls, numactl-pinned, perf-counted;
#     a batched profile: one batch of 1 warmup + BATCH_REPEATS measured calls, results
#     in ${SWEEP}_b$CH_BATCH) on $SCRATCH with a timeout below the SLURM walltime;
#   * copy the case directory to $WORK and add job_info.json (job, node, image digest).
#
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1

set -uo pipefail
source "${HPC_DIR:-$SLURM_SUBMIT_DIR/hpc/vista}/env.sh"
load_apptainer
: "${CONFIG:?}" "${NSTEPS:?}" "${TIMEOUT_S:?}"
# Batched profile: this job is one batch (CH_BATCH, exported by submit.sh --batches).
if [[ -n ${BATCH_REPEATS:-} ]]; then
    : "${CH_BATCH:?profile $CH_PROFILE runs in batches; submit with submit.sh --batches}"
    select_batch "$CH_BATCH" || exit 2
fi

CASE=$CONFIG/$(printf 'N%04d' "$NSTEPS")
SCR=$SCRATCH_ROOT/$SWEEP/$CASE
DST=$WORK_ROOT/$SWEEP/$CASE
mkdir -p "$SCRATCH_ROOT/$SWEEP" "$WORK_ROOT/$SWEEP"

if [[ "${FORCE:-0}" != 1 && -f "$DST/status.json" ]] && grep -q '"status": "ok"' "$DST/status.json"; then
    echo "$CASE already ok in $DST; nothing to do (FORCE=1 to rerun)"; exit 0
fi
# Earlier attempt: keep the $WORK copy; the $SCRATCH copy is a duplicate of it, unless
# the earlier job died before copying back, in which case the scratch copy is kept.
stamp=$(date -u +%Y%m%dT%H%M%SZ)
SUP=$WORK_ROOT/_superseded/$SWEEP/$CONFIG
if [[ -d "$DST" ]]; then
    mkdir -p "$SUP"; mv "$DST" "$SUP/$(basename "$DST").$stamp"
    echo "moved earlier attempt $DST -> $SUP/$(basename "$DST").$stamp"
    [[ -d "$SCR" ]] && rm -rf "$SCR"
elif [[ -d "$SCR" ]]; then
    mkdir -p "$SUP"; mv "$SCR" "$SUP/$(basename "$SCR").scratch.$stamp"
    echo "moved uncopied scratch attempt $SCR -> $SUP/$(basename "$SCR").scratch.$stamp"
fi

# The profile's Julia/XLA flags (none for the default profile); a fresh HLO dump dir.
orchestrator_flag_args "$SWEEP/$CASE"
HLO=$SCRATCH_ROOT/_hlo/$SWEEP/$CASE
[[ ${XLA_DUMP:-0} == 1 ]] && { rm -rf "$HLO"; mkdir -p "$HLO"; }

start=$(date -u +%FT%TZ)
echo "[$start] $CASE on $(hostname), job $SLURM_JOB_ID, orchestrator timeout ${TIMEOUT_S}s"
apptainer_exec julia /opt/channelanigans/bench/run_cases.jl \
    --configs "$CONFIG" --nsteps "$NSTEPS" --out "/results/$SWEEP" \
    --timeout "$TIMEOUT_S" --perf "${PERF:-on}" --numactl on --core "$CORE" "${ORCH_FLAGS[@]}" \
    -- $DRIVER_ARGS
rc=$?
end=$(date -u +%FT%TZ)

# XLA's memory figure (buffer assignment), as the August job scripts collected it.
[[ ${XLA_DUMP:-0} == 1 && -d "$SCR" ]] && xla_total_bytes "$HLO" > "$SCR/xla_buffer_assignment.txt"

mkdir -p "$DST"
cp -a "$SCR/." "$DST/"
hpc_commit=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)
[[ $hpc_commit != unknown ]] && ! git -C "$REPO_DIR" diff --quiet HEAD -- hpc && hpc_commit="$hpc_commit-dirty"
cat > "$DST/job_info.json" <<EOF
{
  "slurm_job_id": "$SLURM_JOB_ID",
  "slurm_job_name": "${SLURM_JOB_NAME:-}",
  "partition": "${SLURM_JOB_PARTITION:-}",
  "account": "${SLURM_JOB_ACCOUNT:-$ACCOUNT}",
  "node": "$(hostname)",
  "start_utc": "$start",
  "end_utc": "$end",
  "walltime_limit": "$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null)",
  "orchestrator_timeout_s": $TIMEOUT_S,
  "orchestrator_exit_code": $rc,
  "core": $CORE,
  "driver_args": "$DRIVER_ARGS",
  "perf_mode": "${PERF:-on}",
  "image_profile": "${CH_PROFILE:-current}",
  "batch": $([[ -n ${BATCH_REPEATS:-} ]] && echo "$CH_BATCH" || echo null),
  "sweep": "$SWEEP",
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
echo "[$end] $CASE finished: status=${status:-none} (orchestrator exit $rc); copied to $DST"
exit $rc
