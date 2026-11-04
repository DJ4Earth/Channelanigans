# Shared settings for the Vista (TACC) runs of the CPU AD cost table.
# Sourced by gate.slurm, case.slurm, submit.sh and report.sh; edit here only.

# ---- Allocation / partition ------------------------------------------------
ACCOUNT=Transfer-of-ECCO-glo
PARTITION=gg                 # Grace-Grace CPU nodes (NVIDIA Grace, Neoverse V2)
MAX_WALLTIME_H=48            # gg limit; cases predicted to need more are capped

# ---- Email notifications (one email per job; empty MAIL_USER = none) ---------
# A case that ends as t (timeout) or m (out of memory) is still a successful SLURM
# job, so use END and check `report.sh status` for the table outcome.
MAIL_USER=josek97@utexas.edu
MAIL_TYPE=END,FAIL

# ---- Container image ---------------------------------------------------------
# The SIF was pulled from the registry image below. The tag is mutable, the digest
# is not: the digest is what identifies the image that produced the results.
SIF=/work/08955/jlk3675/vista/channelanigans-2026-11-cpu.sif
IMAGE_REF=channelanigans:2026-11-cpu
IMAGE_DIGEST=sha256:b9c94c1fbac86e8b7cdc709540fd97850f2069a61bbf480f5709b31b43676df0

# ---- Results -----------------------------------------------------------------
# Jobs write to $SCRATCH (fast, purged after inactivity) and copy each finished
# case to $WORK (persistent). Everything downstream reads the $WORK copy.
SCRATCH_ROOT=$SCRATCH/channelanigans_results
WORK_ROOT=$WORK/channelanigans_results
SWEEP=table                  # subdirectory for the table sweep
LOG_DIR=$WORK_ROOT/slurm_logs

# ---- Benchmark settings (same as the Docker proofing runs) -------------------
CORE=0                       # numactl -C core for the benchmark process
DRIVER_ARGS="--save-gradients"   # per-driver extras; protocol defaults: 1 warmup + 12 measured
TIMEOUT_MARGIN_S=1800        # orchestrator timeout = walltime - margin, so a slow
                             # case is classified "t" and copied back before SLURM kills the job

# Directory holding these scripts (and the repo checkout they belong to)
HPC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$HPC_DIR/../.." && pwd)

# Apptainer: --cleanenv keeps the host environment (incl. your JULIA_DEPOT_PATH)
# out of the container. A per-job writable depot is stacked in front of the frozen,
# precompiled one. SLURM variables are passed back in so result.json records them.
apptainer_exec() {
    local depot=${TMPDIR:-/tmp}/channelanigans_depot_${SLURM_JOB_ID:-$$}
    mkdir -p "$depot"
    apptainer exec --cleanenv \
        --bind "$SCRATCH_ROOT:/results" \
        --bind "$WORK_ROOT:/work_results" \
        --env "JULIA_DEPOT_PATH=$depot:/opt/julia_depot" \
        --env "SLURM_JOB_ID=${SLURM_JOB_ID:-},SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}" \
        --env "CHANNELANIGANS_IMAGE_DIGEST=$IMAGE_DIGEST" \
        "$SIF" "$@"
}

# Lmod's `module` is not safe under `set -u`/`set -e` (it references unset variables,
# which silently kills the calling script), so relax both just for the module load.
load_apptainer() {
    if ! command -v apptainer >/dev/null 2>&1; then
        local saved=$-
        set +eu
        module load tacc-apptainer
        [[ $saved == *u* ]] && set -u
        [[ $saved == *e* ]] && set -e
    fi
    command -v apptainer >/dev/null 2>&1 || { echo "apptainer not found (tried: module load tacc-apptainer)" >&2; exit 2; }
}

# HH:MM:SS → seconds
hms_to_s() { local IFS=:; set -- $1; echo $(( 10#$1 * 3600 + 10#$2 * 60 + 10#${3:-0} )); }
