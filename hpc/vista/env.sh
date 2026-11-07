# Shared settings for the Vista (TACC) runs of the CPU AD cost table.
# Sourced by gate.slurm, case.slurm, submit.sh and report.sh; edit here only.
# Image-specific settings live in the profile block below.

# ---- Allocation / partition ------------------------------------------------
ACCOUNT=Transfer-of-ECCO-glo
PARTITION=gg                 # Grace-Grace CPU nodes (NVIDIA Grace, Neoverse V2)
MAX_WALLTIME_H=48            # gg limit; cases predicted to need more are capped

# ---- Email notifications (one email per job; empty MAIL_USER = none) ---------
# A case that ends as t (timeout) or m (out of memory) is still a successful SLURM
# job, so use END and check `report.sh status` for the table outcome.
MAIL_USER=josek97@utexas.edu
MAIL_TYPE=END,FAIL

# ---- Image profile -------------------------------------------------------------
# Which container image a run uses, and where its results go. Selected with
# `submit.sh --profile NAME` (default: current), which exports CH_PROFILE to the jobs.
# The SIF was pulled from the registry image named in IMAGE_REF. The tag is mutable,
# the digest is not: the digest identifies the image that produced the results.
#   SIF            the pulled image on Vista
#   IMAGE_COMMIT   repo commit baked into the image (its BUILD_INFO.txt)
#   SWEEP          results subdirectory (under $SCRATCH_ROOT and $WORK_ROOT)
#   TABLE_CONFIGS  configs for `--configs table`; "table" = the rows in configs.toml
#   JOB_PREFIX     SLURM job-name prefix, so two profiles never see each other's jobs
#   WALLTIMES_FILE where `report.sh status` writes its walltime predictions
#   GATE_WALLTIME / GATE_E2E_TIMEOUT_S   gate job limit / its end-to-end test timeout
CH_PROFILE=${CH_PROFILE:-current}
case $CH_PROFILE in
    current)   # current package versions (Julia 1.12.7, Reactant 0.2.304, ...)
        SIF=/work/08955/jlk3675/vista/channelanigans-2026-11-cpu.sif
        IMAGE_REF=channelanigans:2026-11-cpu
        IMAGE_DIGEST=sha256:b9c94c1fbac86e8b7cdc709540fd97850f2069a61bbf480f5709b31b43676df0
        IMAGE_COMMIT=8fcc665421191cb84c1d203829c0e17a89865834
        SWEEP=table
        TABLE_CONFIGS=table
        JOB_PREFIX=ch
        WALLTIMES_FILE=walltimes.tsv
        GATE_WALLTIME=01:30:00
        GATE_E2E_TIMEOUT_S=3600
        ;;
    aug2026)   # July/August 2026 environment of the original runs (envs/aug2026)
        # Second build: the first (2026-11-aug2026-cpu, commit 348789f) failed to save
        # gradient files on Vista (JLD2 mmap under Julia 1.11 on 64 KiB pages); see
        # bench/common.jl. SIF sha256 11a66cae...0029165.
        SIF=/work/08955/jlk3675/vista/channelanigans-2026-11-aug2026-cpu-2.sif
        IMAGE_REF=ghcr.io/jlk9/channelanigans-artifact:2026-11-aug2026-cpu-2
        IMAGE_DIGEST=sha256:d9a137735986b897d88c22337f3b06b16868b3a19dcbd6c2089bad1c508edcb3
        IMAGE_COMMIT=d7c03aaa476c88c715b2af701c0627889490225a
        SWEEP=table_aug2026
        # The original table's rows: no enzyme_revolve_log2.
        TABLE_CONFIGS=reactant_full,reactant_after_enzyme,reactant_mincut_off,reactant_periodic,reactant_periodic_mincut_off,reactant_binomial_10,reactant_binomial_log2,enzyme_naive
        JOB_PREFIX=ch-aug
        WALLTIMES_FILE=walltimes_aug2026.tsv
        # Enzyme 0.13.198 spends ~36 min compiling the first gradient (seen in August).
        GATE_WALLTIME=02:30:00
        GATE_E2E_TIMEOUT_S=5400
        ;;
    *)
        echo "unknown image profile '$CH_PROFILE' (known: current, aug2026)" >&2
        return 2 2>/dev/null || exit 2
        ;;
esac

# ---- Results -----------------------------------------------------------------
# Jobs write to $SCRATCH (fast, purged after inactivity) and copy each finished
# case to $WORK (persistent). Everything downstream reads the $WORK copy.
SCRATCH_ROOT=$SCRATCH/channelanigans_results
WORK_ROOT=$WORK/channelanigans_results
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
