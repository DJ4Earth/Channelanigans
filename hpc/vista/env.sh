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
#   RUN_JULIA_FLAGS / RUN_XLA_FLAGS     optional: Julia and XLA flags for the drivers;
#                  unset = the orchestrator's defaults (-O0, multi-threaded Eigen off)
#   XLA_DUMP       1 = also dump XLA's HLO (--xla_dump_to), as the August job scripts did,
#                  and keep its buffer-assignment "Total bytes used" lines in each case
#                  (xla_buffer_assignment.txt; extra information, not the memory metric)
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
        # Enzyme 0.13.198 spends ~30-36 min compiling the first gradient.
        GATE_WALLTIME=02:30:00
        GATE_E2E_TIMEOUT_S=5400
        # Runtime flags of the original August job scripts, applied uniformly to every
        # row (as in the binomial-checkpointing job script; the scripts differed slightly
        # between rows): Julia's default -O2, XLA's multi-threaded Eigen left on (its
        # default), multi-output fusion on, and an HLO dump. The dump's "Total bytes used"
        # is kept as extra information; the table's memory metric is peak RSS, as in August.
        RUN_JULIA_FLAGS=""
        RUN_XLA_FLAGS="--xla_backend_extra_options=xla_cpu_use_multi_output_fusion"
        XLA_DUMP=1
        ;;
    jll443)    # current versions, but Reactant from main with Reactant_jll 0.0.443 (envs/jll443)
        # Same runtime settings as `current` (orchestrator defaults: -O0, multi-threaded
        # Eigen off). Run in batches of 4 measured calls (see "Batches" below).
        # TODO after the image is built and pulled: SIF sha256, IMAGE_DIGEST, IMAGE_COMMIT.
        SIF=/work/08955/jlk3675/vista/channelanigans-2026-11-jll443-cpu.sif
        IMAGE_REF=ghcr.io/jlk9/channelanigans-artifact:2026-11-jll443-cpu
        IMAGE_DIGEST=sha256:TODO
        IMAGE_COMMIT=TODO
        SWEEP=table_jll443
        TABLE_CONFIGS=table
        JOB_PREFIX=ch-j443
        WALLTIMES_FILE=walltimes_jll443.tsv
        GATE_WALLTIME=01:30:00
        GATE_E2E_TIMEOUT_S=3600
        BATCH_REPEATS=4
        # enzyme_naive N=256 runs out of memory; under perf that is recorded as "e", so
        # run it without perf (as its 2026-11 table entry was).
        PERF_OFF_CASES="enzyme_naive:256"
        ;;
    *)
        echo "unknown image profile '$CH_PROFILE' (known: current, aug2026, jll443)" >&2
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
MAX_JOBS=${MAX_JOBS:-40}     # submit.sh keeps at most this many of your jobs in the queue
                             # (all of them, incl. idev); TACC's per-user limit is ~40-50

# ---- Batches -------------------------------------------------------------------
# A profile that sets BATCH_REPEATS runs every case as several independent batches:
# separate jobs (usually on different nodes), each a fresh process with its own
# compilation, 1 warmup call and BATCH_REPEATS measured calls. Batch K writes to
# ${SWEEP}_bK with job names $JOB_PREFIX-bK-...; `report.sh final` pools the measured
# calls of all finished batches into $SWEEP (bench/merge_batches.jl).
# select_batch K switches SWEEP and JOB_PREFIX to batch K (case.slurm gets K as CH_BATCH).
SWEEP_BASE=$SWEEP
JOB_PREFIX_BASE=$JOB_PREFIX
[[ -n ${BATCH_REPEATS:-} ]] && DRIVER_ARGS="--save-gradients --repeats $BATCH_REPEATS"
select_batch() {
    [[ $1 =~ ^[1-9][0-9]*$ ]] || { echo "bad batch number '$1'" >&2; return 2; }
    SWEEP=${SWEEP_BASE}_b$1
    JOB_PREFIX=$JOB_PREFIX_BASE-b$1
}

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

# Orchestrator arguments for the profile's Julia/XLA flags, in the array ORCH_FLAGS
# (empty for profiles that use the orchestrator's defaults). $1: subdirectory of
# $SCRATCH_ROOT/_hlo for the XLA HLO dump, used when XLA_DUMP=1.
orchestrator_flag_args() {
    ORCH_FLAGS=()
    [[ -n ${RUN_JULIA_FLAGS+x} ]] && ORCH_FLAGS+=(--julia-flags "$RUN_JULIA_FLAGS")
    if [[ -n ${RUN_XLA_FLAGS+x} ]]; then
        local xla=$RUN_XLA_FLAGS
        [[ ${XLA_DUMP:-0} == 1 ]] && xla="$xla --xla_dump_to=/results/_hlo/$1"
        ORCH_FLAGS+=(--xla-flags "$xla")
    fi
    return 0
}

# Lines of XLA's buffer assignment with "Total bytes used", one per compiled module,
# prefixed with the dump file name. $1: HLO dump directory on the host. Depending on the
# XLA version the line is in *-buffer-assignment.txt or *-buffer-assignment-values.txt
# (the August job scripts grepped only the former and found nothing), so search both.
xla_total_bytes() {
    grep -H "Total bytes used" "$1"/*buffer-assignment*.txt 2>/dev/null | sed "s|^$1/||"
    return 0
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
