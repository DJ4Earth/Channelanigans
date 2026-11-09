#!/bin/bash
# Inspect and post-process the sweep in $WORK. Needs a compute node (idev), because
# Vista does not allow Apptainer on login nodes; nothing here is heavy.
#
#   hpc/vista/report.sh status   # per-case status, timings, memory; predicts walltimes
#                                # for unfinished cases into hpc/vista/$WALLTIMES_FILE
#   hpc/vista/report.sh final    # table (LaTeX/CSV/Markdown), gradient cross-check,
#                                # raw-data export + archive, all into $WORK_ROOT/$SWEEP
#
# Batched profiles (BATCH_REPEATS in env.sh): `status` reports every batch ${SWEEP}_bK
# (predictions into ${WALLTIMES_FILE%.tsv}_bK.tsv); `final` first pools all finished
# batches into $SWEEP (bench/merge_batches.jl), then runs the usual steps on the pool
# and archives the pool plus every batch. It can be rerun at any time: cases whose
# later batches have not finished yet are pooled from the batches that have.
#
# Everything runs inside the same image as the benchmarks. The image profile comes from
# CH_PROFILE (exported by `submit.sh --profile NAME --report MODE`; default current).

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
# Vista only allows Apptainer on compute nodes (on login nodes it is a stub).
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "report.sh runs the container, so it needs a compute node. Either submit it as a job:" >&2
    echo "    hpc/vista/submit.sh --profile $CH_PROFILE --report ${1:-status}" >&2
    echo "or run it inside an interactive session:  idev -p $PARTITION -N 1 -n 1 -t 00:30:00 -A $ACCOUNT" >&2
    exit 2
fi
load_apptainer
mkdir -p "$SCRATCH_ROOT" "$WORK_ROOT/$SWEEP"
RES=/work_results/$SWEEP
# The profile's table configs ("table" = the rows in configs.toml).
CFG_ARGS=(); [[ $TABLE_CONFIGS != table ]] && CFG_ARGS=(--configs "$TABLE_CONFIGS")
echo "profile: $CH_PROFILE   image: $IMAGE_REF   results: $WORK_ROOT/$SWEEP"

case ${1:-status} in
    status)
        plan() {   # plan <results dir under /work_results> <walltimes file> [ncalls]
            apptainer exec --cleanenv --bind "$WORK_ROOT:/work_results" --bind "$HPC_DIR:/hpc" "$SIF" \
                julia /hpc/plan.jl "/work_results/$1" /opt/channelanigans/bench/configs.toml "/hpc/$2" \
                      "$MAX_WALLTIME_H" "$TIMEOUT_MARGIN_S" "$TABLE_CONFIGS" ${3:+"$3"}
        }
        if [[ -n ${BATCH_REPEATS:-} ]]; then
            for d in "$WORK_ROOT/${SWEEP_BASE}"_b*; do
                [[ -d $d ]] || continue
                k=${d##*_b}
                echo; echo "===== batch $k ($d)"
                plan "$(basename "$d")" "${WALLTIMES_FILE%.tsv}_b$k.tsv" $((BATCH_REPEATS + 1))
            done
        else
            plan "$SWEEP" "$WALLTIMES_FILE"
        fi
        echo "queued/running:"; squeue -u "$USER" -o '%.10i %.44j %.8T %.10M %.10l %R' || true
        ;;
    final)
        if [[ -n ${BATCH_REPEATS:-} ]]; then
            echo "== pool batches";  apptainer_exec julia /opt/channelanigans/bench/merge_batches.jl \
                                         --root /work_results --sweep "$SWEEP_BASE"
        fi
        echo "== table";            apptainer_exec julia /opt/channelanigans/bench/make_table.jl --out "$RES" --ci --flops --delta "${CFG_ARGS[@]}"
        echo "== gradient check";   apptainer_exec julia /opt/channelanigans/bench/check_gradients.jl --out "$RES" \
                                        || echo "gradient check reported failures (see above)"
        if [[ -n ${BATCH_REPEATS:-} ]]; then
            echo "== raw-data export"; apptainer_exec julia /opt/channelanigans/bench/export_raw.jl --out "$RES"
            # archive: the pool and every batch (logs, JSON, perf.csv; no gradient files)
            ( cd "$WORK_ROOT" && tar -czf "$SWEEP/raw_data.tar.gz" --exclude='*.jld2' \
                  --exclude=raw_data.tar.gz "$SWEEP" "$SWEEP"_b* )
            echo "  archive (pool + batches; gradient .jld2 excluded) → $WORK_ROOT/$SWEEP/raw_data.tar.gz"
        else
            echo "== raw-data export"; apptainer_exec julia /opt/channelanigans/bench/export_raw.jl --out "$RES" --archive
        fi
        echo "outputs in $WORK_ROOT/$SWEEP (table.tex, table.csv, export/)"
        ;;
    *) sed -n '2,8p' "$0"; exit 2 ;;
esac
