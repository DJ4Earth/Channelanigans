#!/bin/bash
# Inspect and post-process the sweep in $WORK (login node is fine; nothing heavy).
#
#   hpc/vista/report.sh status   # per-case status, timings, memory; predicts walltimes
#                                # for unfinished cases into hpc/vista/walltimes.tsv
#   hpc/vista/report.sh final    # table (LaTeX/CSV/Markdown), gradient cross-check,
#                                # raw-data export + archive, all into $WORK_ROOT/$SWEEP
#
# Everything runs inside the same image as the benchmarks.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
load_apptainer
mkdir -p "$SCRATCH_ROOT" "$WORK_ROOT/$SWEEP"
RES=/work_results/$SWEEP

case ${1:-status} in
    status)
        apptainer exec --cleanenv --bind "$WORK_ROOT:/work_results" --bind "$HPC_DIR:/hpc" "$SIF" \
            julia /hpc/plan.jl "$RES" /opt/channelanigans/bench/configs.toml /hpc/walltimes.tsv \
                  "$MAX_WALLTIME_H" "$TIMEOUT_MARGIN_S"
        echo "queued/running:"; squeue -u "$USER" -o '%.10i %.40j %.8T %.10M %.10l %R' || true
        ;;
    final)
        echo "== table";            apptainer_exec julia /opt/channelanigans/bench/make_table.jl --out "$RES" --ci --flops --delta
        echo "== gradient check";   apptainer_exec julia /opt/channelanigans/bench/check_gradients.jl --out "$RES" \
                                        || echo "gradient check reported failures (see above)"
        echo "== raw-data export";  apptainer_exec julia /opt/channelanigans/bench/export_raw.jl --out "$RES" --archive
        echo "outputs in $WORK_ROOT/$SWEEP (table.tex, table.csv, export/)"
        ;;
    *) sed -n '2,8p' "$0"; exit 2 ;;
esac
