#!/bin/bash
# Package the Zenodo files that exist only on Vista, one folder per record (the same
# records as zenodo/package_mac.sh):
#
#   <data>_superseded.tar.gz   earlier attempts that case.slurm (or a manual promotion)
#                              moved to $WORK_ROOT/_superseded/<sweep>/; never deleted
#   slurm_logs_<record>.tar.gz SLURM logs of the record's jobs (job-name prefix ch- or ch-aug-)
#   gate_<record>.tar.gz       gate runs ($WORK_ROOT/gate/<jobid>/) made with the record's image
#   diagnostics_aug2026.tar.gz (aug2026 only) the diag_* jobs cited in the data README
#   sif_sha256.txt             SHA-256 of the SIF the jobs ran (the SIF itself only with WITH_SIF=1)
#   SHA256SUMS_vista
#
# Run it in an idev session or as a job (tar/sha256 over a few GB; not on a login node):
#   sbatch -A Transfer-of-ECCO-glo -o $WORK/channelanigans_results/slurm_logs/%x.%j.out \
#          zenodo/package_vista.sh                 (from the repo root)
#   zenodo/package_vista.sh [current|aug2026|all]  (default: all)
#
# Environment:
#   STAGE=DIR   output root (default: $SCRATCH/channelanigans_zenodo); DIR/<record>/...
#   WITH_SIF=1  also copy the SIF into the record (~6 GB each; the docker archive from
#               package_mac.sh is the primary image file; to rebuild a SIF from it:
#               gunzip -k <file>.docker.tar.gz
#               apptainer build x.sif docker-archive://<file>.docker.tar)
#   FORCE=1     rebuild files that already exist
#
#SBATCH -J ch-zenodo-package
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

set -euo pipefail
HPC_DIR=${HPC_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/hpc/vista}
STAGE=${STAGE:-$SCRATCH/channelanigans_zenodo}
FORCE=${FORCE:-0}
WITH_SIF=${WITH_SIF:-0}

case ${1:-all} in
    all) records=(current aug2026) ;;
    current|aug2026) records=("$1") ;;
    *) sed -n '2,25p' "$0"; exit 2 ;;
esac

# tar_gz <output> <dir to cd into> <paths...>
tar_gz() {
    local out=$1 dir=$2; shift 2
    if [[ -f $out && $FORCE != 1 ]]; then echo "  keep  $(basename "$out") (exists)"; return; fi
    echo "  make  $(basename "$out")  ($# entries)"
    (cd "$dir" && tar -czf "$out.part" "$@")
    mv "$out.part" "$out"
}

for rec in "${records[@]}"; do
    # Profile settings (SIF, SWEEP, JOB_PREFIX, IMAGE_DIGEST, WORK_ROOT, LOG_DIR) from env.sh
    case $rec in current) prof=current; data=vista_2026-11 ;; aug2026) prof=aug2026; data=vista_2026-11_aug2026 ;; esac
    eval "$(CH_PROFILE=$prof bash -c 'source "$1/env.sh" >/dev/null
        printf "SIF=%q SWEEP=%q JOB_PREFIX=%q IMAGE_DIGEST=%q WORK_ROOT=%q LOG_DIR=%q\n" \
               "$SIF" "$SWEEP" "$JOB_PREFIX" "$IMAGE_DIGEST" "$WORK_ROOT" "$LOG_DIR"' _ "$HPC_DIR")"
    out=$STAGE/$rec
    mkdir -p "$out"
    echo "== $rec ($SWEEP, $JOB_PREFIX-*) -> $out"

    # 1. Superseded attempts
    if [[ -d $WORK_ROOT/_superseded/$SWEEP ]]; then
        tar_gz "$out/${data}_superseded.tar.gz" "$WORK_ROOT/_superseded" "$SWEEP"
    else
        echo "  none  _superseded/$SWEEP"
    fi

    # 2. SLURM logs: ch-aug-* for aug2026; ch-* for current, but not ch-aug-* or this
    #    script's own ch-zenodo-* logs
    mapfile -t logs < <(cd "$LOG_DIR" && ls | awk -v p="$JOB_PREFIX-" -v r="$rec" \
        'index($0, p) == 1 && !(r == "current" && ($0 ~ /^ch-aug-/ || $0 ~ /^ch-zenodo-/))' | sort)
    if (( ${#logs[@]} > 0 )); then
        tar_gz "$out/slurm_logs_$rec.tar.gz" "$LOG_DIR" "${logs[@]}"
    else
        echo "  none  SLURM logs with prefix $JOB_PREFIX-"
    fi

    # 3. Gate runs made with this record's image (the digest is in their result files)
    gates=()
    for g in "$WORK_ROOT"/gate/*/; do
        [[ -d $g ]] && grep -rqs "${IMAGE_DIGEST#sha256:}" "$g" && gates+=("gate/$(basename "$g")")
    done
    if (( ${#gates[@]} > 0 )); then
        tar_gz "$out/gate_$rec.tar.gz" "$WORK_ROOT" "${gates[@]}"
    else
        echo "  none  gate runs with image $IMAGE_DIGEST"
    fi

    # 4. Diagnostics (all aug2026; older attempts kept under their timestamped names)
    if [[ $rec == aug2026 ]]; then
        mapfile -t diags < <(cd "$WORK_ROOT" && ls -d diag_* 2>/dev/null | sort)
        if (( ${#diags[@]} > 0 )); then
            tar_gz "$out/diagnostics_aug2026.tar.gz" "$WORK_ROOT" "${diags[@]}"
        else
            echo "  none  diag_* in $WORK_ROOT"
        fi
    fi

    # 5. The SIF: its checksum always, the file itself only on request
    if [[ -f $out/sif_sha256.txt && $FORCE != 1 ]]; then
        echo "  keep  sif_sha256.txt (exists)"
    else
        echo "  hash  $(basename "$SIF")"
        (cd "$(dirname "$SIF")" && sha256sum "$(basename "$SIF")") > "$out/sif_sha256.txt"
    fi
    if [[ $WITH_SIF == 1 ]]; then
        if [[ -f $out/$(basename "$SIF") && $FORCE != 1 ]]; then
            echo "  keep  $(basename "$SIF") (exists)"
        else
            echo "  copy  $(basename "$SIF")"; cp "$SIF" "$out/"
        fi
    fi

    # 6. Checksums and sizes
    (cd "$out" && sha256sum $(ls | grep -v -e '^SHA256SUMS' -e '\.part$') > SHA256SUMS_vista)
    echo "  files:"; (cd "$out" && ls -lh | awk 'NR > 1 {printf "    %8s  %s\n", $5, $9}')
done
echo
echo "staged under $STAGE; nothing uploaded"
