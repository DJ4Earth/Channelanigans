#!/bin/bash
# Package the Zenodo files that live on the Mac, one folder per record:
#
#   current   image ghcr.io/jlk9/channelanigans-artifact:2026-11-cpu   (sha256:b9c94c1f...)
#             + the bulky files of data/vista_2026-11
#   aug2026   image ghcr.io/jlk9/channelanigans-artifact:2026-11-aug2026-cpu-2 (sha256:d9a13773...)
#             + the bulky files of data/vista_2026-11_aug2026
#             + the original July/August 2026 runs (old_runs_pre_artifact/) and evidence/
#
# Each record also gets a snapshot of this repo at HEAD (git archive) and a SHA256SUMS.
# The Vista-only files (_superseded/, diagnostics, SLURM logs, SIF checksums) come from
# zenodo/package_vista.sh. Nothing is uploaded; see the Zenodo plan in the data READMEs.
#
#   zenodo/package_mac.sh [current|aug2026|all]     (default: all)
#
# Environment:
#   STAGE=DIR      output root (default: ~/channelanigans_zenodo); DIR/<record>/...
#   SKIP_IMAGES=1  skip the docker save step (the slow one: ~6 GB per image)
#   FORCE=1        rebuild files that already exist (default: keep them, so a rerun resumes)
#
# Needs: docker (with both images pulled at the exact digests), git, bsdtar, shasum.

set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STAGE=${STAGE:-$HOME/channelanigans_zenodo}
IMAGE_REPO=ghcr.io/jlk9/channelanigans-artifact
FORCE=${FORCE:-0}
SKIP_IMAGES=${SKIP_IMAGES:-0}
export COPYFILE_DISABLE=1          # macOS tar: no AppleDouble ._ files

case ${1:-all} in
    all) records=(current aug2026) ;;
    current|aug2026) records=("$1") ;;
    *) sed -n '2,22p' "$0"; exit 2 ;;
esac

gz() { if command -v pigz >/dev/null 2>&1; then pigz -n; else gzip -n; fi; }

# tar_gz <output> <dir to cd into> <paths...>; writes to a .part file first
tar_gz() {
    local out=$1 dir=$2; shift 2
    if [[ -f $out && $FORCE != 1 ]]; then echo "  keep  $(basename "$out") (exists)"; return; fi
    echo "  make  $(basename "$out")"
    (cd "$dir" && tar -cf - --exclude '.DS_Store' --exclude '._*' "$@") | gz > "$out.part"
    mv "$out.part" "$out"
}

commit=$(git -C "$REPO" rev-parse HEAD)
short=${commit:0:7}
if ! git -C "$REPO" diff --quiet HEAD; then
    echo "note: tracked files have uncommitted changes; the repo snapshot is HEAD ($short) without them" >&2
fi

for rec in "${records[@]}"; do
    case $rec in
        current) tag=2026-11-cpu;            digest=sha256:b9c94c1fbac86e8b7cdc709540fd97850f2069a61bbf480f5709b31b43676df0; data=vista_2026-11 ;;
        aug2026) tag=2026-11-aug2026-cpu-2;  digest=sha256:d9a137735986b897d88c22337f3b06b16868b3a19dcbd6c2089bad1c508edcb3; data=vista_2026-11_aug2026 ;;
    esac
    out=$STAGE/$rec
    mkdir -p "$out"
    echo "== $rec -> $out"

    # 1. Container image: docker save of the exact digest
    ref=$IMAGE_REPO:$tag
    img=$out/channelanigans-artifact_${tag}.docker.tar.gz
    if [[ $SKIP_IMAGES == 1 ]]; then
        echo "  skip  image (SKIP_IMAGES=1)"
    else
        digests=$(docker image inspect --format '{{join .RepoDigests " "}}' "$ref")
        [[ " $digests " == *" $IMAGE_REPO@$digest "* ]] || {
            echo "$ref is not $digest locally (has: $digests); docker pull $IMAGE_REPO@$digest first" >&2; exit 2; }
        platform=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$ref")
        docker image inspect "$ref" > "$out/image_inspect.json"
        if [[ -f $img && $FORCE != 1 ]]; then
            echo "  keep  $(basename "$img") (exists)"
        else
            echo "  make  $(basename "$img")  ($platform; several minutes)"
            docker save --platform "$platform" "$ref" | gz > "$img.part"
            mv "$img.part" "$img"
        fi
    fi

    # 2. Bulky data not in git: raw_data.tar.gz (all case folders incl. perf.csv, as
    #    built by bench/export_raw.jl --archive) and the gradient fields (.jld2)
    raw=$REPO/data/$data/raw_data.tar.gz
    [[ -f $raw ]] || { echo "missing $raw" >&2; exit 2; }
    if [[ -f $out/${data}_raw_data.tar.gz && $FORCE != 1 ]]; then
        echo "  keep  ${data}_raw_data.tar.gz (exists)"
    else
        echo "  copy  ${data}_raw_data.tar.gz"; cp "$raw" "$out/${data}_raw_data.tar.gz"
    fi
    njld=$(find "$REPO/data/$data" -name '*.jld2' | wc -l | tr -d ' ')
    (( njld > 0 )) || { echo "no .jld2 files under data/$data" >&2; exit 2; }
    (cd "$REPO/data" && find "$data" -name '*.jld2' | sort) > "$out/.jld2_list"
    if [[ -f $out/${data}_gradients.tar.gz && $FORCE != 1 ]]; then
        echo "  keep  ${data}_gradients.tar.gz (exists)"
    else
        echo "  make  ${data}_gradients.tar.gz ($njld files)"
        (cd "$REPO/data" && tar -cf - -T "$out/.jld2_list") | gz > "$out/${data}_gradients.tar.gz.part"
        mv "$out/${data}_gradients.tar.gz.part" "$out/${data}_gradients.tar.gz"
    fi
    rm -f "$out/.jld2_list"

    # 3. The original July/August 2026 runs and their scripts (aug2026 record only)
    if [[ $rec == aug2026 ]]; then
        old=()
        for d in "$REPO"/old_runs_pre_artifact/cpu_runs_* "$REPO"/old_runs_pre_artifact/runs_optimized \
                 "$REPO"/old_runs_pre_artifact/runs_after_enzyme; do
            [[ -d $d ]] && old+=("old_runs_pre_artifact/$(basename "$d")")
        done
        (( ${#old[@]} > 0 )) || { echo "no original runs under old_runs_pre_artifact/" >&2; exit 2; }
        tar_gz "$out/original_runs_2026-07-08.tar.gz" "$REPO" "${old[@]}" evidence
    fi

    # 4. The repo at HEAD (code, SLURM scripts, committed data and READMEs)
    snap=$out/Channelanigans-$short.tar.gz
    if [[ -f $snap && $FORCE != 1 ]]; then
        echo "  keep  $(basename "$snap") (exists)"
    else
        echo "  make  $(basename "$snap")"
        git -C "$REPO" archive --format=tar --prefix="Channelanigans-$short/" HEAD | gz > "$snap.part"
        mv "$snap.part" "$snap"
    fi

    # 5. Checksums and sizes (package_vista.sh writes its own SHA256SUMS_vista)
    (cd "$out" && shasum -a 256 $(ls | grep -v -e '^SHA256SUMS' -e '\.part$') > SHA256SUMS)
    echo "  files:"; (cd "$out" && ls -lh | awk 'NR > 1 {printf "    %8s  %s\n", $5, $9}')
done
echo
echo "repo snapshot: $commit"
echo "next: zenodo/package_vista.sh on Vista for _superseded/, diagnostics, SLURM logs, SIF checksums"
