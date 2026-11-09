#!/bin/bash
# Create and fill the Zenodo drafts for the two records (current, aug2026) through the
# Zenodo REST API. It never publishes: review each draft on the website and publish
# there (publishing is permanent; files cannot be changed afterwards).
#
#   zenodo/upload.sh create   <record>        new draft; prints its id and reserved DOI
#   zenodo/upload.sh metadata <record> [DIR]  send metadata_<record>.json + description_<record>.html
#                                             (the snapshot commit in the description is read from
#                                             DIR/Channelanigans-<commit>.tar.gz; DIR as for upload)
#   zenodo/upload.sh upload   <record> [DIR]  upload the files in DIR (default $STAGE/<record>);
#                                             files already in the draft with the same MD5 are skipped
#   zenodo/upload.sh status   <record>        the draft's state, DOI and files
#
# Environment:
#   ZENODO_TOKEN    personal access token (scopes deposit:write, deposit:actions); required.
#                   Set it in the shell (read -rs ZENODO_TOKEN; export ZENODO_TOKEN), never in
#                   a file in this repo. It is passed to curl through a file descriptor, so it
#                   does not appear in the process list.
#   ZENODO_URL      https://sandbox.zenodo.org (default, for a trial run) or https://zenodo.org
#   DEPOSITION_ID   the draft's id (printed by `create`); required by every other command
#   STAGE           as in package_mac.sh / package_vista.sh (default ~/channelanigans_zenodo
#                   on macOS, $SCRATCH/channelanigans_zenodo elsewhere)
#
# Needs: bash, curl, python3 (any 3.x; for JSON).

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ZENODO_URL=${ZENODO_URL:-https://sandbox.zenodo.org}
ZENODO_URL=${ZENODO_URL%/}
if [[ $(uname) == Darwin ]]; then STAGE=${STAGE:-$HOME/channelanigans_zenodo}
else STAGE=${STAGE:-${SCRATCH:-$HOME}/channelanigans_zenodo}; fi

usage() { sed -n '2,24p' "$0"; exit 2; }
cmd=${1:-}; rec=${2:-}
[[ -n $cmd && ( $rec == current || $rec == aug2026 ) ]] || usage
[[ -n ${ZENODO_TOKEN:-} ]] || { echo "ZENODO_TOKEN is not set" >&2; exit 2; }
echo "server: $ZENODO_URL   record: $rec" >&2

# api <METHOD> <path or absolute URL> [curl args...]; prints the response body
api() {
    local method=$1 url=$2; shift 2
    [[ $url == http* ]] || url=$ZENODO_URL$url
    curl -sS --fail-with-body -X "$method" \
         -K <(printf 'header = "Authorization: Bearer %s"\n' "$ZENODO_TOKEN") "$@" "$url"
}
# json <python expression on d>: evaluate against the JSON on stdin
json() { python3 -c 'import json, sys; d = json.load(sys.stdin); print(eval(sys.argv[1]))' "$1"; }
need_id() { [[ -n ${DEPOSITION_ID:-} ]] || { echo "set DEPOSITION_ID (printed by: $0 create $rec)" >&2; exit 2; }; }
md5_of() { if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }

case $cmd in
    create)
        r=$(api POST /api/deposit/depositions -H 'Content-Type: application/json' -d '{}')
        id=$(json 'd["id"]' <<< "$r")
        doi=$(json 'd.get("metadata", {}).get("prereserve_doi", {}).get("doi", "(none)")' <<< "$r")
        echo "created draft $id for '$rec'"
        echo "  reserved DOI: $doi"
        echo "  edit page:    $ZENODO_URL/deposit/$id"
        echo "  next:         export DEPOSITION_ID=$id   (keep this per record and per server)"
        ;;
    metadata)
        need_id
        meta=$HERE/metadata_$rec.json desc=$HERE/description_$rec.html
        # The description names the repo snapshot being uploaded: take its commit from
        # the one Channelanigans-<commit>.tar.gz in the record's staging folder.
        shopt -s nullglob
        snaps=("${3:-$STAGE/$rec}"/Channelanigans-*.tar.gz)
        (( ${#snaps[@]} == 1 )) || {
            echo "need exactly one Channelanigans-<commit>.tar.gz in ${3:-$STAGE/$rec} (found ${#snaps[@]});" \
                 "run zenodo/package_mac.sh first, and delete snapshots of older commits" >&2; exit 2; }
        snap_commit=$(basename "${snaps[0]}" .tar.gz); snap_commit=${snap_commit#Channelanigans-}
        echo "snapshot commit: $snap_commit" >&2
        body=$(python3 - "$meta" "$desc" "$snap_commit" <<'EOF'
import json, sys
meta = json.load(open(sys.argv[1]))
desc = open(sys.argv[2]).read().replace("{{SNAPSHOT_COMMIT}}", sys.argv[3])
todo = [k for k, v in meta.items() if "TODO" in json.dumps(v)] + (["description"] if "TODO" in desc else [])
if todo:
    sys.exit("still TODO in: " + ", ".join(todo) + " (fill them in first)")
meta["description"] = desc
print(json.dumps({"metadata": meta}))
EOF
)
        r=$(api PUT "/api/deposit/depositions/$DEPOSITION_ID" -H 'Content-Type: application/json' --data-binary @- <<< "$body")
        echo "metadata set: $(json 'd["metadata"]["title"]' <<< "$r")"
        ;;
    upload)
        need_id
        dir=${3:-$STAGE/$rec}
        [[ -d $dir ]] || { echo "no directory $dir" >&2; exit 2; }
        dep=$(api GET "/api/deposit/depositions/$DEPOSITION_ID")
        bucket=$(json 'd["links"]["bucket"]' <<< "$dep")
        have=$(json '"\n".join(f["filename"] + " " + f["checksum"].replace("md5:", "") for f in d.get("files", []))' <<< "$dep")
        shopt -s nullglob
        for f in "$dir"/*; do
            name=$(basename "$f")
            [[ -f $f && $name != *.part && $name != .* ]] || continue
            sum=$(md5_of "$f")
            if grep -qx "$name $sum" <<< "$have"; then echo "  same  $name"; continue; fi
            grep -q "^$name " <<< "$have" && echo "  replace $name (different checksum)"
            echo "  put   $name ($(du -h "$f" | cut -f1))"
            r=$(api PUT "$bucket/$name" -H 'Content-Type: application/octet-stream' --upload-file "$f")
            got=$(json 'd["checksum"]' <<< "$r")
            [[ $got == "md5:$sum" ]] || { echo "checksum mismatch for $name: server $got, local md5:$sum" >&2; exit 1; }
        done
        echo "done; check: $0 status $rec"
        ;;
    status)
        need_id
        r=$(api GET "/api/deposit/depositions/$DEPOSITION_ID")
        python3 -c '
import json, sys
d = json.load(sys.stdin); m = d.get("metadata", {})
print("title:     ", m.get("title", "(none)"))
print("state:     ", d.get("state"), " submitted:", d.get("submitted"))
print("DOI:       ", d.get("doi") or m.get("prereserve_doi", {}).get("doi", "(none)"), "(reserved until published)" if not d.get("doi") else "")
print("edit page: ", d.get("links", {}).get("html"))
fs = d.get("files", [])
print("files:      %d, %.2f GB" % (len(fs), sum(f["filesize"] for f in fs) / 1e9))
for f in sorted(fs, key=lambda f: f["filename"]):
    print("   %10.1f MB  %s  md5:%s" % (f["filesize"] / 1e6, f["filename"], f["checksum"]))' <<< "$r"
        ;;
    *) usage ;;
esac
