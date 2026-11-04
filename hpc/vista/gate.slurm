#!/bin/bash
# Gate job: verify a gg node can produce valid table data before the sweep.
#
#   hpc/vista/submit.sh --gate        (from the repo root)
#
# 1. Node facts (CPU, memory, kernel, perf settings) and image provenance.
# 2. FLOP counters: perf_gate_test.jl counts one 1000×1000 Float64 matmul with the
#    Neoverse V2 FP events. Expected 2·1000³ = 2.0e9 FLOPs (each FMA counts 2).
# 3. End to end: reactant_full and enzyme_naive at N=4 with 2 measured calls through
#    the real orchestrator (perf + numactl forced on), so per-call FLOP attribution is
#    tested with the real events. Results in $WORK_ROOT/gate/.
#
#SBATCH -J ch-gate
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00

set -uo pipefail
# SLURM runs a spooled copy of this script, so locate env.sh via the submit dir.
source "${HPC_DIR:-$SLURM_SUBMIT_DIR/hpc/vista}/env.sh"
load_apptainer
GATE_DIR=$WORK_ROOT/gate/$SLURM_JOB_ID
mkdir -p "$SCRATCH_ROOT" "$WORK_ROOT" "$GATE_DIR"
fail=0

echo "===== 1. node and image"
echo "host: $(hostname)   job: $SLURM_JOB_ID   date: $(date -u +%FT%TZ)"
echo "kernel: $(uname -r)   perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
lscpu | grep -E 'Model name|^CPU\(s\)|Socket|NUMA node\(s\)|Thread'
free -g | head -2
numactl -H 2>/dev/null | grep -E 'available|size' || true
echo "apptainer: $(apptainer --version)"
apptainer_exec bash -c 'perf --version; head -3 /opt/channelanigans/BUILD_INFO.txt'
echo "sif sha256: $(sha256sum "$SIF" | cut -d' ' -f1)"  | tee "$GATE_DIR/sif_sha256.txt"

echo
echo "===== 2. FLOP counters (expect ≈ 2.0e9 FLOPs for the counted matmul)"
apptainer_exec bash -c '
    set -e
    d=$(mktemp -d); mkfifo $d/ctl $d/ack
    PERF_CTL_FIFO=$d/ctl PERF_ACK_FIFO=$d/ack \
        perf stat -x, -o $d/perf.csv --delay=-1 -e r80C0,r80C1 --control fifo:$d/ctl,$d/ack -- \
        julia /opt/channelanigans/bench/perf_gate_test.jl
    cat $d/perf.csv
    awk -F, "\$3 ~ /r80C/ || \$4 ~ /r80C/ { s += \$1 } END { printf \"FLOPs counted: %.4e   ratio to 2n^3: %.4f\n\", s, s/2e9 }" $d/perf.csv
    rm -rf $d' 2>&1 | tee "$GATE_DIR/flop_gate.txt"
ratio=$(awk '/ratio to 2n\^3/ {print $NF}' "$GATE_DIR/flop_gate.txt")
if [[ -n "$ratio" ]] && awk -v r="$ratio" 'BEGIN { exit !(r > 0.9 && r < 1.3) }'; then
    echo "FLOP gate: PASS (ratio $ratio)"
else
    echo "FLOP gate: FAIL (ratio '${ratio}' outside 0.9-1.3; inspect perf.csv above)"; fail=1
fi

echo
echo "===== 3. end-to-end attribution test (N=4, 2 measured calls)"
apptainer_exec julia /opt/channelanigans/bench/run_cases.jl \
    --configs reactant_full,enzyme_naive --nsteps 4 --out /work_results/gate/$SLURM_JOB_ID/e2e \
    --perf on --numactl on --core "$CORE" --timeout 3600 \
    -- --repeats 2 --save-gradients || fail=1
apptainer_exec julia -e '
    using JSON
    for c in ("reactant_full", "enzyme_naive")
        p = joinpath(ARGS[1], c, "N0004", "result.json")
        isfile(p) || (println(c, ": no result.json"); continue)
        r = JSON.parsefile(p)
        fl = [x["flops"] for x in r["calls"] if x["kind"] == "measured"]
        println(rpad(c, 16), " attribution: ", r["perf"]["attribution"],
                "   wall per call: ", [round(x["wall_s"], digits=2) for x in r["calls"]],
                "   FLOPs per call: ", fl,
                "   XLA cost-analysis FLOPs: ", get(r, "xla_cost_analysis_flops", "n/a"),
                "   objective: ", r["objective"])
    end' /work_results/gate/$SLURM_JOB_ID/e2e | tee "$GATE_DIR/e2e_summary.txt"
[[ $(grep -c "attribution: ok" "$GATE_DIR/e2e_summary.txt") -eq 2 ]] || fail=1

echo
[[ $fail -eq 0 ]] && echo "GATE PASSED" || echo "GATE FAILED (see above)"
exit $fail
