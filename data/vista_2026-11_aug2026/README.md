# CPU AD cost benchmarks: TACC Vista, November 2026, July/August 2026 environment

This folder holds a rerun of the CPU AD cost table in the software environment of the
original July/August 2026 runs: the cost of one reverse-mode gradient of the channel
model for the 8 AD configurations of the original table at N = 4, 8, 16, 32, 64, 128
and 256 timesteps. It sits next to [`../vista_2026-11/`](../vista_2026-11/), the same
table with current package versions.

> **Status (2026-11-08): complete.** All 56 cases are in: 53 finished (`ok`) and 3
> ran out of the node's memory (`m`): `reactant_mincut_off` at N = 128 and 256, and
> `enzyme_naive` at N = 256. Read "Correctness" before using the
> `reactant_after_enzyme` row or `reactant_binomial_10` at N = 8.

> **Timings vary strongly between Vista nodes.** Each case ran on whichever `gg`
> node was free (33 nodes in total). The same run can take 6% to about 20% longer
> on one node than on another, far more than the confidence intervals in the table
> (usually under 1%), which only cover the 12 calls on one node. Runtime
> differences between rows, or against the original runs, of less than about 20% at
> large N are within this node-to-node spread. See "Reading the table".

## How the data was produced

- **Hardware:** one TACC Vista `gg` node per case (2× NVIDIA Grace, Neoverse V2, 144
  cores, 237 GiB), 33 different nodes in total. The benchmark process is pinned to
  a single core (`numactl -C 0`).
- **Software:** the frozen container image
  `ghcr.io/jlk9/channelanigans-artifact:2026-11-aug2026-cpu-2`, digest
  `sha256:d9a137735986b897d88c22337f3b06b16868b3a19dcbd6c2089bad1c508edcb3` (built
  from repo commit `d7c03aa`). It has Julia 1.11.9 and the exact package set of the
  original runs' Manifest (recovered from the August project directory, last used
  2026-08-03; pinned in [`envs/aug2026/`](../../envs/aug2026/)): Reactant 0.2.279,
  Enzyme 0.13.198, Oceananigans 0.110.14, Checkpointing.jl 0.12.0, CUDA.jl 6.2.1.
- **Runtime flags:** those of the original job scripts, applied to every row:
  Julia's default `-O2`, `JULIA_NUM_THREADS=1`,
  `XLA_FLAGS="--xla_backend_extra_options=xla_cpu_use_multi_output_fusion"` plus an
  HLO dump (`--xla_dump_to`), with XLA's multi-threaded Eigen at its default (on).
  The original scripts differed slightly between rows; the `mincut_off` script, for
  example, turned multi-threaded Eigen off and multi-output fusion off.
- **Protocol, metrics and FLOP counting:** as in `../vista_2026-11/`. Each case runs
  in a fresh process: 1 warmup call (includes compilation; excluded), then 12
  measured calls, each on a freshly built problem. The table reports mean ± 95%
  Student-t confidence interval over the 12 calls. Runtime is wall-clock time per
  call; memory is peak resident memory during the call (Linux `VmHWM`); FLOPs are
  the hardware counters `r80C0` + `r80C1`, counted only inside each measured call.

The full methodology is in [`export/DATA_README.md`](export/DATA_README.md). The SLURM
scripts are in [`hpc/vista/`](../../hpc/vista/) (profile `aug2026` in `env.sh`).

## What differs from the original July/August 2026 runs

The rerun reproduces the original environment as closely as one image allows, and
measures every row the same way. Four differences are deliberate or unavoidable:

1. **Package versions per row.** The original rows were not all run with the same
   versions:
   - `reactant_full`, `reactant_after_enzyme`, `reactant_mincut_off`: Reactant
     0.2.277 and Enzyme 0.13.196 (July 31 to August 2);
   - `reactant_periodic`, `reactant_periodic_mincut_off`: Reactant 0.2.278 and
     Enzyme 0.13.198 (August 2 to 3);
   - `reactant_binomial_10`, `reactant_binomial_log2`: Reactant 0.2.279 (August 3);
   - `enzyme_naive`: Enzyme 0.13.198, Oceananigans 0.110.14.

   The image has the August 3 package set, so only the two binomial rows and
   `enzyme_naive` run exactly the original versions.
2. **Explicit first-step initialization (Reactant rows).** Before the time loop, the
   Reactant driver calls `Oceananigans.initialize!` and `update_state!`. On CPU,
   Oceananigans does this inside the first `time_step!`; under Reactant that step is
   a no-op, so without the explicit calls the first step reads stale halos and
   tendencies and the objective differs from CPU by about 1e-4 after 4 steps. The
   original scripts did not have it. It changes the compiled loop and costs time:
   `reactant_full` at N = 8 took 23.0 s per call without it and 28.0 s with it, on
   the same node (diagnostic job 1128171). The checkpointed rows are hardly
   affected: `reactant_binomial_10` takes 24.4, 50.3 and 102.0 s at N = 8, 16 and
   32, against 24.2, 51.0 and 103.7 s originally (191.5 against 209.1 s at
   N = 64, within the node-to-node spread).
3. **One driver, one initial state, one statistic.** Every row runs
   `bench/driver_reactant.jl` or `bench/driver_enzyme.jl` on the same model
   (`src/ChannelModel.jl`) from the same seeded initial state, and reports the mean
   of 12 calls. The original Enzyme-only script differentiated a different initial
   state by accident: its spinup call was commented out, but the two lines after it,
   `set!(Tᵢ, model.tracers.T)` and `set!(Sᵢ, model.tracers.S)`, still copied the
   never-set (zero) model tracers into the initial conditions. So the original
   `enzyme_naive` row started from T = S = 0, while the original Reactant rows
   started from the stratified state with S = 35.
4. **Gradient files are written with `IOStream`.** JLD2's default memory-mapped
   writes fail under Julia 1.11 on Vista's 64 KiB-page kernel. This happens outside
   the timed window.

## Files

| Path | Contents |
|---|---|
| `table.tex`, `table.csv` | The table: FLOPs, runtime, memory and Δmemory per config and N, with CI half-widths |
| `export/raw_runs.csv` | One row per call (warmup + measured) for every case: time, memory, FLOPs, per-event counts, objective |
| `export/cases.csv` | One row per case: outcome, summary statistics, versions, flags, host |
| `export/DATA_README.md` | How each field is collected (two paragraphs corrected by hand; marked) |
| `<config>/N<nnnn>/result.json` | Full record of a finished case: every call, statistics, gradient summaries, environment |
| `<config>/N<nnnn>/status.json` | Outcome (`ok` / `t` / `m` / `s` / `e`), exact command, flags |
| `<config>/N<nnnn>/job_info.json` | SLURM job, node, image digest |
| `<config>/N<nnnn>/log.txt` | Driver output |
| `<config>/N<nnnn>/xla_buffer_assignment.txt` | XLA's "Total bytes used" per compiled module, from the HLO dump (extra information, not the memory metric) |
| `enzyme_naive/N0128/memory.csv` | Driver RSS and node `MemAvailable`, sampled every 5 s (see below) |

**Not in this repository** because of size (for the Zenodo archive with the image):
`perf.csv` (raw 100 ms counter intervals), `gradients.jld2` (full gradient fields),
`raw_data.tar.gz`, and the sweep's `_superseded/` directory (failed earlier attempts).

## Correctness

`bench/check_gradients.jl` compares the objective and the full gradient fields of
every configuration against `reactant_full` at each N (relative error
max|a − b| / max|b|, tolerance 1e-10). In the current-version sweep everything
agrees to within 1.9e-12. Here, 14 comparisons fail:

| Configuration | N | Worst relative difference |
|---|---|---|
| `reactant_after_enzyme` | 32, 64, 128, 256 | 1.5e-5 (objective), 8.9e-5 (objective), 4.5e-4 (`dv_wind_stress`), 4.3e-4 (`dT_flux`) |
| `reactant_binomial_10` | 8 | 4.1e-4 (`du_wind_stress`); objective agrees |
| `enzyme_naive` | 16, 32, 64, 128 | 1.5e-10, 2.7e-9, 4.6e-8, 1.6e-7 (all `dT_flux`); objective agrees to ~1e-12 |
| `reactant_mincut_off` | 32, 64 | 1.7e-10, 4.2e-8 (`dT_flux`) |
| `reactant_periodic_mincut_off` | 64, 128, 256 | 8.7e-10, 1.1e-8, 5.1e-9 (`dT_flux`) |

Every other comparison passes, and every case returns the identical objective in all
13 of its calls. Two groups:

- **Large differences (open issue).** `reactant_after_enzyme` computes a measurably
  different objective from N = 32 on, and `reactant_binomial_10` one different
  gradient field at N = 8 only. In both cases the program is deterministic, so the
  timings are valid measurements of what that compiled program does. But it is
  not numerically the same gradient as the other rows, which the current-version
  sweep's versions do compute. We have not found the cause. The original logs do
  not record objectives, so we cannot tell whether the original runs had the same
  difference.
- **Small differences growing with N.** Up to 2e-7, almost entirely in `dT_flux`,
  in the native Enzyme row and the two mincut-off rows. This looks like rounding
  differences amplified over the time loop. It is too small to matter for the
  timings, but above the 1e-10 tolerance chosen for the current versions.

To repeat the check (needs the `gradients.jld2` files from the archive):
`julia --project bench/check_gradients.jl --out data/vista_2026-11_aug2026`.

## Reading the table

- **Out of memory:** three cases exceeded the node's memory.
  - `reactant_mincut_off` at N = 128 and 256: XLA failed to allocate a single
    buffer of 411.9 GB and 821.0 GB respectively.
  - `enzyme_naive` at N = 256: killed by the kernel's out-of-memory killer (job
    1128445, run with `perf` off so that the kill is classified as `m`; see
    `../vista_2026-11/README.md` for why).
- **Superscript n:** `enzyme_naive` at N = 128 finished but has no FLOP count
  (`perf` off; see below). The per-call FLOPs of this row are linear in N up to N = 64; we do not
  extrapolate.
- **N = 4:** as in the current-version sweep, several Reactant configurations use
  more memory at N = 4 than at N = 8, and their N = 4 calls are unusually fast: the
  compiled program at very short loops differs.
- **Node-to-node variation on Vista is the largest uncertainty in the runtimes.**
  - **How large:** in diagnostic jobs, the identical native-Enzyme run at N = 64
    (same image, driver, flags and protocol) took 139.7 s per call on one node and
    148.4 s on another (6%). The original Enzyme-only script, run identically in the
    image, took 152.0 s, 150 to 159 s and 169 to 185 s on three nodes (up to about
    20%).
  - **What the intervals cover:** the ± values in the table are 95% confidence
    intervals over the 12 calls on one node, typically under 1% of the mean. They
    do not include this spread between nodes.
  - **Why it matters here:** every case ran on whichever `gg` node was free (33
    different nodes; recorded in `job_info.json` and in the `hostname` column of
    `export/cases.csv`), so rows and N values were measured on different nodes. A
    runtime difference between two cells of less than about 20%, especially for the
    memory-heavy cases (native Enzyme, mincut off, large N), may come from the
    nodes rather than the configurations. Comparisons with the original July/August
    runs, which also ran on unrequested nodes, have the same limitation.
  - **Likely causes:** differences in node state, not in software. The nodes
    started with different amounts of memory already in use (11 GiB on the fastest
    node in the diagnostics, 36 to 37 GiB on the slowest). We have not isolated the
    cause.
  - **What would reduce it:** repeating selected cells on several nodes, or running
    the rows to be compared on one node in one job. Memory and FLOP counts are not
    affected: they agree across nodes to within 0.5%.

## `enzyme_naive` at N = 128: the entry comes from a diagnostic job

The case needs 221.3 GiB (mean peak over the 12 calls) on a 237 GiB node. That fits
only when the node starts nearly empty: on the node that ran it, the node's free
memory fell to 0.4 GiB at the peak. The sweep's own attempts, on nodes that started
with more memory in use, ran out of memory. They are kept in `_superseded/`.

The entry is the `default` variant of `hpc/vista/diag_memory.slurm` (job 1128458,
node i615-002): same image, same driver (`bench/driver_enzyme.jl`), same Julia flags,
same protocol (1 warmup + 12 measured, `--save-gradients`, `numactl -C 0`) as the
sweep, copied into this folder. It differs in two ways that do not affect the
measurement: `perf` was off, so the FLOPs cell is empty; and `XLA_FLAGS` were the
orchestrator's defaults, which native Enzyme never reads. Its `job_info.json`
records this. `memory.csv` is the 5 s memory trace of that run.

The same job also ran the case with `julia --heap-size-hint=200G`: 221.3 GiB and
349.0 s per call, against 221.3 GiB and 346.4 s without. The memory is live data
(Enzyme's stored tape), not uncollected garbage, so no GC setting makes the case fit
with more margin. The original run reported 222.0 GiB.

## Native Enzyme: why it is faster than in the original runs

At N = 64, the original table reports 184.0 s per call for `enzyme_naive`; this
table reports 140.6 s, with the same package versions. Diagnostic jobs ruled out, or
measured, every candidate cause we found:

- **Same work.** FLOPs (3.019e11 here against 3.019e11 originally), allocations
  (20,734,447 against 20,734,450 per call) and peak memory (114.6 against 114.9 GiB)
  match. Both differentiate a plain loop; the `@ad_checkpoint` line in some copies
  of the original script would not run with Checkpointing.jl 0.12.0, and the
  original logs show the plain loop.
- **Not the environment.** Thread count (1), Julia build (official 1.11.9),
  compilation target (`neoverse-v2`), `-O2`, bounds checking (off) and XLA flags
  (native Enzyme does not load XLA) are the same. Run on the same node (job
  1130081), the original script took 163.7 to 180.0 s per call in the original
  Vista setup, 163.3 to 179.8 s with TACC's XALT library and the NVIDIA library
  path removed, and 169.4 to 185.4 s in this image. The image is not faster.
- **Not the measurement method.** The original `@btime` timed a single call after a
  garbage collection (one call far exceeds its 5 s budget), as our driver does.
- **Not the original script's other differences.** Measured in the image (jobs
  1129118 and 1129580): loading CUDA.jl (139.2 against 139.7 s), reusing one model
  across calls (138.0 s), and the T = S = 0 initial state (147.6 against 148.4 s on
  the second node) make no difference.
- **What remains.** Node-to-node variation of up to about 20% for this
  memory-heavy case (see "Reading the table"), and the original script running 2
  to 9% slower than our driver on the same node, depending on node and call. The
  original measurement (184 s) is within the range the original script gives today
  (150 to 185 s), so the difference from the original table is explained by the
  nodes plus a few percent from the script, not by the software.

## Diagnostic jobs

All in the aug2026 image unless stated; results in `$WORK/channelanigans_results/`
on Vista (for the archive).

| Job | Script | Question | Result |
|---|---|---|---|
| 1128171 | `diag_firststep.slurm` | Cost of the first-step initialization | `reactant_full` N = 8: 23.0 s without, 28.0 s with |
| 1128458 | `diag_memory.slurm` | Does `enzyme_naive` N = 128 fit; does a GC hint help | Fits (221.3 GiB, 0.4 GiB to spare); the hint changes nothing |
| 1129118 | `diag_enzyme_harness.slurm` | CUDA.jl loaded, model reuse, original script (N = 64) | 139.7 / 139.2 / 138.0 s; original script 152.0 s |
| 1129580 | `diag_enzyme_harness.slurm` | Initial state (N = 64) | Driver 148.4 s, zero state 147.6 s; original script 150 to 159 s either way |
| 1130081 | `diag_enzyme_env.slurm` | Original Vista setup vs XALT-free vs image, one node (N = 64) | 164 to 180 s / 163 to 180 s / 169 to 185 s |

## Regenerating the table

From the repo root:

```bash
CONFIGS=reactant_full,reactant_after_enzyme,reactant_mincut_off,reactant_periodic,reactant_periodic_mincut_off,reactant_binomial_10,reactant_binomial_log2,enzyme_naive
julia --project bench/make_table.jl --out data/vista_2026-11_aug2026 --ci --flops --delta --configs $CONFIGS
julia --project bench/export_raw.jl --out data/vista_2026-11_aug2026
```

Both read only the committed JSON files. `export_raw.jl` regenerates
`export/DATA_README.md` without the two hand corrections.
