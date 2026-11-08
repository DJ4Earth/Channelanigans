# CPU AD cost benchmarks: TACC Vista, November 2026

This folder holds the data behind the CPU AD cost table: the cost of one
reverse-mode gradient of the channel model for 9 AD configurations at N = 4, 8, 16,
32, 64, 128 and 256 timesteps.

> **Status (2026-11-04): complete.** All 63 cases are in: 60 finished (`ok`) and 3
> ran out of the node's memory (`m`): `reactant_mincut_off` at N = 128 and 256,
> and `enzyme_naive` at N = 256.

## How the data was produced

- **Hardware:** one TACC Vista `gg` node per case (2× NVIDIA Grace, Neoverse V2, 144
  cores, 237 GiB). The benchmark process is pinned to a single core
  (`numactl -C 0`).
- **Software:** the frozen container image
  `sha256:b9c94c1fbac86e8b7cdc709540fd97850f2069a61bbf480f5709b31b43676df0`
  (built from repo commit `8fcc665`). It has Julia 1.12.7, Oceananigans 0.116.1,
  Reactant 0.2.304, Enzyme 0.13.216 and Checkpointing.jl 0.13.0, and runs with
  `julia -O0` and
  `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_backend_extra_options=xla_cpu_use_multi_output_fusion"`.
- **Protocol:** each case runs in a fresh process. It makes 1 warmup call (which
  includes compilation and is excluded from the statistics), then 12 measured
  calls, each on a freshly built problem. The table reports mean ± 95% Student-t
  confidence interval over the 12 calls.
- **Runtime:** wall-clock time of one call.
- **Memory:** peak resident memory during the call (Linux `VmHWM`, reset before
  each call).
- **FLOPs:** hardware counters `r80C0` + `r80C1` (Neoverse V2 floating-point
  operations), counted only inside each measured call. Validated against a matmul
  with a known count (2.0245e9 counted against 2.0e9 exact).

The full methodology, including every field in the CSV files, is in
[`export/DATA_README.md`](export/DATA_README.md). The SLURM scripts that ran the
sweep are in [`hpc/vista/`](../../hpc/vista/).

## Files

| Path | Contents |
|---|---|
| `table.tex`, `table.csv` | The table: FLOPs, runtime, memory and Δmemory per config and N, with CI half-widths |
| `export/raw_runs.csv` | One row per call (warmup + measured) for every case: time, memory, FLOPs, per-event counts, objective |
| `export/cases.csv` | One row per case: outcome, summary statistics, versions, flags, host |
| `export/DATA_README.md` | How each field is collected |
| `<config>/N<nnnn>/result.json` | Full record of a finished case: every call, statistics, gradient summaries, environment |
| `<config>/N<nnnn>/status.json` | Outcome (`ok` / `t` / `m` / `s` / `e`), exact command, flags |
| `<config>/N<nnnn>/job_info.json` | SLURM job, node, image digest, Apptainer version |
| `<config>/N<nnnn>/log.txt` | Driver output |

**Not in this repository** because of size (archived on Zenodo with the container
image; DOI 10.5072/zenodo.637533):
- `perf.csv`: the raw 100 ms hardware-counter intervals behind the per-call FLOPs,
  about 200 MB in total.
- `gradients.jld2`: the full gradient fields used for the cross-configuration
  check, about 390 MB.

## Correctness

Every finished configuration agrees with `reactant_full` at every N, in both the
objective and the full gradient fields:
- worst relative difference among Reactant configurations: 4.6e-13;
- worst between Reactant and the Enzyme-only (no Reactant) configurations: 1.9e-12;
- tolerance: 1e-10.

Run `julia --project bench/check_gradients.jl --out data/vista_2026-11` to repeat
the check; it needs the `gradients.jld2` files from the Zenodo archive.

## Reading the table

- **Out of memory:** three cases exceeded the node's memory.
  - `reactant_mincut_off` at N = 128 and 256: XLA reports it directly. At N = 128 it
    failed to allocate a single 363 GB buffer
    (`RESOURCE_EXHAUSTED: Out of memory allocating 363163280400 bytes`).
  - `enzyme_naive` at N = 256: killed by the kernel's out-of-memory killer (see
    below).
- **N = 4:** several Reactant configurations use more memory at N = 4 than at N = 8,
  and their N = 4 calls are unusually fast. This was seen on every machine tested,
  so it is a property of the compiled program at very short loops, not noise.
- **Two kinds of FLOP count:** the FLOPs are hardware counts of executed
  floating-point operations. For Reactant runs, `result.json` also records XLA's
  static cost-analysis estimate (`xla_cost_analysis_flops`). That estimate is
  lower: 1.61e10 against 2.32e10 counted for `reactant_full` at N = 4. The two
  measure different things.

## `enzyme_naive` at N = 256: run without hardware counters

Memory for this configuration grows linearly in N (138.7 GiB at N = 128), so
N = 256 needs about 270 GiB, more than the node has. The kernel's out-of-memory
killer stopped the process.

**First run, under `perf` (job 1120577): recorded as `e`.**
- `perf stat` reports a killed child process as a clean exit, so the orchestrator
  saw no error.
- SLURM recorded a maximum memory of about 219.5 GiB before the process died.

**Rerun, without `perf` (job 1121831): recorded as `m`.** The kill reached the
orchestrator directly as `SIGKILL`, and it classified the case as `m` ("killed by
SIGKILL (likely OOM killer)"). This rerun is the result reported here; its
`job_info.json` records `"perf_mode": "off"`. A case that runs out of memory has no
FLOPs to report, so nothing is lost by running without counters. The first attempt
is kept in the sweep's `_superseded/` directory and will be included in the Zenodo
archive.

In every other case, `perf` was on and the outcome was classified directly.

## Regenerating the table

From the repo root:

```bash
julia --project bench/make_table.jl --out data/vista_2026-11 --ci --flops --delta
julia --project bench/export_raw.jl --out data/vista_2026-11
```

Both read only the committed JSON files. They reproduce `table.csv`, `table.tex`
and the three `export/` files exactly.
