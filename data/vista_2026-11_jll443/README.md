# CPU AD cost benchmarks: TACC Vista, November 2026, Reactant 0.2.308

This folder holds a rerun of the CPU AD cost table in
[`../vista_2026-11/`](../vista_2026-11/) with a newer Reactant. It measures the cost
of one reverse-mode gradient of the channel model for 9 AD configurations at
N = 4, 8, 16, 32, 64, 128 and 256 timesteps. Only Reactant, its XLA/MLIR binary
(`Reactant_jll`) and Enzyme changed; the model, the configurations, the hardware
and the measurement are the same.

> **Status (2026-11-09): complete.** All 63 cases are in: 60 finished (`ok`) and 3
> ran out of the node's memory (`m`), the same three as in 2026-11
> (`reactant_mincut_off` at N = 128 and 256, `enzyme_naive` at N = 256). Every
> finished case has 12 measured calls from 3 batches on different nodes.

## What changed from `vista_2026-11`

| Package | `vista_2026-11` | this folder |
|---|---|---|
| Reactant | 0.2.304 | 0.2.308 |
| Reactant_jll (XLA/MLIR binary) | 0.0.438 | 0.0.443 |
| Enzyme | 0.13.216 | 0.13.219 |
| EnzymeCore | 0.8.26 | 0.8.27 |
| Enzyme_jll | 0.0.303 | 0.0.307 |
| CRC32c (stdlib) | – | 1.11.0 (new dependency of Reactant) |

The other 159 packages, including Julia 1.12.7, Oceananigans 0.116.1 and
Checkpointing.jl 0.13.0, are identical. The pinned environment is
[`envs/jll443`](../../envs/jll443/); its Manifest started from the 2026-11 image's
Manifest and was re-resolved only as far as these bumps required.

## Main result

Every Reactant configuration is 2–4× faster than in 2026-11, and almost all of
the gain is fewer executed floating-point operations. The Enzyme-only
configurations are unchanged, which also shows that the nodes and the measurement
behave as before. Ratios are this folder ÷ `vista_2026-11`, over N = 8 to 256:

| Configuration | Runtime | FLOPs | Peak memory |
|---|---|---|---|
| `reactant_full` | 0.36–0.41 | 0.38–0.40 | 0.5 GiB lower |
| `reactant_after_enzyme` | 0.37–0.38 | 0.40 | 0.2 GiB lower |
| `reactant_mincut_off` | 0.40–0.57 | 0.32–0.34 | same |
| `reactant_periodic` | 0.33–0.36 | 0.48–0.51 | 1.1 GiB lower |
| `reactant_periodic_mincut_off` | 0.25–0.29 | 0.25–0.26 | same |
| `reactant_binomial_10` | 0.45–0.50 | 0.40–0.45 | 0.5 GiB lower |
| `reactant_binomial_log2` | 0.47–0.50 | 0.44–0.47 | 0.4 GiB lower |
| `enzyme_naive` | 0.99–1.03 | 1.00 | same |
| `enzyme_revolve_log2` | 0.98–1.01 | 1.00 | same |

- **Reactant is now faster than Enzyme alone.** `reactant_full` takes 0.67–0.72× the
  time of `enzyme_naive` per call (it took 1.13–1.85× in 2026-11), and executes fewer
  FLOPs: 481 against 601 GFLOP at N = 128 (it was 1251 against 601).
- **With checkpointing,** `reactant_binomial_log2` is 5–6× faster than
  `enzyme_revolve_log2` (204 s against 1095 s at N = 128), at about 1.5× the memory
  (8.7 against 6.0 GiB).
- **Memory** is essentially unchanged, so the memory comparison of 2026-11 still
  holds.
- **N = 4** improved less (runtime ratio about 0.6, FLOP ratio 0.58), because at
  N = 4 Reactant was already close to Enzyme.

## How the data was produced

- **Hardware:** one TACC Vista `gg` node per batch (2× NVIDIA Grace, Neoverse V2,
  144 cores, 237 GiB). The benchmark process is pinned to a single core
  (`numactl -C 0`).
- **Software:** the frozen container image
  `ghcr.io/jlk9/channelanigans-artifact:2026-11-jll443-cpu`, digest
  `sha256:bc5ed0f2d08f3c234360fbd31ff66924ddba22d9231a9fe33c8187412b688d6b`
  (built from repo commit `3f6c63d`), run with `julia -O0` and
  `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_backend_extra_options=xla_cpu_use_multi_output_fusion"`,
  as in 2026-11.
- **Protocol: batches.** Each case ran as 3 independent *batches*: separate SLURM
  jobs, usually on different nodes, each a fresh process with its own compilation,
  1 warmup call (excluded from the statistics) and 4 measured calls. The 12
  measured calls of the 3 batches are pooled, and the table reports mean ± 95%
  Student-t confidence interval over the pool. In 2026-11 the 12 calls came from
  one process on one node.
  - In all, 183 batch jobs ran on 46 different nodes. Each batch's node, job and
    means are in `export/batches.csv`.
  - Because the batches ran on different nodes, the confidence intervals include
    node-to-node variation. This is intended: it reflects what a user of the
    machine would see.
  - A case whose first batch ran out of memory got no further batches.
- **Runtime, memory, FLOPs:** measured as in 2026-11: wall-clock time of one call;
  peak resident memory during the call (Linux `VmHWM`, reset before each call);
  hardware counters `r80C0` + `r80C1` counted only inside each measured call.

The full methodology, including every field in the CSV files and a section on the
batches, is in [`export/DATA_README.md`](export/DATA_README.md). The SLURM scripts
are in [`hpc/vista/`](../../hpc/vista/) (profile `jll443` in `env.sh`; `fill.sh`
kept the queue filled within TACC's per-user job limit).

## Files

| Path | Contents |
|---|---|
| `table.tex`, `table.csv` | The table: FLOPs, runtime, memory and Δmemory per config and N, with CI half-widths |
| `export/raw_runs.csv` | One row per call (warmup + measured) of every batch, with its `batch` and `hostname` |
| `export/cases.csv` | One row per case: outcome, pooled statistics, `n_batches`, versions, flags |
| `export/batches.csv` | One row per batch: outcome, node, SLURM job, per-batch means |
| `export/DATA_README.md` | How each field is collected |
| `<config>/N<nnnn>/result.json` | Pooled record of a case: every call (tagged with its batch and node), statistics, gradient summaries, environment, `batches` |
| `<config>/N<nnnn>/status.json` | Outcome (`ok` / `t` / `m` / `s` / `e`) and the batches behind it |
| `batches/table_jll443_b<K>/<config>/N<nnnn>/` | Batch K as it ran: `result.json`, `status.json`, `progress.json`, `job_info.json` (SLURM job, node, image digest, Apptainer version, script commit), `log.txt` |

**Not in this repository** because of size (to be archived on Zenodo; the
complete archive, `raw_data.tar.gz`, is on Vista in
`$WORK/channelanigans_results/table_jll443/`):
- `perf.csv`: the raw 100 ms hardware-counter intervals behind the per-call FLOPs.
- `gradients.jld2`: the full gradient fields used for the cross-configuration
  check.

## Correctness

- **Within each case,** every measured call of every batch returned the identical
  objective, and the gradient norms of every field are identical across batches
  (`gradients_max_rel_spread_across_batches` is 0 for every case).
- **Across configurations,** every configuration agrees with `reactant_full` at
  every N. From the committed gradient summaries: Reactant configurations within
  1.5e-13, Enzyme-only configurations within 7.8e-13 (objective and gradient
  norms). The full-field check (`bench/check_gradients.jl`) needs the
  `gradients.jld2` files.
- **Against `vista_2026-11`:** the Enzyme-only configurations reproduce 2026-11
  bit for bit (objective and gradient summaries). The Reactant configurations agree
  with 2026-11 within 1.3e-14 in the objective and 8.2e-14 in the gradient
  summaries. The new Reactant computes the same gradient with fewer operations.

## Reading the table

- **Out of memory:** the same three cases as in 2026-11.
  - `reactant_mincut_off` at N = 128 and 256: XLA reports it directly
    (`Out of memory`).
  - `enzyme_naive` at N = 256: killed by the kernel's out-of-memory killer. This
    case ran without hardware counters from the start (`"perf_mode": "off"` in its
    `job_info.json`), because under `perf` the kill is reported as a clean exit; see
    [`../vista_2026-11/README.md`](../vista_2026-11/README.md).
- **Node-to-node spread:** for most cases the batch means agree within a few
  percent (median spread between the fastest and slowest batch: 1.2%). Two cases
  stand out, both `reactant_mincut_off`, the configuration with by far the largest
  memory footprint:
  - N = 8: one batch averaged 20.5 s per call, the other two 12.1–12.2 s;
  - N = 32: one batch averaged 73.0 s, the other two 48.2–49.3 s.

  In both, the slow batch was consistently slow on all its calls and executed the
  same number of FLOPs, so the node, not the computation, was slower. Their wide
  confidence intervals (±2.62 s and ±7.59 s) are this spread.
- **N = 4:** several Reactant configurations use more memory at N = 4 than at N = 8,
  as in 2026-11; it is a property of the compiled program at very short loops.

## Regenerating

From the repo root. Pooling the batches (writes
`batches/table_jll443/`, which reproduces the pooled `<config>/N<nnnn>/` files
here exactly):

```bash
julia --project bench/merge_batches.jl --root data/vista_2026-11_jll443/batches --sweep table_jll443
```

The table and the export, from the pooled files:

```bash
julia --project bench/make_table.jl --out data/vista_2026-11_jll443 --ci --flops --delta
julia --project bench/export_raw.jl --out data/vista_2026-11_jll443
```

These read only the committed JSON files and reproduce `table.csv`, the
`export/` files and `table.tex` exactly (apart from the source path in the first
line of `table.tex`).
