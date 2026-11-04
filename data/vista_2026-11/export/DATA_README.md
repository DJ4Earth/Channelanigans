# Raw benchmark data: how it was collected

## What is measured

Each *case* is one (configuration, N) pair: one row of the cost table (`config`,
see `bench/configs.toml`) at one number of time steps `N` in the differentiated
window. The differentiated function runs the reentrant-channel model
(`src/ChannelModel.jl`: 80×160×32 `ImmersedBoundaryGrid`, Float64,
`HydrostaticFreeSurfaceModel`) for N steps of Δt = 2.5 min from a fixed, seeded
initial condition and returns the zonal transport through the mid-channel section
(Sv). One *call* is one evaluation of the reverse-mode gradient of that objective
(`Enzyme.autodiff(ReverseWithPrimal, …)`) with respect to the initial T/S fields,
surface forcing fields and layer thicknesses:

- `driver = reactant`: the gradient function compiled once with Reactant (`@compile`,
  options per configuration: `optimize`, `mincut`, `checkpointing`); calls execute
  the compiled XLA program on the CPU.
- `driver = enzyme`: plain Julia + Enzyme on the CPU, no Reactant (optionally with
  Checkpointing.jl). The first call includes JIT compilation.

Every call starts from a freshly built model and freshly zeroed adjoint (shadow)
storage, so all calls of a case compute exactly the same thing; the objective of
every call is recorded (`objective_sv`) and checked to be identical.

## Execution environment

- One fresh Julia process per case (`bench/run_cases.jl` → `bench/driver_*.jl`),
  inside the archived container image; `-O0`, `JULIA_NUM_THREADS=1`,
  `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_backend_extra_options=xla_cpu_use_multi_output_fusion"`.
- The process is pinned to a single core with `numactl -C <core>` (column
  `numactl` in cases.csv). XLA's CPU runtime may still use several threads; with
  pinning they share that one core.
- Exact software versions, Julia flags, host, CPU and git commit are recorded per
  case (cases.csv; full detail in each `result.json` → `environment`).

## Per-case protocol

1. (Reactant only) compile the gradient function; compile time → `compile_s`.
2. Warmup call (`kind = warmup`, `call_index = 0`): not used for statistics. For the
   Enzyme driver it includes JIT compilation. Its objective and gradient fields are
   the reference used for cross-configuration correctness checks.
3. 12 measured calls (`kind = measured`, `call_index = 1…12`).

Before every call (outside the timed window): build a new problem, `GC.gc()` twice,
`malloc_trim(0)` (return freed heap to the OS, so the pre-call memory does not depend
on the previous call), then reset the memory high-water mark (below).

## Per-call fields (raw_runs.csv)

- `wall_s`: wall-clock time of the call, `time_ns()` immediately before and after.
  Reactant programs are compiled with `sync=true`, so the call returns only after
  the computation has finished.
- `base_rss_bytes`, `peak_rss_bytes`, `delta_rss_bytes` (Linux, `memory_method =
  linux_vmhwm`): writing `5` to `/proc/self/clear_refs` resets the process high-water
  mark `VmHWM` to the current resident set size; that value is `base_rss_bytes`.
  `peak_rss_bytes` is `VmHWM` read right after the call, i.e. the maximum resident
  set size of the whole process during the call (model, compiled code and all
  temporaries included); `delta_rss_bytes = peak − base`. `/proc` reports KiB; values
  are stored in bytes (KiB × 1024). (`memory_method = ps_sampler_*` marks proofing
  runs on macOS, where RSS was sampled every 50 ms with `ps`; not used for the paper.)
- `flops`, and one column per perf event: hardware floating-point operation counts
  from `perf stat` on NVIDIA Grace (Neoverse V2): `r80C0` = `FP_SCALE_OPS_SPEC`
  (SVE, lane-scaled) and `r80C1` = `FP_FIXED_OPS_SPEC` (scalar and NEON);
  `flops = r80C0 + r80C1`. A fused multiply-add counts as 2. The events count
  speculatively executed operations (≈1–2 % overcount) and do not distinguish
  precision. perf counts all threads of the benchmark process.
  Counting is enabled only inside each measured call: the driver sends
  `enable`/`disable` to perf through `--control` FIFOs immediately around the call
  (`window_start_unix`, `window_end_unix` bracket the counted window). perf runs in
  interval mode (`-I 100`, 100 ms): each interval reports the count accumulated in
  it, or `<not counted>` if counting was disabled throughout. Because the next
  problem is built with counting disabled (seconds, i.e. many intervals), each
  maximal run of consecutive counted intervals is exactly one measured call; the
  call's count is the sum over that run. The number of runs must equal the number
  of measured calls (`perf_attribution` in cases.csv), and the per-call counts sum to
  the run total by construction. The raw interval output is kept in each case's
  `perf.csv`. Warmup calls are never counted (`flops` empty).
- `objective_sv`: the objective value returned by that call.

## Cases that did not complete

`case_status`: `ok`, or `m` (out of memory: an OOM message in the log or the process
was killed with SIGKILL), `t` (exceeded the per-case wall-clock limit), `s`
(segmentation fault), `e` (other error); `case_reason` gives the detail. A case that
was cut off still contributes the calls it completed (from `progress.json`, written
after every call).

## Summary statistics (cases.csv)

Over the measured calls of a case: mean, sample standard deviation (n − 1), and a
two-sided 95 % Student-t confidence interval for the mean,
mean ± t₀.₉₇₅,ₙ₋₁ · s / √n (n = 12: t = 2.201). Columns `*_mean`, `*_std`,
`*_ci95_halfwidth`, `n_measured`.

## Correctness

Within every case all calls return the identical objective. Across configurations
(`bench/check_gradients.jl`) the objective and all gradient fields of the warmup
calls agree to within 1e-10 relative (observed ≤ 1e-14 among Reactant
configurations and ≤ 8e-13 between Reactant and Enzyme-only).
