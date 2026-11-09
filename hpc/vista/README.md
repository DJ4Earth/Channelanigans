# Running the CPU AD cost table on TACC Vista

Each table case (config × N) runs as its own SLURM job on a whole `gg` node
(NVIDIA Grace), inside the frozen Apptainer image. The settings (allocation, SIF
path, image digest, result locations) live in [`env.sh`](env.sh).

Run every command from the repo root on a login node. `report.sh` uses the
container, and Vista only allows Apptainer on compute nodes, so submit it as a
short job with `submit.sh --report status` (or `--report final`); its output lands
in `$WORK/channelanigans_results/slurm_logs/ch-report-<mode>.<jobid>.out`. Inside an
`idev` session you can also run `hpc/vista/report.sh status` directly.

## Image profiles

Two container images can be run, selected with `--profile` on every `submit.sh`
call. The image settings are in [`env.sh`](env.sh).

| Profile | Image | Results in `$WORK/channelanigans_results/` | Table rows |
|---|---|---|---|
| `current` (default) | current package versions (Julia 1.12.7, Reactant 0.2.304) | `table/` | all 9 in `bench/configs.toml` |
| `aug2026` | the July/August 2026 environment of the original runs ([`envs/aug2026`](../../envs/aug2026/)) | `table_aug2026/` | the original 8 (no Revolve) |
| `jll443` | current versions, except Reactant 0.2.308 (Reactant_jll 0.0.443) and Enzyme 0.13.219 ([`envs/jll443`](../../envs/jll443/)); runs in batches | `table_jll443_b1/`, `_b2/`, … pooled into `table_jll443/` | all 9 |

Each profile gets its own job names (`ch-…` and `ch-aug-…`), so the two sweeps never
skip or collide with each other's jobs. For `aug2026`, the walltimes for every case
are in [`walltimes_aug2026_seed.tsv`](walltimes_aug2026_seed.tsv), seeded from the
original runs' per-call times, so the sweep can be submitted in one go:

```bash
hpc/vista/submit.sh --profile aug2026 --gate
hpc/vista/submit.sh --profile aug2026 --walltimes hpc/vista/walltimes_aug2026_seed.tsv
hpc/vista/submit.sh --profile aug2026 --report status    # predictions -> walltimes_aug2026.tsv
```

## Batched sweeps (profile `jll443`)

A profile with `BATCH_REPEATS` (in `env.sh`) runs every case as independent
*batches*: separate jobs, usually on different nodes, each with 1 warmup and
`BATCH_REPEATS` (4) measured calls. Batch K lands in `table_jll443_bK/`. Any number
of finished batches can be pooled at any time; the pool's confidence intervals then
include node-to-node spread.

TACC allows only ~40–50 queued + running jobs per user, fewer than the 63 cases, so
`fill.sh` reruns `submit.sh` every 10 min to keep the queue topped up to
`--max-jobs` (default 40). Every batch-1 case is submitted before any batch 2, and
each batch goes longest case first. A case whose batch 1 ended other than `ok`
(e.g. `m`) gets no later batches. Walltimes come from the 2026-11 timings
([`walltimes_jll443_seed.tsv`](walltimes_jll443_seed.tsv)), so there are no waves.

```bash
hpc/vista/submit.sh --profile jll443 --gate                 # may run alongside batch 1
tmux new -s fill                                            # reattach on the same login node
hpc/vista/fill.sh --profile jll443 --batches 1,2,3 \
    --walltimes hpc/vista/walltimes_jll443_seed.tsv [--max-jobs 40]
hpc/vista/submit.sh --profile jll443 --report status        # every batch
hpc/vista/submit.sh --profile jll443 --report final         # pool finished batches + table
```

`fill.sh` never resubmits a case that already has a result (it passes `--new-only`);
rerun one by hand with `submit.sh --profile jll443 --batches K --configs C --nsteps N
--walltime HH:MM:SS`. `report final` can be run as often as you like: it pools whatever
has finished (`bench/merge_batches.jl`), and adds `export/batches.csv` with each batch's
node and means.

## 1. Gate (one job, about 1 h)

```bash
hpc/vista/submit.sh --gate
```

The gate checks the node, the FLOP counters (one matmul with a known count, 2·10⁹)
and the full measurement pipeline (`reactant_full` and `enzyme_naive` at N = 4). The
log ends with `GATE PASSED`. The files land in `$WORK/channelanigans_results/gate/<jobid>/`.

## 2. Wave 1: N = 4 and 8 for every config (18 jobs)

```bash
hpc/vista/submit.sh --nsteps 4,8 --walltime 04:00:00
```

These are final table entries. The planner also uses them to predict walltimes
for larger N.

## 3. Later waves

```bash
hpc/vista/submit.sh --report status   # status of every case; writes hpc/vista/walltimes.tsv
hpc/vista/submit.sh --nsteps 16 --walltimes hpc/vista/walltimes.tsv
# ... then repeat status → submit for 32,64 and 128,256
```

The predictions use the two largest finished N for each config, so going one or
two N values at a time keeps them accurate. Cases are capped at 48 h. A case that
runs out of time or memory is recorded as `t` or `m`; both are legitimate table
entries.

## 4. Finalize

```bash
hpc/vista/submit.sh --report final
```

This builds `table.tex` and `table.csv`, cross-checks gradients across configs,
and writes the raw-data export (`export/raw_runs.csv`, `cases.csv`,
`DATA_README.md`, `raw_data.tar.gz`) into `$WORK/channelanigans_results/table/`.

## Notes

- **Resubmitting is safe.** Cases that are already `ok` in `$WORK`, or that are
  still queued, are skipped. A failed earlier attempt is moved to `_superseded/`,
  never deleted. Pass `--force` to rerun cases that are already ok.
- **Results are copied to `$WORK`.** Jobs write to `$SCRATCH` and copy each finished
  case to `$WORK`. The orchestrator stops 30 min before the walltime, so the copy
  happens even when a case times out.
- **Every case records its provenance.** Each case directory gets a
  `job_info.json` with the job ID, node, image digest, Apptainer version and the
  commit of these scripts. The benchmark code's own commit is in `result.json`.
