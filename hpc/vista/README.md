# Running the CPU AD cost table on TACC Vista

Each table case (config × N) runs as its own SLURM job on a whole `gg` node
(NVIDIA Grace), inside the frozen Apptainer image. The settings (allocation, SIF
path, image digest, result locations) live in [`env.sh`](env.sh).

Run every command from the repo root. Run `submit.sh` on a login node; it doesn't
use the container. `report.sh` does use the container, and Vista only allows
Apptainer on compute nodes, so run it inside a short interactive session:
`idev -p gg -N 1 -n 1 -t 00:30:00`.

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
hpc/vista/report.sh status      # status of every case; writes hpc/vista/walltimes.tsv
hpc/vista/submit.sh --nsteps 16 --walltimes hpc/vista/walltimes.tsv
# ... then repeat status → submit for 32,64 and 128,256
```

The predictions use the two largest finished N for each config, so going one or
two N values at a time keeps them accurate. Cases are capped at 48 h. A case that
runs out of time or memory is recorded as `t` or `m`; both are legitimate table
entries.

## 4. Finalize

```bash
hpc/vista/report.sh final
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
