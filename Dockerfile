# Frozen environment for the CPU AD cost benchmarks (Oceananigans + Enzyme ± Reactant).
#
# Build (from the repo root):
#   docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t channelanigans-artifact .
#
# Run (outputs land in ./results on the host):
#   docker run --rm -v "$PWD/results:/work/results" channelanigans-artifact \
#       julia /opt/channelanigans/bench/run_cases.jl --nsteps 4,8
#
# The project lives read-only in /opt/channelanigans with a pre-populated, precompiled
# depot in /opt/julia_depot; the working directory /work is where results are written.
# This layout also works under Apptainer/Singularity (read-only image).

# Official Julia image (Debian 13 "trixie"), pinned by multi-arch index digest.
FROM julia:1.12.7@sha256:a1fbd9d099a0517bdb70e7471955f09dc92beb822e7a5a74938f7ec8d5682693

ARG TARGETARCH
ARG GIT_COMMIT=unknown

# numactl: pin the benchmark to one core; procps: `ps` (RSS sampler fallback);
# linux-perf: FLOP counting on hosts that expose a PMU (Vista Grace nodes).
RUN apt-get update \
 && apt-get install -y --no-install-recommends numactl procps linux-perf \
 && rm -rf /var/lib/apt/lists/*

# Precompile for multiple CPU targets so the image is fast on NVIDIA Grace
# (neoverse-v2) and still runs on any other CPU of the same architecture.
ENV JULIA_DEPOT_PATH=/opt/julia_depot \
    JULIA_PROJECT=/opt/channelanigans \
    JULIA_NUM_THREADS=1 \
    JULIA_PKG_PRECOMPILE_AUTO=0
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        echo 'generic;neoverse-v2,clone_all' > /opt/julia_cpu_target; \
    else \
        echo 'generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)' > /opt/julia_cpu_target; \
    fi

WORKDIR /opt/channelanigans

# 1. Dependencies (cached layer): resolve from Project.toml's exact [compat] pins,
#    force Reactant's CPU build, download and precompile everything.
COPY Project.toml ./
COPY docker/LocalPreferences.toml ./LocalPreferences.toml
RUN export JULIA_CPU_TARGET="$(cat /opt/julia_cpu_target)" \
 && julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); Pkg.status()'

# 2. Benchmark code.
COPY src ./src
COPY bench ./bench

# 3. Smoke test at build time: exercises Reactant compilation and its CUDA-extension
#    kernel path (pulling any lazily downloaded artifacts into the image) and checks
#    Reactant vs CPU forward consistency.
RUN export JULIA_CPU_TARGET="$(cat /opt/julia_cpu_target)" \
 && mkdir -p /tmp/smoke && cd /tmp/smoke \
 && XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_backend_extra_options=xla_cpu_use_multi_output_fusion" \
    julia -O0 /opt/channelanigans/bench/forward_consistency.jl 1 --reactant-init 2>&1 | tee /opt/channelanigans/BUILD_SMOKE.txt \
 && grep -q "Reactant  drifting fluid cells      0" /opt/channelanigans/BUILD_SMOKE.txt \
 && rm -rf /tmp/smoke

# 4. Provenance: exact resolved versions (the image's Manifest.toml is the frozen set).
RUN { echo "git_commit: ${GIT_COMMIT}"; echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
      echo "arch: ${TARGETARCH}  JULIA_CPU_TARGET: $(cat /opt/julia_cpu_target)"; echo; \
      julia -e 'using InteractiveUtils; versioninfo()'; echo; \
      julia -e 'using Pkg; Pkg.status(mode=Pkg.PKGMODE_MANIFEST)'; } > /opt/channelanigans/BUILD_INFO.txt 2>&1 \
 && chmod -R a+rX /opt/julia_depot /opt/channelanigans

# Runtime: no network access needed; a writable depot is stacked in front of the
# frozen one so Julia can write logs/caches even when the image is read-only.
ENV JULIA_PKG_OFFLINE=true \
    JULIA_DEPOT_PATH=/tmp/julia_depot:/opt/julia_depot \
    CHANNELANIGANS_GIT_COMMIT=${GIT_COMMIT} \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_backend_extra_options=xla_cpu_use_multi_output_fusion"

WORKDIR /work
CMD ["julia", "/opt/channelanigans/bench/run_cases.jl", "--help"]
