"""
    Measure

Measurement plumbing shared by both benchmark drivers, so every table row is
measured identically:

- **Runtime**: wall clock (`time_ns`) around one call. Reactant thunks are compiled
  with `sync=true`, so the call returns only after the computation finishes.
- **Peak memory**:
  - Linux: `/proc/self/clear_refs` ← "5" resets VmHWM to the *current* RSS, then
    VmHWM after the call is the in-call high-water mark (exact). Both the high-water
    mark and the delta over the pre-call RSS are reported.
  - elsewhere (macOS proofing): an external `ps` sampler polls this process's RSS
    every `sample_interval` seconds during the call (approximate; may miss very
    short transients).
- **FLOPs**: optional. If the process runs under
  `perf stat --control fifo:\$PERF_CTL_FIFO,\$PERF_ACK_FIFO -D -1 ...`, counters are
  enabled only inside each measured call. perf itself writes the totals; the
  orchestrator divides by the number of measured calls.

FIFOs are opened read-write (`O_RDWR` never blocks), so nothing hangs when perf is
absent or not yet attached.
"""
module Measure

using Printf
using InteractiveUtils: versioninfo
import Pkg

export measured_call, environment_info, write_json_atomic

#####
##### perf gating
#####

const PERF_CTL = Ref{Union{Nothing, IO}}(nothing)
const PERF_ACK = Ref{Union{Nothing, IO}}(nothing)

function __init__()
    if haskey(ENV, "PERF_CTL_FIFO") && haskey(ENV, "PERF_ACK_FIFO")
        PERF_CTL[] = open(ENV["PERF_CTL_FIFO"], read=true, write=true)
        PERF_ACK[] = open(ENV["PERF_ACK_FIFO"], read=true, write=true)
    end
end

perf_active() = PERF_CTL[] !== nothing

function perf_command(cmd)
    perf_active() || return nothing
    println(PERF_CTL[], cmd)
    flush(PERF_CTL[])
    readline(PERF_ACK[])   # perf replies "ack"
    return nothing
end

perf_enable()  = perf_command("enable")
perf_disable() = perf_command("disable")

#####
##### Memory
#####

const HAS_PROC = Sys.islinux() && isfile("/proc/self/status")

function vm_kib(field)
    status = read("/proc/self/status", String)
    m = match(Regex("$(field):\\s+(\\d+) kB"), status)
    return parse(Int, m.captures[1])   # KiB despite the "kB" label
end

"Start an external RSS sampler for this process; returns (process, outfile)."
function start_rss_sampler(interval)
    out = tempname()
    pid = getpid()
    script = """
        max=0
        while kill -0 $pid 2>/dev/null; do
            r=\$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
            [ -n "\$r" ] && [ "\$r" -gt "\$max" ] && max=\$r && echo \$max > $out
            sleep $interval
        done
        """
    proc = run(`sh -c $script`; wait=false)
    sleep(2interval)   # let it take a baseline sample
    return proc, out
end

function stop_rss_sampler(proc, out)
    kill(proc)
    wait(proc)
    kib = isfile(out) ? parse(Int, strip(read(out, String))) : -1
    rm(out; force=true)
    return kib
end

# Return freed heap memory to the OS (glibc only). Without this, memory freed by a
# previous call stays resident, so the pre-call RSS (and thus the high-water mark and
# its delta) depends on call history; observed for Enzyme-only runs (base RSS swinging
# 3.5 → 11.8 GiB between consecutive identical calls).
function trim_heap()
    Sys.islinux() || return nothing
    try
        ccall(:malloc_trim, Cint, (Csize_t,), 0)
    catch
    end
    return nothing
end

current_rss_kib() = HAS_PROC ? vm_kib("VmRSS") :
    parse(Int, strip(read(`ps -o rss= -p $(getpid())`, String)))

#####
##### One measured call
#####

"""
    measured_call(f; perf=true, sample_interval=0.05)

Run `f()` once with GC'd heap, reset memory high-water mark, and (if `perf`)
perf counters enabled only around the call. Returns `(value, stats)` where `stats` is a
NamedTuple with `wall_s`, `peak_rss_bytes`, `base_rss_bytes`, `delta_rss_bytes`,
`memory_method`.
"""
function measured_call(f; perf = true, sample_interval = 0.05)
    GC.gc(); GC.gc()
    trim_heap()

    sampler = nothing
    if HAS_PROC
        write("/proc/self/clear_refs", "5")   # VmHWM := current RSS
        base_kib = vm_kib("VmHWM")
        method = "linux_vmhwm"
    else
        base_kib = current_rss_kib()
        sampler = start_rss_sampler(sample_interval)
        method = "ps_sampler_$(sample_interval)s"
    end

    perf && perf_enable()
    window_start = time()        # Unix time; brackets the perf-enabled window
    t0 = time_ns()
    y = f()
    t1 = time_ns()
    window_end = time()
    perf && perf_disable()

    peak_kib = HAS_PROC ? vm_kib("VmHWM") : max(stop_rss_sampler(sampler...), current_rss_kib())

    stats = (wall_s          = (t1 - t0) / 1e9,
             peak_rss_bytes  = peak_kib * 1024,
             base_rss_bytes  = base_kib * 1024,
             delta_rss_bytes = (peak_kib - base_kib) * 1024,
             memory_method   = method,
             perf_counted    = perf && perf_active(),
             window_start_unix = window_start,
             window_end_unix   = window_end)

    @info @sprintf("measured call: wall %.3f s, peak RSS %.3f GiB (delta %.3f GiB) [%s]",
                   stats.wall_s, stats.peak_rss_bytes / 2^30, stats.delta_rss_bytes / 2^30, method)
    return y, stats
end

#####
##### Provenance
#####

function environment_info()
    pkgs = Dict{String, Any}()
    for (_, dep) in Pkg.dependencies()
        dep.is_direct_dep || dep.name in ("Reactant_jll", "Enzyme_jll", "ReactantCore", "EnzymeCore") || continue
        pkgs[dep.name] = Dict("version"  => string(dep.version),
                              "git_revision" => something(dep.git_revision, ""),
                              "tree_hash" => something(dep.tree_hash, ""))
    end
    vinfo = sprint(versioninfo)
    env_keys = ("XLA_FLAGS", "JULIA_NUM_THREADS", "OMP_NUM_THREADS", "SLURM_JOB_ID",
                "SLURM_JOB_PARTITION", "PERF_CTL_FIFO", "BENCH_JULIA_FLAGS")
    return Dict(
        "julia_version" => string(VERSION),
        "julia_cmd"     => string(Base.julia_cmd()),
        "julia_opt_level" => Base.JLOptions().opt_level,
        "threads"       => Threads.nthreads(),
        "hostname"      => gethostname(),
        "os"            => string(Sys.KERNEL),
        "arch"          => string(Sys.ARCH),
        "cpu"           => Sys.cpu_info()[1].model,
        "ncores"        => Sys.CPU_THREADS,
        "total_memory_bytes" => Sys.total_memory(),
        "perf_active"   => perf_active(),
        "env"           => Dict(k => get(ENV, k, nothing) for k in env_keys),
        "packages"      => pkgs,
        "versioninfo"   => vinfo,
        "git_commit"    => git_commit(),
    )
end

function git_commit()
    # Container images carry no .git; the build records the commit instead.
    c = get(ENV, "CHANNELANIGANS_GIT_COMMIT", "")
    isempty(c) || return c
    try
        root = dirname(@__DIR__)
        sha = strip(read(`git -C $root rev-parse HEAD`, String))
        dirty = !isempty(strip(read(`git -C $root status --porcelain -- src bench Project.toml Manifest.toml`, String)))
        return dirty ? sha * "-dirty" : sha
    catch
        return "unknown"
    end
end

"Write `str` to `path` atomically (write to temp file, then rename)."
function write_json_atomic(path, str)
    mkpath(dirname(path))
    tmp = path * ".tmp"
    write(tmp, str)
    mv(tmp, path; force=true)
    return path
end

end # module
