# Orchestrator: run benchmark cases, one fresh Julia process per (config, N).
#
#   julia --project bench/run_cases.jl [options]
#
# For each case it
#   * builds the driver command (optionally wrapped in `numactl -C <core>` and
#     `perf stat --control` FIFO gating for FLOP counting),
#   * enforces a wall-clock timeout (kills the whole process group),
#   * classifies the outcome:  ok | t (timeout) | m (out of memory) |
#     s (segfault) | e (other error),
#   * writes <out>/<config>/N<nnnn>/{log.txt, status.json[, perf.csv]}
#     and attaches per-call perf counts / FLOPs to each measured call in result.json
#     (or progress.json for a case that was cut off).
#
# This process only loads TOML/JSON; all heavy lifting happens in the children.

using TOML
using JSON
using Printf
using Dates
include(joinpath(@__DIR__, "stats.jl"))

const REPO_ROOT = dirname(@__DIR__)

const DEFAULT_XLA_FLAGS = "--xla_cpu_multi_thread_eigen=false --xla_backend_extra_options=xla_cpu_use_multi_output_fusion"
const NEOVERSE_V2_FP_EVENTS = "r80C0,r80C1"   # FP_SCALE_OPS_SPEC, FP_FIXED_OPS_SPEC

const USAGE = """
usage: julia --project bench/run_cases.jl [options] [-- driver options...]

  --configs LIST       comma-separated config names, or "table" (default: table rows)
  --nsteps LIST        comma-separated timestep counts (default: table nsteps)
  --out DIR            results root (default: results)
  --timeout S          per-case wall-clock limit in seconds, incl. compile and all
                       13 calls (default 172800 = 48 h; size it to the case)
  --julia-flags STR    flags for the child julia (default: "-O0", which cuts Reactant
                       compile time substantially; pass "" to use Julia's default -O2)
  --threads N          JULIA_NUM_THREADS for the child (default 1)
  --xla-flags STR      XLA_FLAGS for the child (default: "$DEFAULT_XLA_FLAGS")
  --perf MODE          auto | on | off   (auto: on if a probe `perf stat` with the events works)
  --perf-events LIST   perf events to count (default $NEOVERSE_V2_FP_EVENTS, Neoverse V2 FP ops;
                       FLOPs = sum of the events)
  --perf-interval-ms M perf -I interval used for per-call attribution (default 100)
  --numactl MODE       auto | on | off   (pin child to one core with `numactl -C <core>`)
  --core K             core for numactl (default 0)
  --skip-existing      skip cases whose status.json already says ok
  --configs-file PATH  alternative configs TOML
  --dry-run            print the commands only

Everything after a literal `--` is passed to each driver (e.g. --save-gradients,
--repeats 12 [default], --warmup 1 [default], --repeat-budget S for smoke runs).
"""

function parse_args(args)
    o = Dict{String, Any}("configs" => "table", "nsteps" => nothing, "out" => "results",
        "timeout" => 172800.0, "julia-flags" => "-O0", "threads" => 1, "xla-flags" => DEFAULT_XLA_FLAGS,
        "perf" => "auto", "perf-events" => NEOVERSE_V2_FP_EVENTS, "perf-interval-ms" => 100,
        "numactl" => "auto", "core" => 0,
        "skip-existing" => false, "dry-run" => false,
        "configs-file" => joinpath(REPO_ROOT, "bench", "configs.toml"), "driver-args" => String[])
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--"
            o["driver-args"] = collect(args[i+1:end]); break
        elseif a in ("-h", "--help")
            print(USAGE); exit(0)
        elseif a in ("--skip-existing", "--dry-run")
            o[a[3:end]] = true
        elseif startswith(a, "--") && i < length(args)
            o[a[3:end]] = args[i+1]; i += 1
        else
            error("unrecognized argument $a\n\n$USAGE")
        end
        i += 1
    end
    o["timeout"] isa AbstractString && (o["timeout"] = parse(Float64, o["timeout"]))
    o["threads"] isa AbstractString && (o["threads"] = parse(Int, o["threads"]))
    o["core"]    isa AbstractString && (o["core"]    = parse(Int, o["core"]))
    o["perf-interval-ms"] isa AbstractString && (o["perf-interval-ms"] = parse(Int, o["perf-interval-ms"]))
    return o
end

have(cmd) = Sys.which(cmd) !== nothing

"True if `cmd` runs successfully (quietly). Used to probe perf/numactl actually work
here, e.g. perf is installed in the image but a VM (Docker Desktop) exposes no PMU."
works(cmd) = try
    success(pipeline(cmd; stdout = devnull, stderr = devnull))
catch
    false
end

perf_works(events) = have("perf") && Sys.islinux() &&
    works(`perf stat -x, -o /dev/null -e $events -- true`)
numactl_works(core) = have("numactl") && works(`numactl -C $core true`)

function resolve_mode(mode, available, what)
    mode == "off" && return false
    mode == "on"  && (available || error("$what requested but not available"); return true)
    return available
end

#####
##### Outcome classification
#####

const OOM_PATTERNS = [r"Out of memory"i, r"RESOURCE_EXHAUSTED", r"OutOfMemoryError", r"oom[-_]kill"i,
                      r"Killed process", r"std::bad_alloc"]
const SEGV_PATTERNS = [r"Segmentation fault", r"signal \(11\)", r"signal 11", r"SIGSEGV"]

function classify(; timed_out, exitcode, termsignal, log, has_result)
    timed_out && return ("t", "exceeded timeout")
    for p in OOM_PATTERNS
        m = match(p, log)
        m === nothing || return ("m", "log matched '$(m.match)'")
    end
    termsignal == 9 && return ("m", "killed by SIGKILL (likely OOM killer)")
    for p in SEGV_PATTERNS
        m = match(p, log)
        m === nothing || return ("s", "log matched '$(m.match)'")
    end
    termsignal == 11 && return ("s", "terminated by SIGSEGV")
    termsignal != 0 && return ("e", "terminated by signal $termsignal")
    exitcode != 0 && return ("e", "exit code $exitcode")
    has_result || return ("e", "driver exited cleanly but wrote no result.json")
    return ("ok", "")
end

#####
##### perf output → per-call counts
#####
#
# perf runs in interval mode (`perf stat -I <ms> -x,`): every <ms> it prints, for each
# event, the count accumulated during that interval. The driver enables counting only
# inside each measured call (FIFO control) and the problem for the next call is built
# with counting disabled, which takes seconds, i.e. many intervals. An interval in
# which an event was never enabled prints `<not counted>`; any interval that overlaps
# an enabled window prints a number (possibly 0). Hence the maximal runs of consecutive
# "counted" intervals are exactly the measured calls, in order, and summing each run
# gives that call's counts. Checks: the number of runs must equal the number of
# measured calls, and the per-call counts sum to the whole-run total by construction.

numval(s) = (v = tryparse(Float64, strip(s)); v === nothing ? nothing : (isinteger(v) ? Int(v) : v))

"Parse `perf stat -I -x,` output → ordered intervals [(t, Dict(event => count|nothing))]."
function parse_perf_intervals(path)
    isfile(path) || return nothing
    intervals = Vector{Tuple{Float64, Dict{String, Any}}}()
    for line in eachline(path)
        (isempty(strip(line)) || startswith(line, "#")) && continue
        cols = split(line, ',')
        length(cols) >= 4 || continue
        t = tryparse(Float64, strip(cols[1]))
        t === nothing && continue
        event = strip(cols[4])
        if isempty(intervals) || intervals[end][1] != t
            push!(intervals, (t, Dict{String, Any}()))
        end
        intervals[end][2][event] = numval(cols[2])   # "<not counted>" → nothing
    end
    return intervals
end

"Group counted intervals into windows (one per measured call); sum counts per event."
function perf_windows(intervals)
    windows = Vector{Dict{String, Any}}()
    inwin = false
    for (_, counts) in intervals
        counted = any(v -> v !== nothing, values(counts))
        if counted
            inwin || push!(windows, Dict{String, Any}())
            for (ev, v) in counts
                w = windows[end]
                w[ev] = get(w, ev, 0) + something(v, 0)
            end
        end
        inwin = counted
    end
    return windows
end

"Attach per-call perf counts/FLOPs to the measured calls in `calls`; returns a summary."
function attach_perf!(calls, windows)
    measured = [c for c in calls if get(c, "kind", "") == "measured"]
    flops(w) = isempty(w) ? nothing : sum(values(w))
    summary = Dict{String, Any}("n_windows" => length(windows), "n_measured" => length(measured))
    # A timed-out case may have one more (partial) window than completed calls.
    ok = length(windows) == length(measured) || length(windows) == length(measured) + 1
    if ok
        for (c, w) in zip(measured, windows)
            c["perf_counts"] = w
            c["flops"] = flops(w)
        end
        summary["attribution"] = length(windows) == length(measured) ? "ok" : "ok (last window partial, dropped)"
    else
        summary["attribution"] = "failed: $(length(windows)) counted windows for $(length(measured)) measured calls"
        @warn "perf attribution " * summary["attribution"]
    end
    return summary
end

#####
##### One case
#####

function run_case(o, name, cfg, N; use_perf, use_numactl)
    dir = joinpath(o["out"], name, @sprintf("N%04d", N))
    status_path = joinpath(dir, "status.json")
    if o["skip-existing"] && isfile(status_path) && JSON.parsefile(status_path)["status"] == "ok"
        @info "skip $name N=$N (already ok)"
        return
    end
    mkpath(dir)
    rm(joinpath(dir, "result.json"); force = true)
    log_path  = joinpath(dir, "log.txt")
    perf_path = joinpath(dir, "perf.csv")

    driver = joinpath(REPO_ROOT, "bench", "driver_$(cfg["driver"]).jl")
    julia  = Base.julia_cmd()   # same julia binary as this orchestrator
    jflags = split(o["julia-flags"])
    inner = `$julia $jflags --project=$REPO_ROOT $driver --config $name --nsteps $N --out $(o["out"]) --configs-file $(o["configs-file"]) $(o["driver-args"])`

    env = copy(ENV)
    env["JULIA_NUM_THREADS"] = string(o["threads"])
    env["XLA_FLAGS"] = o["xla-flags"]
    env["BENCH_JULIA_FLAGS"] = o["julia-flags"]
    delete!(env, "PERF_CTL_FIFO"); delete!(env, "PERF_ACK_FIFO")

    fifos = String[]
    cmd = inner
    if use_perf
        ctl, ack = tempname() * ".perf_ctl", tempname() * ".perf_ack"
        run(`mkfifo $ctl $ack`); append!(fifos, [ctl, ack])
        env["PERF_CTL_FIFO"], env["PERF_ACK_FIFO"] = ctl, ack
        cmd = `perf stat -x, -I $(o["perf-interval-ms"]) -o $perf_path --delay=-1 -e $(o["perf-events"]) --control fifo:$ctl,$ack -- $cmd`
    end
    use_numactl && (cmd = `numactl -C $(o["core"]) $cmd`)

    @info "[$(now())] $name N=$N" command=string(cmd)
    if o["dry-run"]
        println(cmd); return
    end

    t0 = time()
    timed_out = false
    proc = open(log_path, "w") do io
        p = run(pipeline(Cmd(setenv(cmd, env); detach = true); stdout = io, stderr = io); wait = false)
        if timedwait(() -> process_exited(p), o["timeout"]; pollint = 1.0) === :timed_out
            timed_out = true
            pid = getpid(p)
            try run(`kill -KILL -- -$pid`) catch end   # whole process group
            try kill(p, Base.SIGKILL) catch end
        end
        wait(p)
        p
    end
    elapsed = time() - t0
    foreach(f -> rm(f; force = true), fifos)

    log = read(log_path, String)
    result_path = joinpath(dir, "result.json")
    has_result = isfile(result_path)
    status, reason = classify(; timed_out, exitcode = proc.exitcode, termsignal = proc.termsignal,
                              log, has_result)

    # perf: per-call attribution of counts (see parse_perf_intervals / perf_windows)
    perf_summary = nothing
    if use_perf
        intervals = parse_perf_intervals(perf_path)
        windows = intervals === nothing ? Dict{String, Any}[] : perf_windows(intervals)
        totals = Dict{String, Any}()
        for w in windows, (ev, v) in w
            totals[ev] = get(totals, ev, 0) + v
        end
        perf_summary = Dict{String, Any}("events" => o["perf-events"], "interval_ms" => o["perf-interval-ms"],
                                         "totals_all_windows" => totals)
        # result.json (complete case) or progress.json (case cut off): attach per call
        for p in (result_path, joinpath(dir, "progress.json"))
            isfile(p) || continue
            data = JSON.parsefile(p; dicttype = Dict{String, Any})
            calls = data isa AbstractDict ? data["calls"] : data
            merge!(perf_summary, attach_perf!(calls, windows))
            if data isa AbstractDict
                fl = [c["flops"] for c in calls if get(c, "kind", "") == "measured" && get(c, "flops", nothing) !== nothing]
                data["perf"] = perf_summary
                data["flops_stats"] = isempty(fl) ? nothing : sample_stats(fl)
                data["flops_per_call"] = isempty(fl) ? nothing : sum(fl) / length(fl)
            end
            open(p, "w") do io; JSON.json(io, data; pretty = true); end
            p == result_path && break   # result.json supersedes progress.json
        end
    end

    st = Dict{String, Any}("config" => name, "nsteps" => N, "status" => status, "reason" => reason,
        "exitcode" => proc.exitcode, "termsignal" => proc.termsignal, "elapsed_s" => elapsed,
        "timeout_s" => o["timeout"], "command" => string(cmd), "xla_flags" => o["xla-flags"],
        "julia_flags" => o["julia-flags"], "threads" => o["threads"], "numactl" => use_numactl,
        "perf" => perf_summary, "finished_at" => string(now()))
    open(status_path, "w") do io; JSON.json(io, st; pretty = true); end

    msg = @sprintf("%-32s N=%4d  %-2s  %8.1f s  %s", name, N, status, elapsed, reason)
    status == "ok" ? @info(msg) : @warn(msg)
    return status
end

function main(args)
    o = parse_args(args)
    all = TOML.parsefile(o["configs-file"])
    names = o["configs"] == "table" ? all["table"]["rows"] : split(o["configs"], ',')
    nsteps = o["nsteps"] === nothing ? all["table"]["nsteps"] : parse.(Int, split(o["nsteps"], ','))
    for n in names
        haskey(all["configs"], n) || error("unknown config '$n'")
    end

    # auto: use perf/numactl only if a probe run succeeds on this machine. The default
    # perf events are Neoverse V2 (Grace) FP counters; elsewhere pass --perf-events.
    use_perf = resolve_mode(o["perf"], perf_works(o["perf-events"]), "perf (with events $(o["perf-events"]))")
    use_numactl = resolve_mode(o["numactl"], numactl_works(o["core"]), "numactl")

    @info "Running $(length(names)) config(s) × $(length(nsteps)) N" names nsteps timeout=o["timeout"] perf=use_perf numactl=use_numactl xla_flags=o["xla-flags"] julia_flags=o["julia-flags"]
    statuses = Dict{Tuple{String, Int}, String}()
    for n in names, N in nsteps
        s = run_case(o, n, all["configs"][n], N; use_perf, use_numactl)
        s === nothing || (statuses[(n, N)] = s)
    end
    isempty(statuses) || println("\nSummary: ", join(["$(k[1])@$(k[2])=$v" for (k, v) in sort(collect(statuses))], "  "))
end

main(ARGS)
