# Shared driver plumbing: CLI parsing, config lookup, and the measurement protocol.
# Included (not a module) by bench/driver_reactant.jl and bench/driver_enzyme.jl.

using TOML
using JSON
using JLD2
using Printf
using Statistics: median

const REPO_ROOT = dirname(@__DIR__)
include(joinpath(REPO_ROOT, "src", "ChannelModel.jl"))
include(joinpath(REPO_ROOT, "src", "Measure.jl"))
include(joinpath(@__DIR__, "stats.jl"))
using .ChannelModel
using .Measure

const DRIVER_USAGE = """
usage: julia --project bench/driver_<backend>.jl --config NAME --nsteps N [options]

  --config NAME          entry of bench/configs.toml
  --nsteps N             timesteps in the differentiated window
  --out DIR              results root (default: results); writes DIR/NAME/N<nnnn>/result.json
  --warmup W             warmup calls, not used for statistics (default 1; the first
                         call is always the reference for objective/gradients and
                         includes compilation/JIT)
  --repeats R            measured calls used for the statistics (default 12)
  --repeat-budget S      optional cap for smoke runs: reduce measured calls so that
                         R × (warmup time) ≤ S seconds, at least 2 (default: no cap)
  --seed K               RNG seed for the initial-condition noise (default 1234)
  --save-gradients       also write gradient fields of the reference call to JLD2
  --configs-file PATH    alternative configs TOML (default bench/configs.toml)
"""

function parse_driver_args(args)
    opts = Dict{String, Any}(
        "out" => "results", "warmup" => 1, "repeats" => 12, "repeat-budget" => Inf,
        "seed" => 1234, "save-gradients" => false,
        "configs-file" => joinpath(REPO_ROOT, "bench", "configs.toml"))
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("-h", "--help")
            print(DRIVER_USAGE); exit(0)
        elseif a == "--save-gradients"
            opts["save-gradients"] = true
        elseif startswith(a, "--") && i < length(args)
            opts[a[3:end]] = args[i+1]; i += 1
        else
            error("unrecognized argument $a\n\n$DRIVER_USAGE")
        end
        i += 1
    end
    haskey(opts, "config") && haskey(opts, "nsteps") || error("--config and --nsteps are required\n\n$DRIVER_USAGE")
    for k in ("nsteps", "warmup", "repeats", "seed")
        opts[k] isa AbstractString && (opts[k] = parse(Int, opts[k]))
    end
    opts["repeat-budget"] isa AbstractString && (opts["repeat-budget"] = parse(Float64, opts["repeat-budget"]))
    opts["warmup"] >= 1 || error("--warmup must be ≥ 1 (the first call is the reference call)")
    opts["repeats"] >= 2 || error("--repeats must be ≥ 2 (needed for a confidence interval)")
    return opts
end

function load_config(opts, driver)
    all = TOML.parsefile(opts["configs-file"])
    name = opts["config"]
    haskey(all["configs"], name) || error("unknown config '$name'; available: $(join(keys(all["configs"]), ", "))")
    cfg = all["configs"][name]
    cfg["driver"] == driver || error("config '$name' is for driver '$(cfg["driver"])', not '$driver'")
    return cfg
end

"Resolve a `snaps` spec (integer, \"log2\", \"isqrt\") for N timesteps."
function resolve_snaps(spec, N)
    spec isa Integer && return Int(spec)
    spec == "log2"  && (ispow2(N) || error("snaps = \"log2\" requires N to be a power of 2, got $N"); return Int(log2(N)))
    spec == "isqrt" && return isqrt(N)
    error("unrecognized snaps spec $(repr(spec))")
end

case_dir(opts) = joinpath(opts["out"], opts["config"], @sprintf("N%04d", opts["nsteps"]))

#####
##### Gradient summaries (reference call only)
#####

function field_summary(a)
    a = Array(a)
    nonfinite = count(!isfinite, a)
    fa = filter(isfinite, a)
    return Dict("l2" => sqrt(sum(abs2, fa; init=0.0)), "maxabs" => maximum(abs, fa; init=0.0),
                "sum" => sum(fa; init=0.0), "nonfinite" => nonfinite, "size" => collect(size(a)))
end

"Host copies of the gradient fields that are compared across configurations."
gradient_arrays(sh, interior) = Dict(
    "dT"             => convert(Array, interior(sh.dTᵢ)),
    "dS"             => convert(Array, interior(sh.dSᵢ)),
    "du_wind_stress" => convert(Array, interior(sh.du_wind_stress)),
    "dv_wind_stress" => convert(Array, interior(sh.dv_wind_stress)),
    "dT_flux"        => convert(Array, interior(sh.dT_flux)),
    "dDz"            => convert(Array, sh.dΔz))

#####
##### Protocol
#####

strdict(nt::NamedTuple) = Dict{String, Any}(string(k) => v for (k, v) in pairs(nt))

"""
    run_protocol(; build, call, objective, interior, opts, cfg, driver, extra)

Common measurement protocol for one (config, N) case:

1. `build()` a fresh problem + zeroed shadows; call 1 (reference/warmup) runs from
   rest. Its objective and gradients are recorded for cross-configuration checks.
   It includes Reactant's first-execution overhead or the Enzyme driver's JIT, and
   is excluded from the statistics.
2. Further warmup calls (if `--warmup > 1`), each on a freshly built problem.
3. `--repeats` measured calls (default 12), each on a freshly built problem with
   fresh shadows (built outside the timed window), with perf counters enabled only
   inside the call. Runtime and memory are reported as mean ± 95% Student-t CI.
   An optional `--repeat-budget` reduces the count for smoke runs only.

Every call computes exactly the same thing; the per-call objectives are recorded and
checked for consistency. Progress is written after every call to `progress.json`, so
a case cut off by a timeout still leaves its completed calls on disk.
"""
function run_protocol(; build, call, objective, interior, opts, cfg, driver, extra = Dict())
    dir = case_dir(opts)
    mkpath(dir)

    calls = Dict{String, Any}[]
    progress() = write_json_atomic(joinpath(dir, "progress.json"), JSON.json(calls; pretty = true))

    # 1. Reference call (warmup, no perf)
    @info "Reference call (warmup 1/$(opts["warmup"]))"
    pr, sh = build()
    y, st = measured_call(() -> call(pr, sh); perf = false)
    J_ref = objective(y)
    push!(calls, merge(strdict(st), Dict("kind" => "warmup", "objective" => J_ref)))
    progress()
    grads = gradient_arrays(sh, interior)
    grad_summary = Dict(k => field_summary(v) for (k, v) in grads)
    if opts["save-gradients"]
        jld = joinpath(dir, "gradients.jld2")
        jldsave(jld; objective = J_ref, (Symbol(k) => v for (k, v) in grads)...)
        @info "saved gradients to $jld"
    end
    pr = sh = y = nothing
    t_warm = st.wall_s

    # 2. Extra warmups
    for w in 2:opts["warmup"]
        @info "Warmup call $w/$(opts["warmup"])"
        pr, sh = build()
        y, st = measured_call(() -> call(pr, sh); perf = false)
        push!(calls, merge(strdict(st), Dict("kind" => "warmup", "objective" => objective(y))))
        progress()
        t_warm = st.wall_s
        pr = sh = y = nothing
    end

    # 3. Measured calls
    nrep = opts["repeats"]
    if isfinite(opts["repeat-budget"])
        nrep = clamp(floor(Int, opts["repeat-budget"] / max(t_warm, 1e-9)), 2, nrep)
    end
    @info "Measuring $nrep call(s) (last warmup took $(round(t_warm, digits=2)) s)"
    measured = Dict{String, Any}[]
    for r in 1:nrep
        pr, sh = build()
        y, st = measured_call(() -> call(pr, sh); perf = true)
        rec = merge(strdict(st), Dict("kind" => "measured", "objective" => objective(y)))
        push!(calls, rec); push!(measured, rec)
        progress()
        pr = sh = y = nothing
    end

    objectives = [c["objective"] for c in calls]
    finite = all(isfinite, objectives)
    consistent = finite && all(o -> isapprox(o, J_ref; rtol = 1e-12, atol = 0), objectives)
    consistent || @warn "Per-call objectives differ" objectives

    runtime = sample_stats([m["wall_s"] for m in measured])
    peak    = sample_stats([Float64(m["peak_rss_bytes"]) for m in measured])
    delta   = sample_stats([Float64(m["delta_rss_bytes"]) for m in measured])
    result = Dict{String, Any}(
        "config"       => opts["config"],
        "config_spec"  => cfg,
        "driver"       => driver,
        "nsteps"       => opts["nsteps"],
        "seed"         => opts["seed"],
        "protocol"     => Dict("warmup" => opts["warmup"], "repeats" => opts["repeats"],
                               "repeat_budget_s" => isfinite(opts["repeat-budget"]) ? opts["repeat-budget"] : nothing,
                               "n_measured" => nrep, "statistic" => "mean ± 95% Student-t CI"),
        # Headline numbers: means over the measured calls (full stats below)
        "runtime_s"        => runtime["mean"],
        "runtime_ci95_halfwidth_s" => runtime["ci95_halfwidth"],
        "peak_rss_bytes"   => peak["mean"],
        "delta_rss_bytes"  => delta["mean"],
        "runtime_stats"    => runtime,
        "peak_rss_stats"   => peak,
        "delta_rss_stats"  => delta,
        "memory_method"    => measured[1]["memory_method"],
        "objective"        => J_ref,
        "objectives_consistent" => consistent,
        "gradients"        => grad_summary,
        "calls"            => calls,
        "environment"      => environment_info(),
    )
    merge!(result, extra)

    path = joinpath(dir, "result.json")
    write_json_atomic(path, JSON.json(result; pretty = true))
    @info @sprintf("[%s N=%d] runtime %.3f ± %.3f s (95%% CI, n=%d), peak RSS %.3f GiB, objective %.12e → %s",
                   opts["config"], opts["nsteps"], runtime["mean"], runtime["ci95_halfwidth"], nrep,
                   peak["mean"] / 2^30, J_ref, path)

    finite || (@error "Non-finite objective"; exit(3))
    return result
end
