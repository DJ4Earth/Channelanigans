# Pool batched runs of the CPU AD cost table into one results tree.
#
#   julia --project bench/merge_batches.jl --root DIR --sweep NAME
#
# A batched sweep (hpc/vista/env.sh, BATCH_REPEATS) runs each (config, N) case as
# several independent batches: separate processes, usually on different nodes, each
# with its own compilation, 1 warmup and R measured calls, in DIR/NAME_b1,
# DIR/NAME_b2, ... This script writes DIR/NAME/<config>/N<nnnn>/ with the usual
# status.json and result.json, so make_table.jl, check_gradients.jl and export_raw.jl
# work on DIR/NAME unchanged:
#
#   * The measured calls of every batch that finished ok are pooled (batch 1 first);
#     runtime, memory and FLOP statistics are recomputed over the pool (n = sum of R).
#     Each call gets a `batch` number, and the `hostname` and `slurm_job_id` of its
#     batch. Warmup calls of every batch are kept (kind = warmup).
#   * The reference objective and gradients (result fields `objective`, `gradients`,
#     gradients.jld2) are those of the first ok batch. Every call of every batch must
#     return the same objective (`objectives_consistent`); the relative spread of the
#     gradient norms across batches is recorded (`gradients_max_rel_spread_across_batches`).
#   * `batches` (in result.json and status.json) lists every batch of the case,
#     including ones that did not finish ok, with its outcome, node and per-batch
#     means, so the node-to-node spread stays visible.
#   * A case with no ok batch takes status/progress from its first batch (e.g. "m").
#   * Batches still running (no status.json yet) are left out; rerun when they finish.

using JSON
using Printf
include(joinpath(@__DIR__, "stats.jl"))

function parse_args(args)
    o = Dict{String, Any}()
    i = 1
    while i <= length(args)
        a = args[i]
        startswith(a, "--") && i < length(args) || error("usage: merge_batches.jl --root DIR --sweep NAME")
        o[a[3:end]] = args[i+1]; i += 2
    end
    haskey(o, "root") && haskey(o, "sweep") || error("usage: merge_batches.jl --root DIR --sweep NAME")
    return o
end

readjson(p) = isfile(p) ? JSON.parsefile(p; dicttype = Dict{String, Any}) : nothing
writejson(p, x) = open(io -> JSON.json(io, x; pretty = true), p, "w")
relspread(v) = (m = maximum(abs, v); m == 0 ? 0.0 : (maximum(v) - minimum(v)) / m)
mean_of(calls, f) = (x = [Float64(c[f]) for c in calls if get(c, f, nothing) !== nothing];
                     isempty(x) ? nothing : sum(x) / length(x))

function main(args)
    o = parse_args(args)
    root, sweep = o["root"], o["sweep"]
    bdirs = Dict{Int, String}()
    for d in readdir(root)
        m = match(Regex("^" * sweep * "_b([0-9]+)\$"), d)
        m === nothing || (bdirs[parse(Int, m[1])] = joinpath(root, d))
    end
    isempty(bdirs) && error("no batch directories $(sweep)_b<K> in $root")
    bnums = sort(collect(keys(bdirs)))
    println("batches: ", join(["b$k" for k in bnums], ", "))

    cases = Set{Tuple{String, String}}()
    for k in bnums, cfg in readdir(bdirs[k])
        isdir(joinpath(bdirs[k], cfg)) || continue
        for nd in readdir(joinpath(bdirs[k], cfg))
            startswith(nd, "N") && isdir(joinpath(bdirs[k], cfg, nd)) && push!(cases, (cfg, nd))
        end
    end

    out = joinpath(root, sweep)
    counts = Dict{String, Int}()
    for (cfg, nd) in sort(collect(cases))
        # every batch of this case that has finished (has status.json)
        bs = []
        for k in bnums
            dir = joinpath(bdirs[k], cfg, nd)
            st = readjson(joinpath(dir, "status.json"))
            st === nothing && continue
            res = st["status"] == "ok" ? readjson(joinpath(dir, "result.json")) : nothing
            st["status"] == "ok" && res === nothing && (st["status"] = "e"; st["reason"] = "no result.json")
            push!(bs, (; k, dir, st, res, job = readjson(joinpath(dir, "job_info.json"))))
        end
        isempty(bs) && continue
        dst = joinpath(out, cfg, nd)
        mkpath(dst)
        foreach(f -> rm(joinpath(dst, f); force = true),
                ("status.json", "result.json", "progress.json", "gradients.jld2"))

        host(b) = b.job !== nothing ? b.job["node"] :
                  b.res !== nothing ? b.res["environment"]["hostname"] : nothing
        jobid(b) = b.job === nothing ? nothing : b.job["slurm_job_id"]
        summary(b) = begin
            s = Dict{String, Any}("batch" => b.k, "status" => b.st["status"], "reason" => b.st["reason"],
                "elapsed_s" => b.st["elapsed_s"], "hostname" => host(b), "slurm_job_id" => jobid(b))
            if b.res !== nothing
                meas = [c for c in b.res["calls"] if c["kind"] == "measured"]
                merge!(s, Dict("n_measured" => length(meas), "wall_s_mean" => mean_of(meas, "wall_s"),
                    "peak_rss_bytes_mean" => mean_of(meas, "peak_rss_bytes"),
                    "delta_rss_bytes_mean" => mean_of(meas, "delta_rss_bytes"),
                    "flops_mean" => mean_of(meas, "flops"), "objective" => b.res["objective"],
                    "compile_s" => get(b.res, "compile_s", nothing),
                    "perf_attribution" => get(get(b.st, "perf", nothing) === nothing ? Dict() : b.st["perf"], "attribution", nothing)))
            end
            s
        end
        batches = [summary(b) for b in bs]
        ok = [b for b in bs if b.res !== nothing]

        if isempty(ok)
            # no usable batch: keep the first batch's outcome (e.g. m) and its partial calls
            b = bs[1]
            st = copy(b.st); st["batches"] = batches
            writejson(joinpath(dst, "status.json"), st)
            p = joinpath(b.dir, "progress.json")
            isfile(p) && cp(p, joinpath(dst, "progress.json"))
            counts[st["status"]] = get(counts, st["status"], 0) + 1
            continue
        end

        ref = ok[1].res
        calls = Dict{String, Any}[]
        for b in ok, c in b.res["calls"]
            c = copy(c)
            c["batch"] = b.k; c["hostname"] = host(b); c["slurm_job_id"] = jobid(b)
            push!(calls, c)
        end
        measured = [c for c in calls if c["kind"] == "measured"]
        runtime = sample_stats([c["wall_s"] for c in measured])
        peak    = sample_stats([Float64(c["peak_rss_bytes"]) for c in measured])
        delta   = sample_stats([Float64(c["delta_rss_bytes"]) for c in measured])
        fl = [c["flops"] for c in measured if get(c, "flops", nothing) !== nothing]
        J = ref["objective"]
        consistent = all(b -> b.res["objectives_consistent"], ok) &&
                     all(c -> isapprox(c["objective"], J; rtol = 1e-12, atol = 0), calls)
        spread = Dict{String, Any}(k => relspread([b.res["gradients"][k]["l2"] for b in ok])
                                   for k in keys(ref["gradients"]))
        hosts = unique(filter(!isnothing, [host(b) for b in ok]))

        res = copy(ref)
        merge!(res, Dict{String, Any}(
            "calls" => calls,
            "runtime_s" => runtime["mean"], "runtime_ci95_halfwidth_s" => runtime["ci95_halfwidth"],
            "peak_rss_bytes" => peak["mean"], "delta_rss_bytes" => delta["mean"],
            "runtime_stats" => runtime, "peak_rss_stats" => peak, "delta_rss_stats" => delta,
            "flops_stats" => isempty(fl) ? nothing : sample_stats(fl),
            "flops_per_call" => isempty(fl) ? nothing : sum(fl) / length(fl),
            "objectives_consistent" => consistent,
            "gradients_max_rel_spread_across_batches" => spread,
            "batches" => batches,
            "environment" => merge(copy(ref["environment"]), Dict{String, Any}("hostname" => join(hosts, ";"))),
        ))
        res["protocol"] = merge(copy(ref["protocol"]), Dict{String, Any}(
            "n_measured" => length(measured), "n_batches" => length(ok),
            "batches_pooled" => [b.k for b in ok],
            "statistic" => "mean ± 95% Student-t CI over the pooled measured calls of all ok batches"))
        attrs = [something(get(something(get(b.st, "perf", nothing), Dict()), "attribution", nothing), "none") for b in ok]
        perf = get(ok[1].st, "perf", nothing)
        if perf !== nothing
            perf = copy(perf)
            perf["attribution"] = all(a -> startswith(a, "ok"), attrs) ? "ok" :
                join(["b$(b.k): $a" for (b, a) in zip(ok, attrs)], "; ")
            res["perf"] = perf
        end
        st = copy(ok[1].st)
        notok = [b for b in bs if b.res === nothing]
        merge!(st, Dict{String, Any}("status" => "ok",
            "reason" => isempty(notok) ? "" : join(["b$(b.k) $(b.st["status"]): $(b.st["reason"])" for b in notok], "; "),
            "elapsed_s" => sum(b.st["elapsed_s"] for b in bs), "perf" => perf, "batches" => batches))
        writejson(joinpath(dst, "result.json"), res)
        writejson(joinpath(dst, "status.json"), st)
        g = joinpath(ok[1].dir, "gradients.jld2")
        isfile(g) && cp(g, joinpath(dst, "gradients.jld2"))
        counts["ok"] = get(counts, "ok", 0) + 1

        consistent || @warn "$cfg $nd: objectives differ across calls/batches"
        isempty(notok) || @warn "$cfg $nd: pooled $(length(ok)) ok batch(es); not ok: $(st["reason"])"
        maxspread = maximum(values(spread); init = 0.0)
        maxspread > 1e-10 && @warn @sprintf("%s %s: gradient norms differ across batches by %.1e", cfg, nd, maxspread)
    end
    println("wrote $(sum(values(counts); init = 0)) case(s) to $out: ",
            join(["$v $k" for (k, v) in sort(collect(counts))], ", "))
end

main(ARGS)
