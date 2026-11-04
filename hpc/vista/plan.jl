# Status of the sweep + walltime predictions for the cases not yet done.
#
#   (via hpc/vista/report.sh status)   julia plan.jl RESULTS_DIR CONFIGS_TOML WALLTIMES_OUT MAX_WALLTIME_H MARGIN_S
#
# For each config, fits per-call time t(N) and peak memory M(N) to the finished cases
# (power law through the two largest N; one point → linear), then predicts the
# elapsed time of a full case (1 warmup + 12 measured calls) as
#     fixed overhead (compile, JIT, process start; from the largest finished N) × 1.5
#   + 13 · t(N)
# The walltime asked for is that × 1.25 + MARGIN_S (the orchestrator timeout margin),
# rounded up to 30 min, at least 1 h, capped at MAX_WALLTIME_H. Cases whose predicted
# memory exceeds the node or whose predicted time exceeds the cap are flagged; they
# are still listed (they become legitimate m / t table entries).

using JSON, TOML, Printf

const NCALLS = 13

function load(dir)
    st = joinpath(dir, "status.json")
    isfile(st) || return nothing
    s = JSON.parsefile(st)
    r = isfile(joinpath(dir, "result.json")) ? JSON.parsefile(joinpath(dir, "result.json")) : nothing
    p = r === nothing && isfile(joinpath(dir, "progress.json")) ? JSON.parsefile(joinpath(dir, "progress.json")) : nothing
    calls = r !== nothing ? r["calls"] : p !== nothing ? p : Any[]
    meas = [c["wall_s"] for c in calls if c["kind"] == "measured"]
    peak = isempty(calls) ? nothing : maximum(c["peak_rss_bytes"] for c in calls)
    mem = r === nothing ? nothing : r["environment"]["total_memory_bytes"]
    return (status = s["status"], elapsed = s["elapsed_s"], ncalls = length(calls),
            tcall = isempty(meas) ? nothing : sum(meas) / length(meas), peak, mem)
end

"Power law through the two largest points (exponent clamped to `lims`); linear if one point."
function fit(pts, lims)
    pts = sort(filter(p -> p[2] !== nothing && p[2] > 0, pts))
    isempty(pts) && return nothing
    if length(pts) == 1
        (N1, y1) = pts[1]; return N -> y1 * N / N1
    end
    (N1, y1), (N2, y2) = pts[end-1], pts[end]
    p = clamp(log(y2 / y1) / log(N2 / N1), lims...)
    return N -> y2 * (N / N2)^p
end

hms(s) = @sprintf("%02d:%02d:00", s ÷ 3600, (s % 3600) ÷ 60)

function main(resdir, cfgfile, wtfile, maxh, margin)
    all = TOML.parsefile(cfgfile)
    names, nsteps = all["table"]["rows"], all["table"]["nsteps"]
    maxs = maxh * 3600
    lines = String[]
    @printf("%-30s %5s  %-6s %10s %10s %9s   %s\n", "config", "N", "status", "elapsed", "t/call", "peak GiB", "next walltime / notes")
    for name in names
        cases = Dict(N => load(joinpath(resdir, name, @sprintf("N%04d", N))) for N in nsteps)
        done = [(N, c) for (N, c) in cases if c !== nothing && c.tcall !== nothing]
        tfit = fit([(N, c.tcall) for (N, c) in done], (1.0, 2.0))
        mfit = fit([(N, Float64(c.peak)) for (N, c) in done if c.status == "ok"], (0.0, 2.0))
        big = isempty(done) ? nothing : last(sort(done; by = first))[2]
        fixed = big === nothing ? nothing : max(big.elapsed - (big.ncalls) * big.tcall, 60.0)
        mems = [c.mem for (_, c) in done if c.mem !== nothing]
        nodemem = isempty(mems) ? nothing : first(mems)
        oomed = sort([N for (N, c) in cases if c !== nothing && c.status == "m"])
        for N in nsteps
            c = cases[N]
            stat = c === nothing ? "-" : c.status
            el = c === nothing ? "" : @sprintf("%.0f s", c.elapsed)
            tc = c === nothing || c.tcall === nothing ? "" : @sprintf("%.1f s", c.tcall)
            pk = c === nothing || c.peak === nothing ? "" : @sprintf("%.2f", c.peak / 2^30)
            note = ""
            if stat != "ok"
                if tfit === nothing
                    note = "no finished case for this config yet"
                else
                    pred = 1.5fixed + NCALLS * tfit(N)
                    wt = clamp(ceil(Int, (1.25pred + margin) / 1800) * 1800, 3600, maxs)
                    note = @sprintf("%s (predicted %.1f h)", hms(wt), pred / 3600)
                    1.25pred + margin > maxs && (note *= "  likely t")
                    if mfit !== nothing && nodemem !== nothing && mfit(N) > 0.95nodemem
                        note *= @sprintf("  likely m (%.0f GiB predicted)", mfit(N) / 2^30)
                    end
                    any(<(N), oomed) && (note *= "  smaller N was m")
                    push!(lines, "$name\t$N\t$(hms(wt))")
                end
            end
            @printf("%-30s %5d  %-6s %10s %10s %9s   %s\n", name, N, stat, el, tc, pk, note)
        end
    end
    write(wtfile, join(lines, "\n") * (isempty(lines) ? "" : "\n"))
    println("\nwrote $(length(lines)) walltime(s) to $wtfile")
end

main(ARGS[1], ARGS[2], ARGS[3], parse(Int, ARGS[4]), parse(Int, ARGS[5]))
