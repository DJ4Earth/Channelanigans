# Cross-check objectives and gradients across configurations.
#
#   julia --project bench/check_gradients.jl [--out results] [--reference reactant_full]
#                                            [--rtol-same 1e-10] [--rtol-cross 1e-10]
#
# For every N, compares each ok configuration against the reference configuration:
#   * objective (zonal transport, Sv)
#   * summary norms of each gradient field (always available)
#   * full gradient fields, max |a-b| / max |b|, if both cases were run with
#     --save-gradients
#
# Tolerances: configurations on the same backend as the reference should agree to
# `rtol-same` (checkpointing/min-cut change the schedule, not the arithmetic);
# Reactant vs Enzyme-only CPU comparisons use `rtol-cross`.
# Observed on the Mac (2026-11-03, Oceananigans 0.116.1, Reactant 0.2.304,
# Enzyme 0.13.216), N = 4 and 8: worst same-backend error ≤ 1e-14, worst
# cross-backend error ≤ 8e-13 (objective and every full gradient field).
# Cross-backend agreement requires the explicit first-step initialization in
# bench/driver_reactant.jl; without it the backends differ by ~1e-4 in J.
#
# Exit code 1 if any comparison fails.

using JSON
using JLD2
using Printf

function parse_args(args)
    o = Dict{String, Any}("out" => "results", "reference" => "reactant_full",
                          "rtol-same" => 1e-10, "rtol-cross" => 1e-10)
    for i in 1:2:length(args)
        k = args[i]; startswith(k, "--") || error("bad argument $k")
        o[k[3:end]] = args[i+1]
    end
    for k in ("rtol-same", "rtol-cross")
        o[k] isa AbstractString && (o[k] = parse(Float64, o[k]))
    end
    return o
end

relerr(a, b) = abs(a - b) / max(abs(b), eps())

function main(args)
    o = parse_args(args)
    out = o["out"]
    cases = Dict{Int, Dict{String, Any}}()   # N => config => result
    for cfg in readdir(out), nd in (isdir(joinpath(out, cfg)) ? readdir(joinpath(out, cfg)) : String[])
        rp = joinpath(out, cfg, nd, "result.json")
        sp = joinpath(out, cfg, nd, "status.json")
        isfile(rp) || continue
        isfile(sp) && JSON.parsefile(sp)["status"] != "ok" && continue
        r = JSON.parsefile(rp)
        r["_dir"] = joinpath(out, cfg, nd)
        get!(cases, r["nsteps"], Dict{String, Any}())[cfg] = r
    end

    failures = 0
    for N in sort(collect(keys(cases)))
        byname = cases[N]
        ref_name = haskey(byname, o["reference"]) ? o["reference"] : first(sort(collect(keys(byname))))
        ref = byname[ref_name]
        println("\n== N = $N   (reference: $ref_name, J = $(@sprintf("%.15e", ref["objective"])) Sv)")
        ref_jld = joinpath(ref["_dir"], "gradients.jld2")
        for name in sort(collect(keys(byname)))
            name == ref_name && continue
            r = byname[name]
            tol = r["driver"] == ref["driver"] ? o["rtol-same"] : o["rtol-cross"]
            errs = Dict{String, Float64}("objective" => relerr(r["objective"], ref["objective"]))
            for (k, s) in ref["gradients"]
                haskey(r["gradients"], k) || continue
                errs["|$k|"] = relerr(r["gradients"][k]["l2"], s["l2"])
                r["gradients"][k]["nonfinite"] > 0 && (errs["nonfinite($k)"] = Inf)
            end
            jld = joinpath(r["_dir"], "gradients.jld2")
            if isfile(jld) && isfile(ref_jld)
                a, b = load(jld), load(ref_jld)
                for k in keys(b)
                    k == "objective" && continue
                    haskey(a, k) || continue
                    errs["$k(field)"] = maximum(abs, a[k] .- b[k]; init = 0.0) / max(maximum(abs, b[k]; init = 0.0), eps())
                end
            end
            worst = maximum(values(errs))
            ok = worst <= tol
            ok || (failures += 1)
            wk = argmax(errs)
            @printf("  %-32s %-4s worst rel. err %.2e (%s), tol %.0e\n", name, ok ? "ok" : "FAIL", worst, wk, tol)
        end
    end
    println(failures == 0 ? "\nAll comparisons passed." : "\n$failures comparison(s) FAILED.")
    exit(failures == 0 ? 0 : 1)
end

main(ARGS)
