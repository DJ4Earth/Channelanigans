# Forward-model consistency check: Reactant-compiled vs plain-CPU Oceananigans.
#
#   julia --project bench/forward_consistency.jl [N=1] [--reactant-init]
#                                                [--weno-division default|normal|float32]
#
# Runs N forward steps (no AD) of the channel model on CPU() and on ReactantState()
# from bit-identical initial conditions and reports
#   * the objective (zonal transport) on both,
#   * max relative field differences (u, v, w, T, S, η),
#   * conservation of the initially uniform salinity (S ≡ 35, no forcing, no-flux BCs):
#     flux-form advection must keep it exactly 35 in every fluid cell.
#
# Known result (2026-11-03, Oceananigans 0.116.1, Reactant 0.2.304, CUDA.jl 6.2.2):
#   * without --reactant-init: CPU keeps S ≡ 35; Reactant drifts in ~38.7k fluid
#     cells after one step (mostly the surface layer, max ~2.8e-3 next to the wall)
#     and J differs by ~1e-4 after 4 steps. Cause: the Reactant extension makes
#     `maybe_prepare_first_time_step!` a no-op (initialization is left to
#     `first_time_step!`), so calling `time_step!` directly skips the initial
#     `update_state!` (halo fill, immersed masking, pressure, tendencies).
#   * with --reactant-init (what bench/driver_reactant.jl does): J agrees to ~8e-13
#     after 8 steps, all fields to ≤ 1e-10, S ≡ 35 on both backends.

include(joinpath(dirname(@__DIR__), "src", "ChannelModel.jl"))
using .ChannelModel
using Oceananigans
using Oceananigans.Architectures: ReactantState
using CUDA        # required by Reactant for KernelAbstractions kernels, even on CPU
using Reactant
using Printf
Reactant.set_default_backend("cpu")

const NS = something(tryparse(Int, get(filter(a -> !startswith(a, "--"), ARGS), 1, "1")), 1)
const DIVISION = let i = findfirst(==("--weno-division"), ARGS)
    spec = i === nothing ? "default" : ARGS[i+1]
    spec == "default" ? nothing :
    spec == "normal"  ? Oceananigans.Utils.NormalDivision :
    spec == "float32" ? Oceananigans.Utils.ConvertingDivision{Float32} :
    error("unknown --weno-division $spec")
end

# --reactant-init: before the loop, do on Reactant what vanilla `time_step!` does at
# iteration 0 via `maybe_prepare_first_time_step!` (a no-op in the Reactant extension):
# `initialize!` (= reconcile_state!) + `update_state!`.
const REACTANT_INIT = "--reactant-init" in ARGS

function forward!(model, Tᵢ, Sᵢ, u, v, flux, Δz)
    set_conditions!(model, Tᵢ, Sᵢ, u, v, flux)
    if REACTANT_INIT && Oceananigans.Architectures.architecture(model.grid) isa ReactantState
        Oceananigans.initialize!(model)
        Oceananigans.TimeSteppers.update_state!(model)
    end
    Δt = model.clock.last_Δt
    @trace track_numbers = false for i = 1:NS
        time_step!(model, Δt)
    end
    return zonal_transport(model, Δz)
end

state(m) = (u = m.velocities.u, v = m.velocities.v, w = m.velocities.w,
            T = m.tracers.T, S = m.tracers.S, η = m.free_surface.displacement)
host(f) = convert(Array, interior(f))

c = build_problem(CPU(); weno_weight_computation = DIVISION)
Jc = forward!(c.model, c.Tᵢ, c.Sᵢ, c.u_wind_stress, c.v_wind_stress, c.T_flux, c.Δz)

r = build_problem(ReactantState(); to_device = Reactant.to_rarray, weno_weight_computation = DIVISION)
rforward! = @compile raise_first=true raise=true sync=true forward!(r.model, r.Tᵢ, r.Sᵢ, r.u_wind_stress, r.v_wind_stress, r.T_flux, r.Δz)
Jr = Float64(rforward!(r.model, r.Tᵢ, r.Sᵢ, r.u_wind_stress, r.v_wind_stress, r.T_flux, r.Δz))

println("\n== Forward consistency after N = $NS step(s), WENO division = $(something(DIVISION, "architecture default")), reactant-init = $REACTANT_INIT")
@printf("objective  CPU %.16e   Reactant %.16e   rel. diff %.2e\n", Jc, Jr, abs(Jr - Jc) / abs(Jc))
for (k, fc) in pairs(state(c.model))
    a, b = host(fc), host(getproperty(state(r.model), k))
    d = abs.(a .- b); s = maximum(abs, a)
    @printf("  %-2s max|CPU| %.3e   max|Δ|/max|CPU| %.2e   at %s\n", k, s, maximum(d) / max(s, eps()), Tuple(argmax(d)))
end

fluid = host(c.model.tracers.S) .!= 0     # immersed (wall) cells are zeroed after a step
for (name, m) in (("CPU", c.model), ("Reactant", r.model))
    s = host(m.tracers.S)
    drift = abs.(s .- 35) .* fluid
    nbad = count(>(1e-10), drift)
    @printf("uniform-S conservation  %-8s  drifting fluid cells %6d / %d   max |S-35| %.3e\n",
            name, nbad, count(fluid), maximum(drift))
end
