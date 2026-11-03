# Reactant + Enzyme (MLIR) driver: one (config, N) case per process.
#
#   julia --project bench/driver_reactant.jl --config reactant_full --nsteps 16
#
# Normally launched by bench/run_cases.jl, which sets XLA_FLAGS / threads / perf.

include(joinpath(@__DIR__, "common.jl"))

const OPTS = parse_driver_args(ARGS)
const CFG  = load_config(OPTS, "reactant")

using Oceananigans
using Oceananigans.Architectures: ReactantState
# CUDA.jl must be loaded even on CPU: Reactant compiles KernelAbstractions kernels
# through its CUDA extension (`ka_with_reactant`) and then raises them to StableHLO.
using CUDA
using Reactant
using Enzyme

Reactant.set_default_backend("cpu")

#####
##### Configuration → compile-time constants
#####

const NSTEPS = OPTS["nsteps"]
const MINCUT = CFG["mincut"]::Bool

const OPTIMIZE = CFG["optimize"] == "default" ? true : Symbol(CFG["optimize"])

# `checkpointing = true` in @trace means Periodic(isqrt(N)) for static loop bounds;
# we construct that explicitly so the option can be passed as a value.
const CHECKPOINTING = let c = CFG["checkpointing"]
    c == "none"     ? false :
    c == "periodic" ? Reactant.Periodic(isqrt(NSTEPS)) :
    c == "binomial" ? Reactant.Binomial(resolve_snaps(CFG["snaps"], NSTEPS)) :
    error("unknown Reactant checkpointing '$c'")
end

@info "Reactant config" config=OPTS["config"] nsteps=NSTEPS optimize=OPTIMIZE mincut=MINCUT checkpointing=CHECKPOINTING

#####
##### Differentiated function
#####

function loop!(model)
    Δt = model.clock.last_Δt
    @trace mincut = MINCUT checkpointing = CHECKPOINTING track_numbers = false for i = 1:NSTEPS
        time_step!(model, Δt)
    end
    return nothing
end

function estimate_tracer_error(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, temp_flux, Δz, mld)
    set_conditions!(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, temp_flux)
    # On CPU, `time_step!` runs `maybe_prepare_first_time_step!` at iteration 0, which
    # calls `reconcile_state!` + `update_state!` (mask immersed cells, fill tracer halos,
    # hydrostatic pressure, closure fields, momentum tendencies) after the ICs are set.
    # The Oceananigans Reactant extension makes that a no-op and expects
    # `first_time_step!` to do it instead, so we do it explicitly here. Without this the
    # first Reactant step reads stale halos/tendencies and diverges from CPU (~1e-4 in J
    # after 4 steps); with it the two backends agree to round-off (see
    # bench/forward_consistency.jl --reactant-init).
    Oceananigans.initialize!(model)
    Oceananigans.TimeSteppers.update_state!(model)
    loop!(model)
    return zonal_transport(model, Δz)
end

function differentiate_tracer_error(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, temp_flux, Δz, mld,
                                    dmodel, dTᵢ, dSᵢ, du_wind_stress, dv_wind_stress, dtemp_flux, dΔz, dmld)
    return autodiff(set_strong_zero(Enzyme.ReverseWithPrimal),
                    estimate_tracer_error, Active,
                    Duplicated(model, dmodel),
                    Duplicated(Tᵢ, dTᵢ),
                    Duplicated(Sᵢ, dSᵢ),
                    Duplicated(u_wind_stress, du_wind_stress),
                    Duplicated(v_wind_stress, dv_wind_stress),
                    Duplicated(temp_flux, dtemp_flux),
                    Duplicated(Δz, dΔz),
                    Duplicated(mld, dmld))
end

#####
##### Build, compile, measure
#####

build() = begin
    pr = build_problem(ReactantState(); seed = OPTS["seed"], to_device = Reactant.to_rarray)
    sh = make_shadows(pr, Enzyme.make_zero)
    (pr, sh)
end

args(pr, sh) = (pr.model, pr.Tᵢ, pr.Sᵢ, pr.u_wind_stress, pr.v_wind_stress, pr.T_flux, pr.Δz, pr.mld,
                sh.dmodel, sh.dTᵢ, sh.dSᵢ, sh.du_wind_stress, sh.dv_wind_stress, sh.dT_flux, sh.dΔz, sh.dmld)

pr0, sh0 = build()
@info "Built $(pr0.model)"

@info "Compiling..."
const compile_t0 = time_ns()
const rdifferentiate_tracer_error = @compile optimize=OPTIMIZE raise_first=true raise=true sync=true differentiate_tracer_error(args(pr0, sh0)...)
const compile_s = (time_ns() - compile_t0) / 1e9
@info "compile time $(round(compile_s, digits=2)) s"

xla_flops = try
    Float64(Reactant.XLA.cost_analysis(rdifferentiate_tracer_error).flops)
catch e
    @warn "cost_analysis unavailable" exception=e
    nothing
end
pr0 = sh0 = nothing

run_protocol(;
    build,
    call      = (pr, sh) -> rdifferentiate_tracer_error(args(pr, sh)...),
    objective = y -> Float64(y[2]),
    interior  = Oceananigans.interior,
    opts      = OPTS,
    cfg       = CFG,
    driver    = "reactant",
    extra     = Dict("compile_s" => compile_s,
                     "xla_cost_analysis_flops" => xla_flops,
                     "reactant_options" => Dict("optimize" => string(OPTIMIZE), "mincut" => MINCUT,
                                                "checkpointing" => string(CHECKPOINTING))))
