# Enzyme-only (LLVM-level AD, no Reactant) driver: one (config, N) case per process.
#
#   julia --project bench/driver_enzyme.jl --config enzyme_naive --nsteps 16
#   julia --project bench/driver_enzyme.jl --config enzyme_revolve_log2 --nsteps 16
#
# checkpointing = "none"     plain `for` loop; Enzyme stores the full tape
#               = "periodic" Checkpointing.jl `Periodic(snaps)`
#               = "revolve"  Checkpointing.jl `Revolve(snaps)` (binomial)
#
# The first call includes Julia/Enzyme JIT compilation (there is no separate
# compile step); it is the reference/warmup call and is not used for the timings.

include(joinpath(@__DIR__, "common.jl"))

const OPTS = parse_driver_args(ARGS)
const CFG  = load_config(OPTS, "enzyme")

using Oceananigans
using Enzyme
using Checkpointing

# Needed for Enzyme to differentiate Oceananigans on CPU (as in the original script).
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.strictAliasing!(false)

const NSTEPS = OPTS["nsteps"]
const CHECKPOINT_MODE = CFG["checkpointing"]
const SNAPS = CHECKPOINT_MODE == "none" ? nothing : resolve_snaps(CFG["snaps"], NSTEPS)

@info "Enzyme config" config=OPTS["config"] nsteps=NSTEPS checkpointing=CHECKPOINT_MODE snaps=SNAPS

# A fresh scheme object per call: Checkpointing.jl schemes carry mutable state
# (storage, Revolve counters).
make_scheme() =
    CHECKPOINT_MODE == "periodic" ? Checkpointing.Periodic(SNAPS) :
    CHECKPOINT_MODE == "revolve"  ? Checkpointing.Revolve(SNAPS) :
    error("unknown Enzyme checkpointing '$CHECKPOINT_MODE'")

#####
##### Differentiated function
#####

function loop_naive!(model, scheme)
    Δt = model.clock.last_Δt
    for i = 1:NSTEPS
        time_step!(model, Δt)
    end
    return nothing
end

# Checkpointing.jl requires the loop body not to capture floats alongside the mutable
# model (Enzyme activity analysis), so Δt is read through the struct. `time_step!`
# sets `clock.last_Δt = Δt`, so this is the same constant Δt every step.
function loop_checkpointed!(model, scheme)
    @ad_checkpoint scheme for i = 1:NSTEPS
        time_step!(model, model.clock.last_Δt)
    end
    return nothing
end

const loop! = CHECKPOINT_MODE == "none" ? loop_naive! : loop_checkpointed!

function estimate_tracer_error(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, temp_flux, Δz, mld, scheme)
    set_conditions!(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, temp_flux)
    loop!(model, scheme)
    return zonal_transport(model, Δz)
end

function differentiate_tracer_error(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, temp_flux, Δz, mld,
                                    dmodel, dTᵢ, dSᵢ, du_wind_stress, dv_wind_stress, dtemp_flux, dΔz, dmld,
                                    scheme)
    return autodiff(set_strong_zero(Enzyme.ReverseWithPrimal),
                    estimate_tracer_error, Active,
                    Duplicated(model, dmodel),
                    Duplicated(Tᵢ, dTᵢ),
                    Duplicated(Sᵢ, dSᵢ),
                    Duplicated(u_wind_stress, du_wind_stress),
                    Duplicated(v_wind_stress, dv_wind_stress),
                    Duplicated(temp_flux, dtemp_flux),
                    Duplicated(Δz, dΔz),
                    Duplicated(mld, dmld),
                    Const(scheme))
end

#####
##### Measure
#####

build() = begin
    pr = build_problem(CPU(); seed = OPTS["seed"])
    sh = make_shadows(pr, Enzyme.make_zero)
    (pr, sh)
end

call(pr, sh) = differentiate_tracer_error(
    pr.model, pr.Tᵢ, pr.Sᵢ, pr.u_wind_stress, pr.v_wind_stress, pr.T_flux, pr.Δz, pr.mld,
    sh.dmodel, sh.dTᵢ, sh.dSᵢ, sh.du_wind_stress, sh.dv_wind_stress, sh.dT_flux, sh.dΔz, sh.dmld,
    CHECKPOINT_MODE == "none" ? nothing : make_scheme())

run_protocol(;
    build, call,
    objective = y -> Float64(y[2]),
    interior  = Oceananigans.interior,
    opts      = OPTS,
    cfg       = CFG,
    driver    = "enzyme",
    extra     = Dict("enzyme_options" => Dict("checkpointing" => CHECKPOINT_MODE, "snaps" => SNAPS,
                                              "looseTypeAnalysis" => true, "strictAliasing" => false)))
