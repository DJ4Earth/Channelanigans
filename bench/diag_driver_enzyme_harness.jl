# Diagnostic copy of bench/driver_enzyme.jl, never used for table entries.
#
# Question: the August 2026 Enzyme-only script, run today on the same nodes with the same
# Julia 1.11.9 + Manifest, computes the same gradient 8-18% slower than
# bench/driver_enzyme.jl. Which of its differences causes that?
#
# hpc/vista/diag_enzyme_harness.slurm mounts this file over the image's
# bench/driver_enzyme.jl, so bench/run_cases.jl runs it as the enzyme_naive config.
# Two switches (default off = identical to bench/driver_enzyme.jl), each reproducing one
# difference of the August script:
#
#   --load-cuda on|off     `using CUDA` after Oceananigans, as the August script does. More
#                          methods on Oceananigans/KernelAbstractions functions can turn
#                          statically dispatched calls into dynamic ones.
#   --reuse-model on|off   build the model and shadows once and reuse them for every call,
#                          as the August script does (each call resets only T, S, the
#                          forcing and the clock; velocities, free surface, tendencies and
#                          the accumulated shadows carry over). The per-call objectives
#                          then differ, so run_protocol warns; that is expected here.
#   --zero-ics on|off      differentiate from T = 0, S = 0 instead of the stratified T
#                          (+ seeded noise) and S = 35. The August Enzyme-only script did
#                          this by accident: with its spinup call commented out,
#                          `set!(Tᵢ, model.tracers.T)` / `set!(Sᵢ, model.tracers.S)` copied
#                          the never-set (zero) tracers into the initial conditions.
#
# Every gradient call is also wrapped in @timed and logged as a line
#   harness call <k>: <s> s, gc <s> s, <GiB> GiB allocated, <n> allocations
# which the slurm script collects (the allocation count is the one @btime printed).

include(joinpath(@__DIR__, "common.jl"))

const OPTS = parse_driver_args(ARGS)
const CFG  = load_config(OPTS, "enzyme")

onoff(key) = let v = get(OPTS, key, "off")
    v in ("on", "off") || error("--$key must be on or off, got $(repr(v))")
    v == "on"
end
const LOAD_CUDA   = onoff("load-cuda")
const REUSE_MODEL = onoff("reuse-model")
const ZERO_ICS    = onoff("zero-ics")

using Oceananigans
LOAD_CUDA && @eval using CUDA
using Enzyme
using Checkpointing

# Needed for Enzyme to differentiate Oceananigans on CPU (as in the original script).
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.strictAliasing!(false)

const NSTEPS = OPTS["nsteps"]
const CHECKPOINT_MODE = CFG["checkpointing"]
CHECKPOINT_MODE == "none" || error("diag_driver_enzyme_harness.jl supports only checkpointing = \"none\" (enzyme_naive)")

@info "Enzyme harness diagnostic" config=OPTS["config"] nsteps=NSTEPS load_cuda=LOAD_CUDA reuse_model=REUSE_MODEL zero_ics=ZERO_ICS cuda_loaded=isdefined(Main, :CUDA)

#####
##### Differentiated function (as in bench/driver_enzyme.jl, checkpointing = "none")
#####

function loop!(model)
    Δt = model.clock.last_Δt
    for i = 1:NSTEPS
        time_step!(model, Δt)
    end
    return nothing
end

function estimate_tracer_error(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, temp_flux, Δz, mld)
    set_conditions!(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, temp_flux)
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
##### Measure
#####

fresh() = begin
    pr = build_problem(CPU(); seed = OPTS["seed"])
    if ZERO_ICS
        set!(pr.Tᵢ, 0)
        set!(pr.Sᵢ, 0)
    end
    sh = make_shadows(pr, Enzyme.make_zero)
    (pr, sh)
end
const SHARED = REUSE_MODEL ? fresh() : nothing
build() = REUSE_MODEL ? SHARED : fresh()

const NCALL = Ref(0)
function call(pr, sh)
    st = @timed differentiate_tracer_error(
        pr.model, pr.Tᵢ, pr.Sᵢ, pr.u_wind_stress, pr.v_wind_stress, pr.T_flux, pr.Δz, pr.mld,
        sh.dmodel, sh.dTᵢ, sh.dSᵢ, sh.du_wind_stress, sh.dv_wind_stress, sh.dT_flux, sh.dΔz, sh.dmld)
    NCALL[] += 1
    @info @sprintf("harness call %d: %.3f s, gc %.3f s, %.3f GiB allocated, %d allocations",
                   NCALL[], st.time, st.gctime, st.bytes / 2^30, Base.gc_alloc_count(st.gcstats))
    return st.value
end

run_protocol(;
    build, call,
    objective = y -> Float64(y[2]),
    interior  = Oceananigans.interior,
    opts      = OPTS,
    cfg       = CFG,
    driver    = "enzyme",
    extra     = Dict("enzyme_options" => Dict("checkpointing" => CHECKPOINT_MODE, "snaps" => nothing,
                                              "looseTypeAnalysis" => true, "strictAliasing" => false),
                     "harness_diagnostic" => Dict("load_cuda" => LOAD_CUDA, "reuse_model" => REUSE_MODEL,
                                                  "zero_ics" => ZERO_ICS,
                                                  "cuda_loaded" => isdefined(Main, :CUDA))))
