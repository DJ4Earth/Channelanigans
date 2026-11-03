"""
    ChannelModel

Shared definition of the differentiable Abernathey-style reentrant channel used by
every benchmark configuration (Reactant and Enzyme-only drivers alike), so that all
rows of the benchmark table differentiate exactly the same model.

The model setup is carried over verbatim from the original
`scripts/abernathey_channel.jl` / `scripts/abernathey_channel_no_reactant.jl`
(which were verified to be identical apart from `@allowscalar`). Deliberate changes:

- the 1e-8 initial-temperature noise is drawn from a seeded RNG, so every
  configuration starts from bit-identical initial conditions (needed to compare
  gradients across configurations);
- spinup, JLD2 field dumps, and plotting hooks are removed (spinup was always 0
  for the benchmark table).

The time-stepping loop itself is *not* defined here: each driver supplies its own
`loop!` (Reactant `@trace`, plain `for`, or Checkpointing.jl `@ad_checkpoint`).
"""
module ChannelModel

using Random
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.TurbulenceClosures: HorizontalFormulation

export build_problem, make_shadows, set_conditions!, zonal_transport,
       Nx, Ny, Nz, x_midpoint, Δt₀

Oceananigans.defaults.FloatType = Float64

#####
##### Parameters
#####

# number of grid points
const Nx = 80
const Ny = 160
const Nz = 32

const x_midpoint = Int(Nx / 2) + 1

# stretched grid
const k_center  = collect(1:Nz)
const Δz_center = @. 10 * 1.104^(Nz - k_center)

const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]
const Lz = sum(Δz_center)

const z_faces = let zf = vcat([-Lz], -Lz .+ cumsum(Δz_center))
    zf[Nz+1] = 0
    zf
end

# Layer thicknesses as a 1×Nz row, used by the zonal-transport objective
const Δz_row = reshape(z_faces[2:end] - z_faces[1:end-1], 1, :)

# Coriolis
const f = -1e-4
const β = 1e-11

const halo_size = 4

# Physical constants
const α  = 2e-4     # [K⁻¹] thermal expansion coefficient
const g  = 9.8061   # [m/s²] gravitational constant
const cᵖ = 3994.0   # [J/K]  heat capacity
const ρ  = 999.8    # [kg/m³] reference density

const parameters = (
    Ly = Ly,
    Lz = Lz,
    Qᵇ = 10 / (ρ * cᵖ) * α * g,   # buoyancy flux magnitude [m² s⁻³]
    Qᵀ = 10 / (ρ * cᵖ),           # temperature flux magnitude
    y_shutoff = 5 / 6 * Ly,       # shutoff location for buoyancy flux [m]
    τ = 0.2 / ρ,                  # surface kinematic wind stress [m² s⁻²]
    μ = 1 / 30days,               # bottom drag damping time-scale [s⁻¹]
    ΔB = 8 * α * g,               # surface vertical buoyancy gradient [s⁻²]
    ΔT = 8,                       # surface vertical temperature gradient
    H = Lz,                       # domain depth [m]
    h = 1000.0,                   # exponential decay scale of stable stratification [m]
    y_sponge = 19 / 20 * Ly,      # southern boundary of sponge layer [m]
    λt = 7.0days                  # relaxation time scale [s]
)

# Timestep size
const Δt₀ = 2.5minutes

#####
##### Grid
#####

function wall_function(x, y)
    zonal = (x > 470kilometers) && (x < 530kilometers)
    gap   = (y < 400kilometers) || (y > 1000kilometers)
    return (Lz+1) * zonal * gap - Lz
end

function make_grid(arch)
    underlying_grid = RectilinearGrid(arch,
        topology = (Periodic, Bounded, Bounded),
        size = (Nx, Ny, Nz),
        halo = (halo_size, halo_size, halo_size),
        x = (0, Lx),
        y = (0, Ly),
        z = z_faces)

    ridge = Field{Center, Center, Nothing}(underlying_grid)
    set!(ridge, wall_function)

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(ridge))
end

#####
##### Model
#####

@inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * p.Lz * model_fields.u[i, j, 1]
@inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * p.Lz * model_fields.v[i, j, 1]

@inline initial_temperature(z, p) = p.ΔT * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))
@inline mask(y, p) = max(0.0, y - p.y_sponge) / (Ly - p.y_sponge)

@inline function temperature_relaxation(i, j, k, grid, clock, model_fields, p)
    timescale = p.λt
    y = ynode(j, grid, Center())
    z = znode(k, grid, Center())
    target_T = initial_temperature(z, p)
    T = @inbounds model_fields.T[i, j, k]
    return -1 / timescale * mask(y, p) * (T - target_T)
end

"""
    build_model(grid; Δt=Δt₀, p=parameters, weno_weight_computation=nothing)

`weno_weight_computation = nothing` keeps Oceananigans' architecture default
(`BackendOptimizedDivision` on CPU, `ConvertingDivision{Float32}` on Reactant).
Pass e.g. `Oceananigans.Utils.ConvertingDivision{Float32}` to force identical WENO
arithmetic across backends.
"""
function build_model(grid; Δt = Δt₀, p = parameters, weno_weight_computation = nothing)
    temperature_flux_bc = FluxBoundaryCondition(Field{Center, Center, Nothing}(grid))
    u_stress_bc = FluxBoundaryCondition(Field{Face, Center, Nothing}(grid))
    v_stress_bc = FluxBoundaryCondition(Field{Center, Face, Nothing}(grid))

    u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form = true, parameters = p)
    v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form = true, parameters = p)

    T_bcs = FieldBoundaryConditions(top = temperature_flux_bc)
    u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
    v_bcs = FieldBoundaryConditions(top = v_stress_bc, bottom = v_drag_bc)

    coriolis = BetaPlane(f₀ = f, β = β)

    FT = Forcing(temperature_relaxation, discrete_form = true, parameters = p)

    # closure (moderately elevating scalar visc/diff)
    κh = 5e-5 # [m²/s] horizontal diffusivity
    νh = 500  # [m²/s] horizontal viscosity
    κz = 5e-5 # [m²/s] vertical diffusivity
    νz = 3e-3 # [m²/s] vertical viscosity

    κz_field = Field{Center, Center, Center}(grid)
    κz_array = zeros(Nx, Ny, Nz)
    κz_add = 5e-5    # m² / s at surface
    decay_scale = 5  # layers
    for k in 1:Nz
        taper = exp(- (k-1) / decay_scale)
        κz_array[:, :, k] .= κz + κz_add * taper
    end
    set!(κz_field, κz_array)

    horizontal_closure = HorizontalScalarDiffusivity(ν = νh, κ = κh)
    vertical_closure   = VerticalScalarDiffusivity(ν = νz, κ = κz_field)
    biharmonic_closure = ScalarBiharmonicDiffusivity(HorizontalFormulation(), Oceananigans.defaults.FloatType; ν = 1e11)

    weno() = weno_weight_computation === nothing ? WENO(order=3) :
                                                   WENO(order=3, weight_computation=weno_weight_computation)

    model = HydrostaticFreeSurfaceModel(
        grid;
        free_surface = SplitExplicitFreeSurface(substeps=10),
        momentum_advection = weno(),
        tracer_advection = weno(),
        buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(Oceananigans.defaults.FloatType)),
        coriolis = coriolis,
        closure = (horizontal_closure, vertical_closure, biharmonic_closure),
        tracers = (:T, :S, :e),
        boundary_conditions = (T = T_bcs, u = u_bcs, v = v_bcs),
        forcing = (T = FT,)
    )

    model.clock.last_Δt = Δt
    return model
end

#####
##### Forcing fields and initial conditions
#####

function T_flux_init(grid, p = parameters)
    @inline temp_flux_function(x, y) = ifelse(y < p.y_shutoff, p.Qᵀ * cos(3π * y / p.Ly), 0.0)
    temp_flux = Field{Center, Center, Nothing}(grid)
    @allowscalar set!(temp_flux, temp_flux_function)
    return temp_flux
end

function u_wind_stress_init(grid, p = parameters)
    @inline u_stress(x, y) = -p.τ * sin(π * y / p.Ly)
    wind_stress = Field{Face, Center, Nothing}(grid)
    @allowscalar set!(wind_stress, u_stress)
    return wind_stress
end

function v_wind_stress_init(grid, p = parameters)
    wind_stress = Field{Center, Face, Nothing}(grid)
    @allowscalar set!(wind_stress, 0)
    return wind_stress
end

# Resting, stratified initial condition plus small seeded noise.
function temperature_salinity_init(grid, p = parameters; seed = 1234)
    rng = Random.Xoshiro(seed)
    noise = 1e-8 .* randn(rng, Nx, Ny, Nz)
    z_centers = (z_faces[1:end-1] .+ z_faces[2:end]) ./ 2   # host-side, exact for this grid
    T_profile = [initial_temperature(z, p) for z in z_centers]
    T_array = reshape(T_profile, 1, 1, Nz) .+ noise

    Tᵢ = Field{Center, Center, Center}(grid)
    Sᵢ = Field{Center, Center, Center}(grid)
    @allowscalar set!(Tᵢ, T_array)
    @allowscalar set!(Sᵢ, 35) # initial salinity
    return Tᵢ, Sᵢ
end

#####
##### Problem assembly
#####

"""
    build_problem(arch; seed=1234, to_device=identity)

Build the model and all differentiated inputs. Returns a NamedTuple whose field
order matches the argument order of `estimate_tracer_error`:
`(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, T_flux, Δz, mld)`.
`to_device` converts the plain `Δz` array (e.g. `Reactant.ConcreteRArray`).
"""
function build_problem(arch; seed = 1234, to_device = identity, weno_weight_computation = nothing)
    grid          = make_grid(arch)
    model         = build_model(grid; weno_weight_computation)
    T_flux        = T_flux_init(model.grid)
    u_wind_stress = u_wind_stress_init(model.grid)
    v_wind_stress = v_wind_stress_init(model.grid)
    Tᵢ, Sᵢ        = temperature_salinity_init(model.grid; seed)
    mld           = Field{Center, Center, Nothing}(model.grid)
    Δz            = to_device(copy(Δz_row))
    return (; model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, T_flux, Δz, mld)
end

"""
    make_shadows(problem, make_zero)

Fresh, zeroed Enzyme shadows for every differentiated input, in the same order.
`make_zero` is `Enzyme.make_zero`, passed in so this module does not depend on Enzyme.
"""
function make_shadows(pr, make_zero)
    grid = pr.model.grid
    return (dmodel         = make_zero(pr.model),
            dTᵢ            = Field{Center, Center, Center}(grid),
            dSᵢ            = Field{Center, Center, Center}(grid),
            du_wind_stress = Field{Face, Center, Nothing}(grid),
            dv_wind_stress = Field{Center, Face, Nothing}(grid),
            dT_flux        = Field{Center, Center, Nothing}(grid),
            dΔz            = make_zero(pr.Δz),
            dmld           = Field{Center, Center, Nothing}(grid))
end

#####
##### Pieces of the differentiated function (loop supplied by drivers)
#####

function set_conditions!(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, temp_flux)
    set!(model.velocities.u.boundary_conditions.top.condition, u_wind_stress)
    set!(model.velocities.v.boundary_conditions.top.condition, v_wind_stress)
    set!(model.tracers.T, Tᵢ)
    set!(model.tracers.S, Sᵢ)
    set!(model.tracers.T.boundary_conditions.top.condition, temp_flux)
    model.clock.iteration = 0
    model.clock.time = 0
    return nothing
end

# Zonal transport through the mid-channel meridional section, in Sverdrups.
function zonal_transport(model, Δz)
    Nx, Ny, Nz = size(model.grid)
    transport = (model.velocities.u[x_midpoint, 1:Ny, 1:Nz] .* model.grid.Δyᵃᶜᵃ) .* Δz
    return sum(transport) / 1e6
end

end # module
