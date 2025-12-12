using Oceananigans
using Printf
using Statistics

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, HorizontalFormulation

using SeawaterPolynomials

using CUDA
using CUDA: @allowscalar

#####
##### Model parameters to set first:
#####

# number of grid points
const Nx = 80  # LowRes: 48
const Ny = 160 # LowRes: 96
const Nz = 32

const x_midpoint = Int(Nx / 2) + 1

# stretched grid
k_center  = collect(1:Nz)
Δz_center = @. 10 * 1.104^(Nz - k_center)

const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]
const Lz = sum(Δz_center)

z_faces = vcat([-Lz], -Lz .+ cumsum(Δz_center))
z_faces[Nz+1] = 0

Δz = z_faces[2:end] - z_faces[1:end-1]

Δz = reshape(Δz, 1, :)

# Coriolis variables:
const f = -1e-4
const β = 1e-11

halo_size = 6 # 3 for non-immersed grid

# Other model parameters:
const α = 2e-4     # [K⁻¹] thermal expansion coefficient
const g = 9.8061   # [m/s²] gravitational constant
const cᵖ = 3994.0  # [J/K]  heat capacity
const ρ = 999.8    # [kg/m³] reference density

parameters = (
    Ly = Ly,
    Lz = Lz,
    Qᵇ = 10 / (ρ * cᵖ) * α * g,  # buoyancy flux magnitude [m² s⁻³]
    Qᵀ = 10 / (ρ * cᵖ),          # temperature flux magnitude
    y_shutoff = 5 / 6 * Ly,      # shutoff location for buoyancy flux [m]
    τ = 0.2 / ρ,                 # surface kinematic wind stress [m² s⁻²]
    μ = 1 / 30days,              # bottom drag damping time-scale [s⁻¹]
    ΔB = 8 * α * g,              # surface vertical buoyancy gradient [s⁻²]
    ΔT = 8,                      # surface vertical temperature gradient
    H = Lz,                      # domain depth [m]
    h = 1000.0,                  # exponential decay scale of stable stratification [m]
    y_sponge = 19 / 20 * Ly,     # southern boundary of sponge layer [m]
    λt = 7.0days                 # relaxation time scale [s]
)

# full ridge function:
function ridge_function(x, y)
    zonal = (Lz+3000)exp(-(x - Lx/2)^2/(1e6kilometers))
    gap   = 1 - 0.5(tanh((y - (Ly/6))/1e5) - tanh((y - (Ly/2))/1e5))
    return zonal * gap - Lz
end

function wall_function(x, y)
    zonal = (x > 470kilometers) && (x < 530kilometers)
    gap   = (y < 400kilometers) || (y > 1000kilometers)
    return (Lz+1) * zonal * gap - Lz
end

function make_grid(architecture, Nx, Ny, Nz, z_faces)

    underlying_grid = RectilinearGrid(architecture,
        topology = (Periodic, Bounded, Bounded),
        size = (Nx, Ny, Nz),
        halo = (halo_size, halo_size, halo_size),
        x = (0, Lx),
        y = (0, Ly),
        z = z_faces)

    # Make into a ridge array:
    ridge = Field{Center, Center, Nothing}(underlying_grid)
    set!(ridge, wall_function)
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(ridge))
    
    return grid
end

#####
##### Model construction:
#####

function build_model(grid, Δt₀, parameters)

    temperature_flux_bc = FluxBoundaryCondition(Field{Center, Center, Nothing}(grid))

    u_stress_bc = FluxBoundaryCondition(Field{Face, Center, Nothing}(grid))
    v_stress_bc = FluxBoundaryCondition(Field{Center, Face, Nothing}(grid))

    @inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * p.Lz * model_fields.u[i, j, 1]
    @inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * p.Lz * model_fields.v[i, j, 1]

    u_drag_bc = FluxBoundaryCondition(u_drag; discrete_form=true, parameters)
    v_drag_bc = FluxBoundaryCondition(v_drag; discrete_form=true, parameters)

    T_bcs = FieldBoundaryConditions(top=temperature_flux_bc)

    u_bcs = FieldBoundaryConditions(top=u_stress_bc, bottom=u_drag_bc)
    v_bcs = FieldBoundaryConditions(top=v_stress_bc, bottom=v_drag_bc)

    #####
    ##### Coriolis
    #####

    coriolis = BetaPlane(f₀ = f, β = β)

    #####
    ##### Forcing and initial condition
    #####
    @inline initial_temperature(z, p) = p.ΔT * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))
    @inline mask(y, p)                = max(0.0, y - p.y_sponge) / (Ly - p.y_sponge)

    @inline function temperature_relaxation(i, j, k, grid, clock, model_fields, p)
        timescale = p.λt
        y = ynode(j, grid, Center())
        z = znode(k, grid, Center())
        target_T = initial_temperature(z, p)
        T = @inbounds model_fields.T[i, j, k]
    
        return -1 / timescale * mask(y, p) * (T - target_T)
    end
    
    FT = Forcing(temperature_relaxation; discrete_form=true, parameters)

    # closure (moderately elevating scalar visc/diff)

    νh = 1e9  # [m²/s] horizontal viscocity
    κz = 5e-5 # [m²/s] vertical diffusivity
    νz = 3e-3 # [m²/s] vertical viscocity

    κz_field = Field{Center, Center, Center}(grid)
    κz_array = zeros(Nx, Ny, Nz)

    κz_add = 5e-5  # m² / s at surface
    decay_scale = 5   # layers
    for k in 1:Nz
        taper = exp(- (k-1) / decay_scale)
        κz_array[:,:,k] .= κz + κz_add * taper
    end
    @show κz_array[1:2,20,:]

    set!(κz_field, κz_array)

    horizontal_closure = HorizontalScalarBiharmonicDiffusivity(ν = νh)
    vertical_closure = VerticalScalarDiffusivity(ν = νz, κ = κz_field)

    @info "Building a model..."

    @allowscalar model = HydrostaticFreeSurfaceModel(
        grid = grid,
        free_surface = SplitExplicitFreeSurface(grid, substeps=30),
        momentum_advection = WENO(order=7),
        tracer_advection = WENO(order=7),
        buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(Oceananigans.defaults.FloatType)),
        coriolis = coriolis,
        closure = (horizontal_closure, vertical_closure),
        tracers = (:T, :S, :e),
        boundary_conditions = (T = T_bcs, u = u_bcs, v = v_bcs),
        forcing = (T = FT,)
    )

    model.clock.last_Δt = Δt₀

    return model
end

#####
##### Special initial and boundary conditions
#####

# Temperature flux:
function T_flux_init(grid, p)
    @inline temp_flux_function(x, y) = ifelse(y < p.y_shutoff, p.Qᵀ * cos(3π * y / p.Ly), 0.0)
    temp_flux = Field{Center, Center, Nothing}(grid)
    @allowscalar set!(temp_flux, temp_flux_function)
    return temp_flux
end

# wind stress:
function u_wind_stress_init(grid, p)
    @inline u_stress(x, y) = -p.τ * sin(π * y / p.Ly)
    wind_stress = Field{Face, Center, Nothing}(grid)
    @allowscalar set!(wind_stress, u_stress)
    return wind_stress
end

function v_wind_stress_init(grid, p)
    wind_stress = Field{Center, Face, Nothing}(grid)
    @allowscalar set!(wind_stress, 0)
    return wind_stress
end

# resting initial condition
function temperature_salinity_init(grid, parameters)
    # Adding some noise to temperature field:
    ε(σ) = σ * randn()
    Tᵢ_function(x, y, z) = parameters.ΔT * (exp(z / parameters.h) - exp(-Lz / parameters.h)) / (1 - exp(-Lz / parameters.h)) + ε(1e-8)
    Tᵢ = Field{Center, Center, Center}(grid)
    Sᵢ = Field{Center, Center, Center}(grid)
    @allowscalar set!(Tᵢ, Tᵢ_function)
    @allowscalar set!(Sᵢ, 35) # Initial Salinity
    return Tᵢ, Sᵢ
end

#####
##### Spin up (because step cound is hardcoded we need separate functions for each loop...)
#####

function spinup_loop!(model)
    Δt = model.clock.last_Δt
    for i = 1:100000000
        if mod(i, 100) == 0
            @info "step $i out of 100000000"
        end
        time_step!(model, Δt)
    end
    return nothing
end

function spinup_reentrant_channel_model!(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, temp_flux)
    # setting IC's and BC's:
    set!(model.velocities.u.boundary_conditions.top.condition, u_wind_stress)
    set!(model.velocities.v.boundary_conditions.top.condition, v_wind_stress)
    set!(model.tracers.T, Tᵢ)
    set!(model.tracers.S, Sᵢ)
    set!(model.tracers.T.boundary_conditions.top.condition, temp_flux)

    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0

    # Step it forward
    spinup_loop!(model)

    return nothing
end

#####
##### Actually creating our model and using these functions to run it:
#####

# Architecture
architecture = GPU()

# Timestep size:
Δt₀ = 10minutes 

# Make the grid:
grid          = make_grid(architecture, Nx, Ny, Nz, z_faces)
model         = build_model(grid, Δt₀, parameters)
T_flux        = T_flux_init(model.grid, parameters)
u_wind_stress = u_wind_stress_init(model.grid, parameters)
v_wind_stress = v_wind_stress_init(model.grid, parameters)
Tᵢ, Sᵢ        = temperature_salinity_init(model.grid, parameters)
mld           = Field{Center, Center, Nothing}(model.grid) # Not used for now

@info "Built $model."
@info "Running the simulation..."

# Spinup the model for a sufficient amount of time, save the T and S from this state:
tic = time()
spinup_reentrant_channel_model!(model, Tᵢ, Sᵢ, u_wind_stress, v_wind_stress, T_flux)
spinup_toc = time() - tic
@show spinup_toc