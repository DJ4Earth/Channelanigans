#using Pkg
# pkg"add Oceananigans CairoMakie"
using Oceananigans
ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Statistics

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, HorizontalFormulation, IsopycnalSkewSymmetricDiffusivity

using Oceananigans.Utils: get_active_cells_map
using Oceananigans.Models: interior_tendency_kernel_parameters

using Oceananigans.Fields: immersed_boundary_condition

using Oceananigans.Utils: launch!

using SeawaterPolynomials

using CUDA

using Reactant
using Oceananigans.Architectures: ReactantState
#Reactant.set_default_backend("cpu")

using Enzyme

using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_tracer_tendencies!,
                                                        compute_hydrostatic_tracer_tendencies!,
                                                        compute_hydrostatic_free_surface_Gc!,
                                                        hydrostatic_free_surface_tracer_tendency

Oceananigans.defaults.FloatType = Float64

using KernelAbstractions: @kernel, @index

@info "To specify architecture uncomment line 'Reactant.set_default_backend(\"cpu\")' "
#Reactant.set_default_backend("cpu")

using Enzyme

Oceananigans.defaults.FloatType = Float64

const Ntimesteps = 25 #100        # Number of timesteps in zonal transport computed / AD'ed part
const Nspinup    = 100 #10000        # Number of timesteps that the model is spun up

graph_directory = "run_abernathy_model_ad_spinup" * string(Nspinup) * "_" * string(Ntimesteps) * "steps_gmredi/"

#
# Model parameters to set first:
#

# number of grid points
const Nx = 80  # LowRes: 48
const Ny = 160 # LowRes: 96
const Nz = 32

const x_midpoint = Int(Nx / 2) + 1

# stretched grid
k_center = collect(1:Nz)
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

halo_size = 4 #3 for non-immersed grid

# Other model parameters:
const α = 2e-4     # [K⁻¹] thermal expansion coefficient
const g = 9.8061   # [m/s²] gravitational constant
const cᵖ = 3994.0   # [J/K]  heat capacity
const ρ = 999.8    # [kg/m³] reference density

parameters = (
    Ly = Ly,
    Lz = Lz,
    Qᵇ = 10 / (ρ * cᵖ) * α * g,            # buoyancy flux magnitude [m² s⁻³]
    Qᵀ = 10 / (ρ * cᵖ),                    # temperature flux magnitude
    y_shutoff = 5 / 6 * Ly,                # shutoff location for buoyancy flux [m]
    τ = 0.2 / ρ,                           # surface kinematic wind stress [m² s⁻²]
    μ = 1 / 30days,                      # bottom drag damping time-scale [s⁻¹]
    ΔB = 8 * α * g,                      # surface vertical buoyancy gradient [s⁻²]
    ΔT = 8,                              # surface vertical temperature gradient
    H = Lz,                              # domain depth [m]
    h = 1000.0,                          # exponential decay scale of stable stratification [m]
    y_sponge = 19 / 20 * Ly,               # southern boundary of sponge layer [m]
    λt = 7.0days                         # relaxation time scale [s]
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

    u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form = true, parameters = parameters)
    v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form = true, parameters = parameters)

    T_bcs = FieldBoundaryConditions(top = temperature_flux_bc)

    u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
    v_bcs = FieldBoundaryConditions(top = v_stress_bc, bottom = v_drag_bc)

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
    
    FT = Forcing(temperature_relaxation, discrete_form = true, parameters = parameters)

    # closure (moderately elevating scalar visc/diff)

    κ_skew_field      = Field{Center, Center, Center}(grid)
    κ_symmetric_field = Field{Center, Center, Center}(grid)

    @allowscalar set!(κ_skew_field, 1e3)
    @allowscalar set!(κ_symmetric_field, 1e3)

    gmredi_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=κ_skew_field, κ_symmetric=κ_symmetric_field)

    @info "Building a model..."

    model = HydrostaticFreeSurfaceModel(
        grid;
        free_surface = SplitExplicitFreeSurface(substeps=10),
        momentum_advection = nothing,
        tracer_advection = WENO(order=3),
        buoyancy = nothing,
        coriolis = nothing,
        closure = gmredi_closure,
        tracers = (:T, :S, :e),
    )

    model.clock.last_Δt = Δt₀

    return model
end

#####
##### Actually creating our model and using these functions to run it:
#####

# Architecture
architecture = ReactantState()

# Timestep size:
Δt₀ = 2.5minutes 

# Make the grid:
grid          = make_grid(architecture, Nx, Ny, Nz, z_faces)
model         = build_model(grid, Δt₀, parameters)

@info "Built $model."

function my_compute_hydrostatic_tracer_tendencies!(model, kernel_parameters; active_cells_map=nothing)

    arch = model.architecture
    grid = model.grid

    tracer_index = 1
    tracer_name = :T

    @show tracer_index, tracer_name

    @inbounds c_tendency    = model.timestepper.Gⁿ[tracer_name]
    @inbounds c_advection   = model.advection[tracer_name]
    @inbounds c_forcing     = model.forcing[tracer_name]
    @inbounds c_immersed_bc = immersed_boundary_condition(model.tracers[tracer_name])

    args = tuple(Val(tracer_index),
                    Val(tracer_name),
                    c_advection,
                    model.closure,
                    c_immersed_bc,
                    model.buoyancy,
                    model.biogeochemistry,
                    model.transport_velocities,
                    model.free_surface,
                    model.tracers,
                    model.closure_fields,
                    model.auxiliary_fields,
                    model.clock,
                    c_forcing)

    launch!(arch, grid, kernel_parameters,
            my_compute_hydrostatic_free_surface_Gc!,
            c_tendency,
            grid,
            args;
            active_cells_map)

    return nothing
end

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function my_compute_hydrostatic_free_surface_Gc!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end

# Trying zonal transport:

@info "Compiling the model run... (if you want to run forward code only, just compile 'estimate_tracer_error')"
tic = time()
rcompute_tracer_tendencies! = @compile raise_first=true raise=true sync=true  my_compute_hydrostatic_tracer_tendencies!(model, :xyz)
compile_toc = time() - tic

@show compile_toc

@info "Running the simulation..."


@info "Spinup the model for $Nspinup timesteps, save the T and S from this state:"
tic = time()
rcompute_tracer_tendencies!(model, :xyz)
spinup_toc = time() - tic
@show spinup_toc
