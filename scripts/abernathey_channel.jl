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
using Oceananigans.Models: interior_tendency_kernel_parameters, surface_kernel_parameters

using Oceananigans.Fields: immersed_boundary_condition

using Oceananigans.Utils: launch!

using Oceananigans.TimeSteppers: update_state!

using SeawaterPolynomials

using CUDA

using Reactant
using Oceananigans.Architectures: ReactantState
#Reactant.set_default_backend("cpu")

using Enzyme

using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_tracer_tendencies!,
                                                        compute_hydrostatic_tracer_tendencies!,
                                                        compute_hydrostatic_free_surface_Gc!,
                                                        hydrostatic_free_surface_tracer_tendency,
                                                        update_vertical_velocities!,
                                                        compute_w_from_continuity!,
                                                        _compute_w_from_continuity!

Oceananigans.defaults.FloatType = Float64


using Oceananigans.Operators: flux_div_xyᶜᶜᶜ, Az⁻¹ᶜᶜᶜ, Δrᶜᶜᶜ, ∂t_σ
using Oceananigans.ImmersedBoundaries: immersed_cell

using KernelAbstractions: @kernel, @index

using InteractiveUtils

@info "To specify architecture uncomment line 'Reactant.set_default_backend(\"cpu\")' "
#Reactant.set_default_backend("cpu")

using Enzyme

#Oceananigans.defaults.FloatType = Float64

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

function wall_function(x, y)
    zonal = (x > 470kilometers) && (x < 530kilometers)
    gap   = (y < 400kilometers) || (y > 1000kilometers)
    return (Lz+1) * zonal * gap - Lz
end


function make_grid(architecture, Nx, Ny, Nz, z_faces)

    underlying_grid = RectilinearGrid(architecture;
        topology = (Periodic, Bounded, Bounded),
        size = (Nx, Ny, Nz),
        halo = (halo_size, halo_size, halo_size),
        x = (0, Lx),
        y = (0, Ly),
        z = z_faces)
        
    return underlying_grid
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

my_compute_w_from_continuity!(grid; parameters = surface_kernel_parameters(grid)) =
    launch!(grid.architecture, grid, parameters, _my_compute_w_from_continuity!, grid)


@kernel function _my_compute_w_from_continuity!(grid)
    i, j = @index(Global, NTuple)

    zero(eltype(grid))
end


@show eltype(grid)

parameters = Oceananigans.Utils.KernelParameters{(1, 1), (-3, -3)}()
my_compute_w_from_continuity!(grid; parameters)