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
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, HorizontalFormulation

using SeawaterPolynomials

using CUDA

using Reactant
using Oceananigans.Architectures: ReactantState
#Reactant.set_default_backend("cpu")

using Enzyme

Oceananigans.defaults.FloatType = Float64

@info "To specify architecture uncomment line 'Reactant.set_default_backend(\"cpu\")' "
#Reactant.set_default_backend("cpu")

using Enzyme

Oceananigans.defaults.FloatType = Float64

const Ntimesteps = 25        # Number of timesteps in zonal transport computed / AD'ed part
const Nspinup    = 100        # Number of timesteps that the model is spun up

graph_directory = "run_abernathy_model_ad_spinup" * string(Nspinup) * "_" * string(Ntimesteps) * "steps/"

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


function make_grid(architecture, Nx, Ny, Nz, z_faces)

    underlying_grid = RectilinearGrid(architecture,
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
arch = ReactantState()

# Timestep size:
Δt₀ = 2.5minutes 

# Make the grid:
grid          = make_grid(arch, Nx, Ny, Nz, z_faces)

# Trying zonal transport:


function print_eltype!(grid)

    FT = eltype(grid)

    @show FT
    @show typeof(grid)

    return nothing
end

@show typeof(grid)

rprint_eltype! = @compile raise_first=true raise=true sync=true  print_eltype!(grid)


resolution = 3 // 2        # degrees
Nx = 360 ÷ resolution      # number of longitude points
Ny = 170 ÷ resolution      # number of latitude points (avoiding poles)
Nz = 10                    # number of vertical levels
size = (Nx, Ny, Nz)
halo = (7, 7, 7)           # halo size for higher-order advection schemes
H = 5000                   # domain depth [m]
latitude = (-85, 85)       # latitude range (avoiding poles for lat-lon grid)
longitude = (0, 360)       # longitude range
z = (-H, 0) 

lat_lon_grid = LatitudeLongitudeGrid(arch; size, halo, latitude, longitude, z)

rprint_eltype! = @compile raise_first=true raise=true sync=true  print_eltype!(lat_lon_grid)
