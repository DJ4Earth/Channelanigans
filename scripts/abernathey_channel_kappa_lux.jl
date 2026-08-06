# abernathey_channel_kappa_lux.jl
#
# Variant of abernathey_channel_kappa_dirfd.jl in which the GM/Redi coefficients
# are produced by a small Lux MLP instead of being a directly-differentiated
# field. The structural delta:
#
#   before:  κᵢ::Field  ── set! ──▶  closure κ fields          (leaf: Duplicated(κᵢ, dκᵢ))
#   after :  ps::NamedTuple ─▶ NN(features(T)) ─▶ closure κ    (leaf: Duplicated(ps, dps))
#
# The κ fields are thereby demoted from parameters to intermediate values; the
# NN evaluation (update_gmredi_κ!) sits INSIDE the differentiated function, in
# the exact slot the `set!(closure.κ, κᵢ)` used to occupy, so Enzyme records
# loss → dynamics → κ → exp/tanh → matmuls → ps on the same tape as before.
# The directional FD check is correspondingly moved from κ-space to weight
# space via Optimisers.destructure. Everything else is untouched.
#
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

using SeawaterPolynomials

using CUDA

using Reactant
using Oceananigans.Architectures: ReactantState
#Reactant.set_default_backend("cpu")

using Enzyme

# --- NN additions -----------------------------------------------------------
using Lux                     # explicit-parameter NN (ps is a NamedTuple of arrays)
using Random                  # Xoshiro rngs for reproducible init / FD directions
using Optimisers              # Adam for the training loop; destructure for the FD check
using Functors: fmap          # tree-map over the Lux parameter NamedTuple
using LinearAlgebra: dot, norm
# ----------------------------------------------------------------------------

Oceananigans.defaults.FloatType = Float64

@info "To specify architecture uncomment line 'Reactant.set_default_backend(\"cpu\")' "
#Reactant.set_default_backend("cpu")

const Ntimesteps = 25        # Number of timesteps in the AD'ed window
const Nspinup    = 100        # Number of timesteps that the model is spun up

graph_directory = "run_abernathy_model_ad_spinup" * string(Nspinup) * "_" * string(Ntimesteps) * "steps/"

# GRADIENT-VERIFICATION MODE:
# Centered(order=2) advection is Float64 end-to-end and smooth. WENO computes its
# smoothness-indicator divisions in Float32 (see ConvertingDivision{Float32} in the
# model type), which puts a ~1e-7 relative noise floor on the primal and destroys
# the FD arm of the comparison. Set false to restore WENO for production runs.
const use_smooth_advection = true

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

const halo_size = 4 #3 for non-immersed grid

# Other model parameters:
const α = 2e-4     # [K⁻¹] thermal expansion coefficient
const g = 9.8061   # [m/s²] gravitational constant
const cᵖ = 3994.0   # [J/K]  heat capacity
const ρ = 999.8    # [kg/m³] reference density

# Baseline GM/Redi coefficient (spinup value, and the scale the NN multiplies) [m²/s]:
const κ_gm_background = 1e3

#####
##### Lux neural network for the GM/Redi coefficients
#####
# Pointwise MLP mapping local, nondimensional features ξ to log-multipliers on
# the background κ:
#
#     (κ_skew, κ_symmetric) = κ_gm_background · exp( s · tanh(NN(ξ)/s) )
#
# Properties this buys:
#   * κ > 0 always (exp), smoothly bounded in κ_bg·e^±s (tanh soft-clip, so the
#     optimizer can't request CFL-breaking κ while exploring, and — unlike
#     clamp — the gradient never hard-zeros);
#   * zero-initialized last layer ⇒ NN output ≡ 0 ⇒ κ ≡ κ_gm_background: the
#     untrained model reproduces the constant-κ baseline run EXACTLY. This is
#     milestone (a) for free.
#   * Corollary of zero-init: at the very first gradient call, dps of the
#     hidden layers is exactly zero (the adjoint passes through the zero last-
#     layer weights); only the last layer receives gradient, which unlocks the
#     hidden layers on subsequent steps. Do not be alarmed by zeros in
#     dps.layer_1 / dps.layer_2 at iteration 1. For FD gradient verification,
#     set zero_init_last_layer = false so every layer participates.

const nfeatures   = 6     # |S|, 1/Ri, N²/N²_ref, L_d/Δx, z/H, y-wall distance
const nn_width    = 16
const κ_log_bound = 3.0   # κ confined to κ_bg·e^±3 ≈ [5e1, 2e4] m²/s

const zero_init_last_layer = false   # ← false for FD gradient-verification runs

nn_zero_init(rng, dims...) = zeros(Oceananigans.defaults.FloatType, dims...)

const nn = Chain(Dense(nfeatures => nn_width, tanh),
                 Dense(nn_width => nn_width, tanh),
                 zero_init_last_layer ?
                     Dense(nn_width => 2; init_weight = nn_zero_init,
                                          init_bias   = nn_zero_init) :
                     Dense(nn_width => 2))
# `nn` holds no numbers — it is pure architecture (an immutable struct), so it
# is safe as a global const closed over by the traced functions. All numbers
# live in `ps`, which is what we differentiate.

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
    ΔT_front = 4.0,                      # meridional front amplitude [K] (half-range of tanh)
    L_front = 200kilometers,             # meridional front width [m]
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

    κh = 5e-5 # [m²/s] horizontal diffusivity
    νh = 500  # [m²/s] horizontal viscocity
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

    set!(κz_field, κz_array)

    horizontal_closure = HorizontalScalarDiffusivity(ν = νh, κ = κh)
    vertical_closure = VerticalScalarDiffusivity(ν = νz, κ = κz_field)

    biharmonic_closure = ScalarBiharmonicDiffusivity(HorizontalFormulation(), Oceananigans.defaults.FloatType;
                                                     ν = 1e11)

    # These fields are overwritten by update_gmredi_κ! (the NN) inside
    # run_reentrant_channel_model! — the values set here only matter for the
    # spinup phase, which runs with the constant background κ.
    κ_skew_field      = Field{Center, Center, Center}(grid)
    κ_symmetric_field = Field{Center, Center, Center}(grid)

    @allowscalar set!(κ_skew_field, κ_gm_background)
    @allowscalar set!(κ_symmetric_field, κ_gm_background)

    gmredi_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=κ_skew_field, κ_symmetric=κ_symmetric_field)

    # Smooth, fully-Float64 advection for gradient verification; WENO for production.
    advection = use_smooth_advection ? Centered(order=2) : WENO(order=3)

    @info "Building a model..."

    model = HydrostaticFreeSurfaceModel(
        grid;
        free_surface = SplitExplicitFreeSurface(substeps=10),
        momentum_advection = advection,
        tracer_advection = advection,
        buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(Oceananigans.defaults.FloatType)),
        coriolis = coriolis,
        closure = (horizontal_closure, vertical_closure, gmredi_closure),
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

# Initial condition: exponential stratification PLUS a surface-intensified
# meridional front. The front tilts the isopycnals so GM/Redi has something to
# act on from the very first timestep:
#   slope ≈ ∂yT / ∂zT ≈ (ΔT_front / 2 L_front) / (ΔT / h) ~ O(10⁻³),
# which is comfortably below the default FluxTapering max slope (10⁻²), so the
# fluxes are untapered and genuinely κ-proportional.
function temperature_salinity_init(grid, p)
    # Adding some noise to temperature field:
    ε(σ) = σ * randn()
    stratification(z) = p.ΔT * (exp(z / p.h) - exp(-Lz / p.h)) / (1 - exp(-Lz / p.h))
    front(y, z)       = 0.5 * p.ΔT_front * tanh((y - p.Ly / 2) / p.L_front) * exp(z / p.h)
    Tᵢ_function(x, y, z) = stratification(z) + front(y, z) + ε(1e-8)
    Tᵢ = Field{Center, Center, Center}(grid)
    Sᵢ = Field{Center, Center, Center}(grid)
    @allowscalar set!(Tᵢ, Tᵢ_function)
    @allowscalar set!(Sᵢ, 35) # Initial Salinity
    return Tᵢ, Sᵢ
end

#####
##### NN features and the NN → κ update (successor of the κᵢ pathway)
#####
# Every feature is local and nondimensional, so the learned map ξ → κ/κ_bg is a
# statement about the flow, not about this grid, domain, or unit system.
# Derivatives are slicing-based finite differences: they trace cleanly under
# Reactant (unlike scalar loops / set! with functions) and they are
# feature-grade approximations — NOT staggered-grid-correct operators. Fine for
# O(1) NN inputs; do not reuse for dynamics.

# Constant feature ingredients, built host-side ONCE and moved to the device.
# Full (Nx,Ny,Nz) arrays cost ~3 MB each — negligible, and they keep
# compute_features to pure broadcasts on same-shaped arrays.
function build_feature_constants()
    zc  = 0.5 .* (z_faces[1:Nz] .+ z_faces[2:Nz+1])       # cell-center depths (Nz,)
    Δzb = reshape(zc[2:Nz] .- zc[1:Nz-1], 1, 1, Nz - 1)   # center-to-center spacing

    yc = (collect(1:Ny) .- 0.5) .* (Ly / Ny)              # cell-center y (Ny,)
    fℓ = abs.(f .+ β .* yc)                               # |f| on the β-plane

    fullz(v) = ones(Nx, Ny, 1) .* reshape(v, 1, 1, Nz)
    fully(v) = ones(Nx, 1, Nz) .* reshape(v, 1, Ny, 1)

    return (
        Δx           = Lx / Nx,
        Δy           = Ly / Ny,
        Δz_between   = Reactant.to_rarray(Δzb),
        f²           = Reactant.to_rarray(fully(fℓ .^ 2)),
        N²_floor     = 1e-9,                                       # [s⁻²] clip for divisions
        N²_ref       = α * g * parameters.ΔT / parameters.h,       # initial pycnocline N²
        Ld_prefactor = Reactant.to_rarray(fully(Lz ./ (π .* fℓ .* (Lx / Nx)))),  # · N → L_d/Δx
        z_over_H     = Reactant.to_rarray(fullz(-zc ./ Lz)),       # fractional depth ∈ (0,1)
        y_dist       = Reactant.to_rarray(fully(min.(yc, Ly .- yc) ./ Ly)),      # wall distance
    )
end

function compute_features(model, fc)
    T = model.tracers.T[1:Nx, 1:Ny, 1:Nz]     # same traced-indexing idiom as the objective
    b = (α * g) .* T   # linear-EOS buoyancy, thermal part (S is uniform here; switch to
                       # full-EOS buoyancy once S is dynamically active / TEOS10 is on)

    # ∂b/∂z between cell centers on the stretched grid, edge-padded at the bottom (k = 1):
    dbdz = (b[:, :, 2:Nz] .- b[:, :, 1:Nz-1]) ./ fc.Δz_between
    dbdz = cat(dbdz[:, :, 1:1], dbdz; dims = 3)
    N²   = max.(dbdz, fc.N²_floor)

    # ∂b/∂x (x periodic: wrap by slicing + cat) and ∂b/∂y (edge-padded at the walls):
    b_e  = cat(b[2:Nx, :, :],  b[1:1, :, :];    dims = 1)
    b_w  = cat(b[Nx:Nx, :, :], b[1:Nx-1, :, :]; dims = 1)
    dbdx = (b_e .- b_w) ./ (2 * fc.Δx)
    dbdy = (b[:, 2:Ny, :] .- b[:, 1:Ny-1, :]) ./ fc.Δy
    dbdy = cat(dbdy[:, 1:1, :], dbdy; dims = 2)

    M² = sqrt.(dbdx .^ 2 .+ dbdy .^ 2)

    slope = M² ./ N²                       # isopycnal slope magnitude |S|
    invRi = (M² .^ 2) ./ (N² .* fc.f²)     # thermal-wind 1/Ri = M⁴/(N²f²)
    N²rel = N² ./ fc.N²_ref                # vertical structure (Ferreira et al. 2005)
    Ld_dx = sqrt.(N²) .* fc.Ld_prefactor   # deformation radius / Δx (local-N proxy)

    # Normalize to O(1) with fixed, hand-chosen scales (fixed ⇒ part of the model
    # definition, not data-dependent), then stack into Lux's (features × batch):
    X = vcat(reshape(slope ./ 1e-3, 1, :),     # front slope is O(1e-3) by construction
             reshape(invRi ./ 1e-2, 1, :),
             reshape(N²rel,         1, :),
             reshape(Ld_dx ./ 3.0,  1, :),
             reshape(fc.z_over_H,   1, :),
             reshape(fc.y_dist,     1, :))
    return X
end

# Successor of `set!(closure.κ, κᵢ)`: evaluate the NN on the CURRENT model state
# and write the resulting coefficient fields into the ISSD closure. Because this
# is called inside the differentiated function, Enzyme carries the closure-field
# adjoints back through reshape/exp/tanh/matmuls into dps — and, via the
# features, into the state the features were computed from.
function update_gmredi_κ!(model, ps, st, fc)
    X = compute_features(model, fc)
    Y, _ = nn(X, ps, st)                  # (2 × Nx·Ny·Nz); st is empty for a plain MLP

    logμ_skew = κ_log_bound .* tanh.(Y[1, :] ./ κ_log_bound)
    logμ_symm = κ_log_bound .* tanh.(Y[2, :] ./ κ_log_bound)

    κ_skew_new = κ_gm_background .* exp.(reshape(logμ_skew, Nx, Ny, Nz))
    κ_symm_new = κ_gm_background .* exp.(reshape(logμ_symm, Nx, Ny, Nz))

    set!(model.closure[3].κ_skew,      κ_skew_new)
    set!(model.closure[3].κ_symmetric, κ_symm_new)
    # VERIFY: set!(field, ::Array) fills the interior. If ISSD's face
    # interpolation of κ reads halo points (x is periodic, so i = 1 does!), the
    # halos still hold the build-time background value. If gradients look wrong
    # only near i = 1/Nx, add explicit halo fills here:
    #   using Oceananigans.BoundaryConditions: fill_halo_regions!
    #   fill_halo_regions!(model.closure[3].κ_skew)
    #   fill_halo_regions!(model.closure[3].κ_symmetric)
    return nothing
end

#####
##### Spin up (because step cound is hardcoded we need separate functions for each loop...)
#####

function spinup_loop!(model)
    Δt = model.clock.last_Δt
    @trace mincut = true track_numbers = false for i = 1:Nspinup
        time_step!(model, Δt)
    end
    return nothing
end

function spinup_reentrant_channel_model!(model, Tᵢ, Sᵢ, ps, st, fconst, u_wind_stress, v_wind_stress, temp_flux)
    # setting IC's and BC's:
    set!(model.velocities.u.boundary_conditions.top.condition, u_wind_stress)
    set!(model.velocities.v.boundary_conditions.top.condition, v_wind_stress)
    set!(model.tracers.T, Tᵢ)
    set!(model.tracers.S, Sᵢ)
    set!(model.tracers.T.boundary_conditions.top.condition, temp_flux)
    update_gmredi_κ!(model, ps, st, fconst)

    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0

    # Step it forward
    spinup_loop!(model)

    return nothing
end

#####
##### Forward simulation (not actually using the Simulation struct)
#####

function loop!(model)
    Δt = model.clock.last_Δt
    @trace mincut = true checkpointing = true track_numbers = false for i = 1:Ntimesteps
        time_step!(model, Δt)
        #update_gmredi_κ!(model, ps, st, fconst)
    end
    return nothing
end

function run_reentrant_channel_model!(model, Tᵢ, Sᵢ, ps, st, fconst, u_wind_stress, v_wind_stress, temp_flux)
    # setting IC's and BC's:
    set!(model.velocities.u.boundary_conditions.top.condition, u_wind_stress)
    set!(model.velocities.v.boundary_conditions.top.condition, v_wind_stress)
    set!(model.tracers.T, Tᵢ)
    set!(model.tracers.S, Sᵢ)
    set!(model.tracers.T.boundary_conditions.top.condition, temp_flux)

    # NN → κ update replaces `set!(model.closure[3].κ_*, κᵢ)`. It must come
    # AFTER the tracer set!s (the features read T) and BEFORE the loop. κ is
    # then frozen for this window ("quasi-static"): computed from the window's
    # initial state, held fixed for Ntimesteps. Phase two — responsive κ — moves
    # this call inside loop! gated on `i % M == 0`, which puts the NN inside the
    # checkpointed @trace region; defer that until this version's gradients are
    # verified.
    update_gmredi_κ!(model, ps, st, fconst)

    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0

    # Step it forward
    loop!(model)

    return nothing
end

# VERIFICATION OBJECTIVE: total squared tracer drift over the window,
#   J = Σᵢⱼₖ (T(t_end) - Tᵢ)².
# GM/Redi modifies T at every cell within a couple of timesteps, so this
# objective has O(1) sensitivity to κ even over a 25-step window — unlike the
# midline zonal transport, which a local κ perturbation cannot reach in one
# hour of model time. (The transport objective is kept below, commented, for
# production use once the gradient is verified.)
function estimate_tracer_error(model, initial_temperature, initial_salinity, ps, st, fconst, u_wind_stress, v_wind_stress, temp_flux, Δz, mld)
    run_reentrant_channel_model!(model, initial_temperature, initial_salinity, ps, st, fconst, u_wind_stress, v_wind_stress, temp_flux)

    Nx, Ny, Nz = size(model.grid)

    T_end  = model.tracers.T[1:Nx, 1:Ny, 1:Nz]
    T_init = initial_temperature[1:Nx, 1:Ny, 1:Nz]

    return sum(abs2, T_end .- T_init)

    # Production objective (zonal transport in Sv):
    # zonal_transport = (model.velocities.u[x_midpoint,1:Ny,1:Nz] .* model.grid.Δyᵃᶜᵃ) .* Δz
    # return sum(zonal_transport) / 1e6
end

function differentiate_tracer_error(model, Tᵢ, Sᵢ, ps, st, fconst, u_wind_stress, v_wind_stress, temp_flux, Δz, mld,
                                   dmodel, dTᵢ, dSᵢ, dps, du_wind_stress, dv_wind_stress, dtemp_flux, dΔz, dmld)

    dedν = autodiff(set_strong_zero(Enzyme.ReverseWithPrimal),
                    estimate_tracer_error, Active,
                    Duplicated(model, dmodel),
                    Duplicated(Tᵢ, dTᵢ),
                    Duplicated(Sᵢ, dSᵢ),
                    Duplicated(ps, dps),    # ← the new differentiable leaf: ∂J/∂weights lands in dps
                    Const(st),              # empty MLP state — pure architecture bookkeeping
                    Const(fconst),          # precomputed constant feature arrays
                    Duplicated(u_wind_stress, du_wind_stress),
                    Duplicated(v_wind_stress, dv_wind_stress),
                    Duplicated(temp_flux, dtemp_flux),
                    Duplicated(Δz, dΔz),
                    Duplicated(mld, dmld))

    return dedν
end

#####
##### Directional-derivative check of dps (NN-weight space)
#####

# Same idea as the old κ-field check, but the direction now lives in the flat
# vector of network weights:  J(ps + εv) ≈ J(ps) + ε ⟨∇J, v⟩.
# Optimisers.destructure provides the flat view of the parameter tree and the
# inverse map back to the NamedTuple structure.
#
# Each FD evaluation replays the AD baseline exactly: rebuild the model, rerun
# the compiled spinup from the pre-spinup ICs (Tᵢ₀, Sᵢ₀), then run the estimate
# window with weights ps + δ v. Only 2 evaluations per epsilon.
#
# IMPORTANT: run this with zero_init_last_layer = false. With the zero-init
# last layer, the hidden-layer components of both dps AND the FD response are
# structurally zero, so the check would only exercise ~1/10 of the weights.

function fd_directional_gradient_check(rspinup!, restimate,
                                       grid, Δt₀, parameters,
                                       Tᵢ₀, Sᵢ₀,      # pre-spinup ICs (deterministic replay of the spinup)
                                       Tᵢ, Sᵢ,        # post-spinup ICs (arguments of the AD'd estimate call)
                                       u_wind_stress, v_wind_stress, T_flux, Δz, mld,
                                       ps_host,       # host copy of the weights the AD gradient was taken at
                                       st, fconst,
                                       dps;           # AD gradient tree (device)
                                       epsilon_range = (1e0, 1e-1, 1e-2, 1e-3))

    flat₀, restruct = Optimisers.destructure(ps_host)

    # Fixed random direction in weight space, unit-normalized:
    v_flat = randn(Xoshiro(7), length(flat₀))
    v_flat ./= norm(v_flat)

    ps    = Reactant.to_rarray(restruct(flat₀))

    # Pull the AD gradient tree to the host, flatten identically, dot with v:
    dps_host  = fmap(x -> x isa AbstractArray ? Array(x) : x, dps)
    g_flat, _ = Optimisers.destructure(dps_host)
    ad_dot    = dot(g_flat, v_flat)

    @info @sprintf("max |dps| = %.6e over %d parameters", maximum(abs, g_flat), length(g_flat))
    @info @sprintf("AD directional derivative ⟨∇J, v⟩ = %+.12e", ad_dot)

    function perturbed_estimate(δ)
        model_fd = build_model(grid, Δt₀, parameters)

        rspinup!(model_fd, Tᵢ₀, Sᵢ₀, ps, st, fconst, u_wind_stress, v_wind_stress, T_flux)

        ps_fd    = Reactant.to_rarray(restruct(flat₀ .+ δ .* v_flat))

        return restimate(model_fd, Tᵢ, Sᵢ, ps_fd, st, fconst, u_wind_stress, v_wind_stress, T_flux, Δz, mld)
    end

    for epsilon in epsilon_range
        outputP = perturbed_estimate(+epsilon)
        outputM = perturbed_estimate(-epsilon)

        fd_dot  = (outputP - outputM) / (2epsilon)
        rel_err = abs(fd_dot - ad_dot) / max(abs(fd_dot), abs(ad_dot), eps(Float64))

        @info @sprintf("    eps = %8.1e   FD = %+.12e   AD = %+.12e   rel. err = %.3e", epsilon, fd_dot, ad_dot, rel_err)
        @info @sprintf("        (outputP = %+.16e, outputM = %+.16e)", outputP, outputM)
    end

    return nothing
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
T_flux        = T_flux_init(model.grid, parameters)
u_wind_stress = u_wind_stress_init(model.grid, parameters)
v_wind_stress = v_wind_stress_init(model.grid, parameters)
Tᵢ, Sᵢ        = temperature_salinity_init(model.grid, parameters)
mld           = Field{Center, Center, Nothing}(model.grid)
Δz            = Reactant.ConcreteRArray(Δz)

# Lux parameters. Lux defaults to Float32 — promote to the model's float type
# before anything touches Enzyme/Reactant, or you'll chase mixed-precision
# ghosts. ps_host stays canonical for the optimizer & FD check; the device
# copy `ps` is what enters the compiled/differentiated calls.
rng = Xoshiro(1234)
ps_host, st = Lux.setup(rng, nn)
ps_host = fmap(x -> x isa AbstractArray ? Oceananigans.defaults.FloatType.(x) : x, ps_host)
ps = Reactant.to_rarray(ps_host)

# Precomputed constant feature ingredients (device-resident):
fconst = build_feature_constants()

# Keep a copy of the *pre-spinup* ICs so the FD check can replay the spinup
# deterministically (Tᵢ/Sᵢ get overwritten with the spun-up state below):
Tᵢ₀ = Field{Center, Center, Center}(model.grid)
Sᵢ₀ = Field{Center, Center, Center}(model.grid)
set!(Tᵢ₀, Tᵢ)
set!(Sᵢ₀, Sᵢ)

@info "Built $model."

dmodel         = Enzyme.make_zero(model)
dTᵢ            = Field{Center, Center, Center}(model.grid)
dSᵢ            = Field{Center, Center, Center}(model.grid)
dps            = Enzyme.make_zero(ps)   # shadow tree, same structure as ps
du_wind_stress = Field{Face, Center, Nothing}(model.grid)
dv_wind_stress = Field{Center, Face, Nothing}(model.grid)
dT_flux        = Field{Center, Center, Nothing}(model.grid)
dmld           = Field{Center, Center, Nothing}(model.grid)
dΔz            = Enzyme.make_zero(Δz)

@allowscalar @show typeof(model.closure[3].κ_skew)
@allowscalar @show typeof(model.closure[3].κ_symmetric)


@info "Compiling the model run... (forward 'restimate_tracer_error' is needed for the FD check)"
tic = time()
rspinup_reentrant_channel_model! = @compile raise_first=true raise=true sync=true  spinup_reentrant_channel_model!(model, Tᵢ, Sᵢ, ps, st, fconst, u_wind_stress, v_wind_stress, T_flux)
restimate_tracer_error = @compile raise_first=true raise=true sync=true estimate_tracer_error(model, Tᵢ, Sᵢ, ps, st, fconst, u_wind_stress, v_wind_stress, T_flux, Δz, mld)
rdifferentiate_tracer_error = @compile raise_first=true raise=true sync=true  differentiate_tracer_error(model, Tᵢ, Sᵢ, ps, st, fconst, u_wind_stress, v_wind_stress, T_flux, Δz, mld,
                                                                                                        dmodel, dTᵢ, dSᵢ, dps, du_wind_stress, dv_wind_stress, dT_flux, dΔz, dmld)
compile_toc = time() - tic

@show compile_toc

@info "Running the simulation..."

using FileIO, JLD2

filename = graph_directory * "data_init.jld2"

if !isdir(graph_directory) Base.Filesystem.mkdir(graph_directory) end

if isa(model.grid, ImmersedBoundaryGrid)
    bottom_height = model.grid.immersed_boundary.bottom_height
else
    bottom_height = Field{Center, Center, Nothing}(model.grid)
    set!(bottom_height, -Lz)
end

@info "Spinup the model for $Nspinup timesteps, save the T and S from this state:"
# The spinup runs with the build-time background κ — the NN only takes over
# inside the estimate window. (Milestone (a): with zero_init_last_layer = true,
# restimate_tracer_error must reproduce the old constant-κ script's output.)
tic = time()
rspinup_reentrant_channel_model!(model, Tᵢ, Sᵢ, ps, st, fconst, u_wind_stress, v_wind_stress, T_flux)
@allowscalar set!(Tᵢ, model.tracers.T)
@allowscalar set!(Sᵢ, model.tracers.S)
spinup_toc = time() - tic
@show spinup_toc

@info "Computing the AD gradient (dps is accumulated in-place):"
dedν = rdifferentiate_tracer_error(model, Tᵢ, Sᵢ, ps, st, fconst, u_wind_stress, v_wind_stress, T_flux, Δz, mld,
                                   dmodel, dTᵢ, dSᵢ, dps, du_wind_stress, dv_wind_stress, dT_flux, dΔz, dmld)

#
# Directional-derivative FD comparison in weight space
# (set zero_init_last_layer = false for this — see note at the check):
#
fd_directional_gradient_check(rspinup_reentrant_channel_model!, restimate_tracer_error,
                              grid, Δt₀, parameters,
                              Tᵢ₀, Sᵢ₀, Tᵢ, Sᵢ,
                              u_wind_stress, v_wind_stress, T_flux, Δz, mld,
                              ps_host, st, fconst, dps;
                              epsilon_range = (1e0, 1e-1, 1e-2, 1e-3))

#
# Training-loop skeleton (milestone (c): overfit a toy target first, e.g. a
# stored run with κ = 2e3, before pointing this at real coarse-grained truth).
# Adam runs on the host copy; a fresh device copy + fresh zero shadows go in
# each epoch. Enzyme ACCUMULATES (+=) into shadows — reusing un-zeroed shadows
# silently sums gradients across epochs. Re-created ConcreteRArrays of the
# same shapes are fine to pass to the compiled function; only shapes/types are
# baked in at @compile time, so no recompilation per epoch.
#
# zero_shadows!(tree) = fmap(x -> x isa AbstractArray ? (x .= 0; x) : x, tree)
#
# opt_state = Optimisers.setup(Optimisers.Adam(1e-3), ps_host)
# for epoch in 1:nepochs
#     ps  = Reactant.to_rarray(ps_host)
#     dps = Enzyme.make_zero(ps)
#     zero_shadows!(dmodel); zero_shadows!(dTᵢ); zero_shadows!(dSᵢ)
#     zero_shadows!(du_wind_stress); zero_shadows!(dv_wind_stress)
#     zero_shadows!(dT_flux); zero_shadows!(dΔz); zero_shadows!(dmld)
#
#     loss = rdifferentiate_tracer_error(model, Tᵢ, Sᵢ, ps, st, fconst,
#                                        u_wind_stress, v_wind_stress, T_flux, Δz, mld,
#                                        dmodel, dTᵢ, dSᵢ, dps,
#                                        du_wind_stress, dv_wind_stress, dT_flux, dΔz, dmld)
#
#     dps_host = fmap(x -> x isa AbstractArray ? Array(x) : x, dps)
#     opt_state, ps_host = Optimisers.update(opt_state, ps_host, dps_host)
#     @info "epoch $epoch   loss = $loss"
# end

