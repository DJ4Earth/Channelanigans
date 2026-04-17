#using Pkg
# pkg"add Oceananigans CairoMakie"
ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Statistics

using CUDA

using Reactant
#Reactant.set_default_backend("cpu")

using Enzyme

@info "To specify architecture uncomment line 'Reactant.set_default_backend(\"cpu\")' "
#Reactant.set_default_backend("cpu")

const Ntimesteps = 25        # Number of timesteps in zonal transport computed / AD'ed part
const Nspinup    = 100        # Number of timesteps that the model is spun up

#####
##### Spin up (because step cound is hardcoded we need separate functions for each loop...)
#####

function my_step!(FT, Δt)
    Δt = convert(FT, Δt)
    return nothing
end

function spinup_loop!(FT, Δt)
    @trace mincut = true track_numbers = false for i = 1:Nspinup
        my_step!(FT, Δt)
    end
    return nothing
end

# Timestep size:
Δt₀ = 200.0

thing = ConcreteRNumber{Float64}(Δt₀)

rspinup_reentrant_channel_model! = @compile raise_first=true raise=true sync=true  spinup_loop!(Float64, thing)
