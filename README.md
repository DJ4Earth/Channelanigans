# Channelanigans

# READ THIS FIRST:

This is a special branch of the `Channelanigans` repo that is linked to the DJ4Earth differentiable ESM repository: [https://github.com/DJ4Earth/differentiable-esm-components-2025](https://github.com/DJ4Earth/differentiable-esm-components-2025). This is released in combination with the manuscript submission "DJ4Earth: Differentiable, and Performance-portable Earth System Modeling via Program Transformations"

To replicate the numerical examples in the paper featuring Oceananigans:

1. Instantiate the environment with the given Project.toml and Manifest.toml files.
2. Run `julia -O0 --project scripts/abernathy_channel.jl` to run the model with AD, producing both results and profiling data.
3. Run `julia -O0 --project scripts/makie_abernathy.jl` to produce plots from the produced data.

To change the directory where model data and graphs are produced, edit lines 30 in `abernathy_channel.jl` and 17 in `makie_abernathy.jl`.
You can also change the number of timesteps in the model spinup and AD run in lines 254 and 284, respectively, of `abernathy_channel.jl` (Reactant requires the number of iterations to be hardcoded).
