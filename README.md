# Channelanigans

# READ THIS FIRST:

This is a special branch of the `Channelanigans` repo that is linked to the DJ4Earth differentiable ESM repository: [https://github.com/DJ4Earth/differentiable-esm-components-2025](https://github.com/DJ4Earth/differentiable-esm-components-2025). This is released in combination with the manuscript submission "DJ4Earth: Differentiable, and Performance-portable Earth System Modeling via Program Transformations"

To replicate the numerical examples in the paper featuring Oceananigans:

1. Instantiate the environment with the given Project.toml and Manifest.toml files.
2. Run `julia -O0 --project scripts/abernathey_channel.jl` to run the model with AD, producing results and storing them into data files.
3. Run `julia -O0 --project scripts/makie_abernathey.jl` to produce plots from the produced data.

You can change the number of timesteps in the model spinup and AD run in lines 36 and 37, respectively, of `abernathy_channel.jl` (Reactant requires the number of iterations to be hardcoded). This will also change the directory name where plots and data are stored.

Lines 451-460 can be uncommented to profile the run time and memory footprint of the model.
Line 463-513 can be uncommented to perform AD vs FD accuracy checks.
