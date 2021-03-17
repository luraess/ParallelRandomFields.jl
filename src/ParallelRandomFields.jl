"""
Module ParallelRandomFields

Enables to sample spatial realisations of 2D and 3D Gaussian random fields with given power spectrum using follwoing covariance functions:
- anisotropic exponential
- isotropic Gaussian.

ParallelRandomFields can be deployed on both (multi-) GPUs and CPUs.

# General overview and examples
https://github.com/luraess/ParallelRandomFields.jl

Functions are defined in the submodules.

# Submodules
- [`ParallelRandomFields.grf2D_Threads`](@ref)
- [`ParallelRandomFields.grf3D_Threads`](@ref)
- [`ParallelRandomFields.grf2D_CUDA`](@ref)
- [`ParallelRandomFields.grf3D_CUDA`](@ref)

To see a description of a function type `?<functionname>`.
"""
module ParallelRandomFields

using ParallelStencil

# include submodules and reset ParallelStencil prior
include("grf2D_Threads.jl")

ParallelStencil.@reset_parallel_stencil()
include("grf3D_Threads.jl")

ParallelStencil.@reset_parallel_stencil()
include("grf2D_CUDA.jl")

ParallelStencil.@reset_parallel_stencil()
include("grf3D_CUDA.jl")

end # Module ParallelRandomFields
