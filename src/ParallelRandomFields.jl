"""
Module ParallelRandomFields

Enables to sample spatial realisations of a 2-D and 3-D Gaussian random fields with given power spectrum: anisotropic exponential and isotropic Gaussian covariance functions. 

# General overview and examples
https://github.com/luraess/ParallelRandomFields.jl

# Functions
- to come

To see a description of a function or a macro type `?<functionname>`.
"""
module ParallelRandomFields

# using Random, Printf, Statistics
# using ParallelStencil

module grf2D_Threads

    export grf2D_expon!, grf2D_gauss!, generate_grf2D

    using ParallelStencil
    using ParallelStencil.FiniteDifferences2D

    @init_parallel_stencil(Threads, Float64, 2)
    macro sin(args...) esc(:(Base.sin($(args...)))) end
    macro cos(args...) esc(:(Base.cos($(args...)))) end

    include(joinpath("shared", "grf2D.jl"))

    include(joinpath("shared", "generate_grf2D.jl"))
end

# ParallelStencil.@reset_parallel_stencil()
# module grf2D_CUDA
#     using ParallelStencil
#     using ParallelStencil.FiniteDifferences2D
#     @init_parallel_stencil(CUDA, Float64, 2)

#     macro sin(args...) esc(:(CUDA.sin($(args...)))) end
#     macro cos(args...) esc(:(CUDA.cos($(args...)))) end

#     include(joinpath(@__DIR__, "shared", "grf2D.jl"))

# end

# ParallelStencil.@reset_parallel_stencil()
# module grf3D_Threads
#     using ParallelStencil
#     using ParallelStencil.FiniteDifferences3D

#     @init_parallel_stencil(Threads, Float64, 3)
#     macro sin(args...) esc(:(Base.sin($(args...)))) end
#     macro cos(args...) esc(:(Base.cos($(args...)))) end

#     # include(joinpath("shared", "grf2D.jl"))

# end

end # Module ParallelRandomFields
