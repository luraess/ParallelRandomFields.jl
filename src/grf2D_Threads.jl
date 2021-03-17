"""
Module grf2D_Threads

Provides CPU (Threads) functions for 2D Gaussian random fields generation with anisotropic exponential and isotropic Gaussian covariance functions.

# Usage
    using ParallelRandomFields.grf2D_Threads

# Functions
- [`grf2D_expon!()`](@ref)
- [`grf2D_gauss!()`](@ref)
- [`generate_grf2D()`](@ref)

To see a description of a function type `?<functionname>`.
"""
module grf2D_Threads

    export grf2D_expon!, grf2D_gauss!, generate_grf2D

    @doc "`grf2D_expon!()`: Compute the Gaussian random field with anisotropic exponential covariance function." :(grf2D_expon!)
    @doc "`grf2D_gauss!()`: Compute the Gaussian random field with isotropic Gaussian covariance function." :(grf2D_gauss!)
    @doc "`generate_grf2D()`: Generate the Gaussian random field with chosen covariance function. Default (but modifiable) input arguments are: `cov_typ=\"expon\", do_viz=true, do_save=false, lx=100.0, ly=100.0, sf=1.0, cl=(10.0, 8.0), k_m=100.0, nh=10000, nx=64, ny=64.`" :(generate_grf2D)

    using ParallelStencil
    using ParallelStencil.FiniteDifferences2D
    @init_parallel_stencil(Threads, Float64, 2)

    macro sin(args...) esc(:(Base.sin($(args...)))) end
    macro cos(args...) esc(:(Base.cos($(args...)))) end

    include(joinpath("shared", "grf2D.jl"))
    include(joinpath("shared", "generate_grf2D.jl"))
end
