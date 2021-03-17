"""
Module grf3D_Threads

Provides CPU (Threads) functions for 3D Gaussian random fields generation with anisotropic exponential and isotropic Gaussian covariance functions.

# Usage
    using ParallelRandomFields.grf3D_Threads

# Functions
- [`grf3D_expon!`](@ref)
- [`grf3D_gauss!`](@ref)
- [`generate_grf3D`](@ref)

To see a description of a function type `?<functionname>`.
"""
module grf3D_Threads

    export grf3D_expon!, grf3D_gauss!, generate_grf3D

    @doc "`grf3D_expon!`: Compute the Gaussian random field with anisotropic exponential covariance function." :(grf3D_expon!)
    @doc "`grf3D_gauss!`: Compute the Gaussian random field with isotropic Gaussian covariance function." :(grf3D_gauss!)
    @doc "`generate_grf3D`: Generate the Gaussian random field with chosen covariance function. Default (but modifiable) input arguments are: `cov_typ=\"expon\", do_viz=true, do_save=false, lx=100.0, ly=100.0, lz=100.0, sf=1.0, cl=(10.0, 8.0, 5.0), k_m=100.0, nh=10000, nx=64, ny=64, nz=64.`" :(generate_grf3D)

    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @init_parallel_stencil(Threads, Float64, 3)

    macro sin(args...) esc(:(Base.sin($(args...)))) end
    macro cos(args...) esc(:(Base.cos($(args...)))) end

    include(joinpath("shared", "grf3D.jl"))
    include(joinpath("shared", "generate_grf3D.jl"))
end
