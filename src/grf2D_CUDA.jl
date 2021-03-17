"""
Module grf2D_CUDA

Provides GPU (CUDA) functions for 2D Gaussian random fields generation with anisotropic exponential and isotropic Gaussian covariance functions.

# Usage
    using ParallelRandomFields.grf2D_CUDA

# Functions
- [`grf2D_expon!()`](@ref)
- [`grf2D_gauss!()`](@ref)
- [`generate_grf2D()`](@ref)

To see a description of a function type `?<functionname>`.
"""
module grf2D_CUDA

    export grf2D_expon!, grf2D_gauss!, generate_grf2D

    @doc raw"""

        grf2D_expon!(Yf, sf, cl, nh, nx, ny, dx, dy; do_reset=true)

    Compute the Gaussian random field `Yf` with anisotropic exponential covariance function.

    # Arguments
    - `Yf::Data.Array`: the Gaussian random field array.
    - `sf::Data.Number`: the standard deviation.
    - `cl::Any`: the correlation length tuple `(clx, cly)` in `x` and `y` dimension.
    - `nh::Int`: the number of harmonics.
    - `nx::Int`: the number of grid points in `x` dimension.
    - `ny::Int`: the number of grid points in `y` dimension.
    - `dx::Data.Number`: the numerical grid size in `x` dimension.
    - `dy::Data.Number`: the numerical grid size in `y` dimension.
    - `do_reset=true`: the optional kwarg to reset the random seed.
    """
    grf2D_expon!()

    @doc raw"""

        grf2D_gauss!(Yf, sf, cl, nh, k_m, nx, ny, dx, dy; do_reset=true)

    Compute the Gaussian random field `Yf` with isotropic Gaussian covariance function.

    # Arguments
    - `Yf::Data.Array`: the Gaussian random field array.
    - `sf::Data.Number`: the standard deviation.
    - `cl::Any`: the correlation length tuple `(clx, cly)` in `x` and `y` dimension.
    - `nh::Int`: the number of harmonics.
    - `k_m::Data.Number`: the maximum value of the wave number.
    - `nx::Int`: the number of grid points in `x` dimension.
    - `ny::Int`: the number of grid points in `y` dimension.
    - `dx::Data.Number`: the numerical grid size in `x` dimension.
    - `dy::Data.Number`: the numerical grid size in `y` dimension.
    - `do_reset=true`: the optional kwarg to reset the random seed.
    """
    grf2D_gauss!()
    
    @doc raw"""

        generate_grf2D(lx=100.0, ly=100.0, sf=1.0, cl=(10.0, 8.0), k_m=100.0, nh=10000, nx=64, ny=64;
                       cov_typ=\"expon\", do_reset=true, do_viz=true, do_save=false)

    Returns the Gaussian random field with exponential `cov_typ=\"expon\"` or Gaussian `cov_typ=\"gauss\"` covariance function for default values. 

    # Arguments
    - `lx=100.0`: the `x` dimension domain extend.
    - `ly=100.0`: the `y` dimension domain extend.
    - `sf=1.0`: the standard deviation.
    - `cl=(10.0, 8.0)`: the correlation length; `(clx, cly)` tuple (`cov_typ=\"expon\"`), `cl[1]/2.0` (`cov_typ=\"gauss\"`).
    - `k_m=100.0`: the maximum value of the wave number.
    - `nh=10000`: the number of harmonics.
    - `nx=64`: the numerical grid size in `x` dimension.
    - `ny=64`: the numerical grid size in `y` dimension.
    - `cov_typ=\"expon\"`: the covariance function `\"expon\"` or `\"gauss\"`.
    - `do_viz=true`: the optional kwarg to enable visualisation.
    - `do_save=false`: the optional kwarg to save the random field to disk.
    - `do_reset=true`: the optional kwarg to reset the random seed.
    """
    generate_grf2D()

    using ParallelStencil
    using ParallelStencil.FiniteDifferences2D
    @init_parallel_stencil(CUDA, Float64, 2)

    macro sin(args...) esc(:(CUDA.sin($(args...)))) end
    macro cos(args...) esc(:(CUDA.cos($(args...)))) end

    include(joinpath("shared", "grf2D.jl"))
    include(joinpath("shared", "generate_grf2D.jl"))
end
