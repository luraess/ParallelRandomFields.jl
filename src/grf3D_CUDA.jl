"""
Module grf3D_CUDA

Provides GPU (CUDA) functions for 3D Gaussian random fields generation with anisotropic exponential and isotropic Gaussian covariance functions.

# Usage
    using ParallelRandomFields.grf3D_CUDA

# Functions
- [`grf3D_expon!()`](@ref)
- [`grf3D_gauss!()`](@ref)
- [`generate_grf3D()`](@ref)

To see a description of a function type `?<functionname>`.
"""
module grf3D_CUDA

    export grf3D_expon!, grf3D_gauss!, generate_grf3D

    @doc raw"""

        grf3D_expon!(Yf, sf, cl, nh, nx, ny, nz, dx, dy, dz; do_reset=true)

    Compute the Gaussian random field `Yf` with anisotropic exponential covariance function.

    # Arguments
    - `Yf::Data.Array`: the Gaussian random field array.
    - `sf::Data.Number`: the standard deviation.
    - `cl::Any`: the correlation length tuple `(clx, cly, clz)` in `x`, `y` and `z` dimension.
    - `nh::Int`: the number of harmonics.
    - `nx::Int`: the number of grid points in `x` dimension.
    - `ny::Int`: the number of grid points in `y` dimension.
    - `nz::Int`: the number of grid points in `z` dimension.
    - `dx::Data.Number`: the numerical grid size in `x` dimension.
    - `dy::Data.Number`: the numerical grid size in `y` dimension.
    - `dz::Data.Number`: the numerical grid size in `z` dimension.
    - `do_reset=true`: the optional kwarg to reset the random seed.
    """
    grf3D_expon!()

    @doc raw"""

        grf3D_gauss!(Yf, sf, cl, nh, k_m, nx, ny, nz, dx, dy, dz; do_reset=true)

    Compute the Gaussian random field `Yf` with isotropic Gaussian covariance function.

    # Arguments
    - `Yf::Data.Array`: the Gaussian random field array.
    - `sf::Data.Number`: the standard deviation.
    - `cl::Any`: the correlation length tuple `(clx, cly, clz)` in `x`, `y` and `z` dimension.
    - `nh::Int`: the number of harmonics.
    - `k_m::Data.Number`: the maximum value of the wave number.
    - `nx::Int`: the number of grid points in `x` dimension.
    - `ny::Int`: the number of grid points in `y` dimension.
    - `nz::Int`: the number of grid points in `z` dimension.
    - `dx::Data.Number`: the numerical grid size in `x` dimension.
    - `dy::Data.Number`: the numerical grid size in `y` dimension.
    - `dz::Data.Number`: the numerical grid size in `z` dimension.
    - `do_reset=true`: the optional kwarg to reset the random seed.
    """
    grf3D_gauss!()
    
    @doc raw"""

        generate_grf3D(lx=100.0, ly=100.0, lz=100.0, sf=1.0, cl=(10.0, 8.0, 5.0), k_m=100.0, nh=10000, nx=64, ny=64, nz=64;
                       cov_typ=\"expon\", do_reset=true, do_viz=true, do_save=false)

    Returns the Gaussian random field with exponential `cov_typ=\"expon\"` or Gaussian `cov_typ=\"gauss\"` covariance function for default values. 

    # Arguments
    - `lx=100.0`: the `x` dimension domain extend.
    - `ly=100.0`: the `y` dimension domain extend.
    - `lz=100.0`: the `z` dimension domain extend.
    - `sf=1.0`: the standard deviation.
    - `cl=(10.0, 8.0, 5.0)`: the correlation length; `(clx, cly, clz)` tuple (`cov_typ=\"expon\"`), `cl[1]/2.0` (`cov_typ=\"gauss\"`).
    - `k_m=100.0`: the maximum value of the wave number.
    - `nh=10000`: the number of harmonics.
    - `nx=64`: the numerical grid size in `x` dimension.
    - `ny=64`: the numerical grid size in `y` dimension.
    - `nz=64`: the numerical grid size in `y` dimension.
    - `cov_typ=\"expon\"`: the covariance function `\"expon\"` or `\"gauss\"`.
    - `do_viz=true`: the optional kwarg to enable visualisation.
    - `do_save=false`: the optional kwarg to save the random field to disk.
    - `do_reset=true`: the optional kwarg to reset the random seed.
    """
    generate_grf3D()
    
    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @init_parallel_stencil(CUDA, Float64, 3)

    include(joinpath("shared", "grf3D.jl"))
    include(joinpath("shared", "generate_grf3D.jl"))
end
