using ParallelRandomFields
using ParallelRandomFields.grf2D_Threads

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2)

using MAT, Plots

@views function generate_grf2D()
    cov_typ = "expon"
    do_viz  = true
    do_save = false
    # Physics
    lx, ly  = 100.0, 100.0  # domain size
    sf      = 1.0           # standard deviation
    # -- exponential setup
    cl_e    = (10.0, 8.0)   # correlation lengths in [x,y]
    # -- gaussian setup
    cl_g    = 5.0           # correlation length isotropic
    k_m     = 100.0         # maximum value of the wave number
    # Numerics
    nh      = 10000         # inner parameter, number of harmonics
    nx, ny  = 64, 64      # numerical grid resolution
    # Derived numerics
    dx, dy  = lx/nx, ly/ny  # numerical grid step size
    # Array allocation
    Yf      = @zeros(nx, ny)
    # Visu init
    if do_viz
        X, Y = -lx/2:dx:lx/2, -ly/2:dy:ly/2
    end

    if cov_typ=="expon"
        # Generate the 2D exponential covariance function
        grf2D_expon!(Yf, sf, cl_e, nh, nx, ny, dx, dy; do_reset=true)
    elseif cov_typ=="gauss"
        # Generate the 2D Gaussian covariance function
        grf2D_gauss!(Yf, sf, cl_g, nh, k_m, nx, ny, dx, dy; do_reset=true)
    else
        error("trying to run with undefined covariance function")
    end
    # Visualisation
    if do_viz
        display(heatmap(X, Y, Array(Yf)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:hot, title="2D RandomField"))
    end
    if do_save  file = matopen("grf2D_$(cov_typ).mat", "w"); write(file, "grf2D", Array(Yf)); close(file)  end

    return
end

generate_grf2D()
