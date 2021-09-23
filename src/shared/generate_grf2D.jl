using MAT, Plots

@views function generate_grf2D(lx=100.0, ly=100.0, sf=1.0, cl=(10.0, 8.0), k_m=100.0, nh=10000, nx=64, ny=64;
                               cov_typ="expon", do_reset=true, do_viz=true, do_save=false)
    # Derived numerics
    dx, dy  = lx/nx, ly/ny  # numerical grid step size
    me = 0
    co1, co2 = 0, 0
    # Array allocation
    Yf      = @zeros(nx, ny)
    # Visu init
    if do_viz
        X, Y = -lx/2:dx:lx/2, -ly/2:dy:ly/2
    end

    if cov_typ=="expon"
        # Generate the 2D exponential covariance function
        wtime_it = grf2D_expon!(Yf, sf, cl, nh, nx, ny, dx, dy; me, co1, co2, do_reset)
    elseif cov_typ=="gauss"
        # Generate the 2D Gaussian covariance function
        wtime_it = grf2D_gauss!(Yf, sf, cl[1]/2.0, nh, k_m, nx, ny, dx, dy; me, co1, co2, do_reset)
    else
        error("Undefined covariance function.")
    end

    # Performance
    A_eff    = 2/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B has to be read: 1 whole-array memaccess)
    T_eff    = A_eff/wtime_it                   # Effective memory throughput [GB/s]
    if (me==0) @printf("T_eff = %1.2f GB/s \n", round(T_eff, sigdigits=3)) end

    # Visualisation
    if do_viz
        display(heatmap(X, Y, Array(Yf)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:hot, title="2D RandomField"))
    end
    if do_save  file = matopen("grf2D_$(cov_typ).mat", "w"); write(file, "grf2D", Array(Yf)); close(file)  end

    return Yf
end
