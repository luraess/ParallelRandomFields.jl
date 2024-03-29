using MAT, Plots

@views function generate_grf3D(lx=100.0, ly=100.0, lz=100.0, sf=1.0, cl=(10.0, 8.0, 5.0), k_m=100.0, nh=10000, nx=64, ny=64, nz=64;
                               cov_typ="expon", do_reset=true, do_viz=true, do_save=false)
    # Derived numerics
    dx, dy, dz  = lx/nx, ly/ny, lz/nz  # numerical grid step size
    me = 0
    co1, co2, co3 = 0, 0, 0
    # Array allocation
    Yf      = @zeros(nx, ny, nz)
    # Visu init
    if do_viz
        y_sl    = Int(ceil(ny/2))
        X, Y, Z = -lx/2:dx:lx/2, -ly/2:dy:ly/2, -lz/2:dz:lz/2
    end

    if cov_typ=="expon"
        # Generate the 3D exponential covariance function
        wtime_it = grf3D_expon!(Yf, sf, cl, nh, nx, ny, nz, dx, dy, dz; me, co1, co2, co3, do_reset)
    elseif cov_typ=="gauss"
        # Generate the 3D Gaussian covariance function
        wtime_it = grf3D_gauss!(Yf, sf, cl[1]/2.0, nh, k_m, nx, ny, nz, dx, dy, dz; me, co1, co2, co3, do_reset)
    else
        error("Undefined covariance function.")
    end

    # Performance
    A_eff    = 2/1e9*nx*ny*nz*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B has to be read: 1 whole-array memaccess)
    T_eff    = A_eff/wtime_it                      # Effective memory throughput [GB/s]
    if (me==0) @printf("T_eff = %1.2f GB/s \n", round(T_eff, sigdigits=2)) end

    # Visualisation
    if do_viz
        display(heatmap(X, Z, Array(Yf)[:,y_sl,:]', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Z[1],Z[end]), c=:hot, title="3D RandomField (y-slice)"))
    end
    if do_save  file = matopen("grf3D_$(cov_typ).mat", "w"); write(file, "grf3D", Array(Yf)); close(file)  end

    return Yf
end
