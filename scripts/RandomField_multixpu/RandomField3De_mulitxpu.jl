const USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
    macro sin(args...) esc(:(CUDA.sin($(args...)))) end
    macro cos(args...) esc(:(CUDA.cos($(args...)))) end
else
    @init_parallel_stencil(Threads, Float64, 3)
    macro sin(args...) esc(:(Base.sin($(args...)))) end
    macro cos(args...) esc(:(Base.cos($(args...)))) end
end
using MAT, ImplicitGlobalGrid, Random, Plots, Printf, Statistics
import MPI
##################################################
@parallel_indices (ix,iy,iz) function compute_1!(Yf::Data.Array, v1::Data.Number, v2::Data.Number, v3::Data.Number, a::Data.Number, b::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number, nx::Int, ny::Int, nz::Int, co1::Int, co2::Int, co3::Int)
    if (ix<=size(Yf,1) && iy<=size(Yf,2) && iz<=size(Yf,3))  tmp  = dx*(co1*(nx-2) + ix - 0.5)*v1 + dy*(co2*(ny-2) + iy - 0.5)*v2 + dz*(co3*(nz-2) + iz - 0.5)*v3  end
    if (ix<=size(Yf,1) && iy<=size(Yf,2) && iz<=size(Yf,3))  Yf[ix,iy,iz] = Yf[ix,iy,iz] + a*@sin( tmp ) + b*@cos( tmp )  end
    return
end

@parallel function compute_2!(Yf::Data.Array, c::Data.Number)
    @all(Yf) = @all(Yf)*c
    return
end
##################################################
@views function RandomField3D()
    do_viz  = false
    do_save = false
    Random.seed!(1234)                # Resetting the random seed
    # Physics
    lx, ly, lz = 100.0, 100.0, 100.0  # domain size
    sf         = 1.0                  # standard deviation
    cl         = (10.0, 8.0, 5.0)     # correlation lengths in [x,y,z]
    # Numerics
    nh         = 10000                # inner parameter, number of harmonics
    nx, ny, nz = 256, 256, 256        # numerical grid resolution
    # Derived numerics
    me, dims, nprocs, coords, comm = init_global_grid(nx, ny, nz) # MPI initialisation
    select_device()                                               # select one GPU per MPI local rank (if >1 GPU per node)
    dx, dy, dz = lx/nx_g(), ly/ny_g(), lz/nz_g()                  # numerical grid step size
    c          = sf/sqrt(nh)
    # Array allocation
    Yf         = @zeros(nx, ny, nz)
    # Scalar allocations
    co1, co2, co3 = coords[1], coords[2], coords[3]
    ϕ=0.0; k=0.0; d=0.0; θ=0.0; v1=0.0; v2=0.0; v3=0.0; a=0.0; b=0.0
    # Preparation of visualisation
    if do_viz
        ENV["GKSwstype"]="nul"; 
        if (me==0)
            if isdir("viz3D_exp")==false mkdir("viz3D_exp") end; loadpath = "./viz3D_exp/"; anim = Animation(loadpath,String[])
            println("Animation directory: $(anim.dir)")
        end
        nx_v, ny_v, nz_v = (nx-2)*dims[1], (ny-2)*dims[2], (nz-2)*dims[3]
        if (nx_v*ny_v*nz_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
        Yf_v   = zeros(nx_v, ny_v, nz_v) # global array for visu
        Yf_inn = zeros(nx-2, ny-2, nz-2) # no halo local array for visu
        y_sl   = Int(ceil(ny_g()/2))
        Xi_g, Yi_g, Zi_g = -lx/2+dx:dx:lx/2-dx, -ly/2+dy:dy:ly/2-dy, -lz/2+dz:dz:lz/2-dz
    end
    if (me==0)  println("Starting 3D RandomField generation (anisotropic exponential covariance function)...")  end
    # Loop over nh harmonics
    for ih = 1:nh
        if (ih==501)  tic()  end
        ϕ    = 2.0*pi*rand()
        # Gaussian spectrum
        flag = true
        while flag
            k  = tan(pi*0.5*rand())
            d  = (k*k)/(1.0+(k*k))
            if (rand()<d)  flag = false  end
        end
        θ    = acos(1.0-2.0*rand())
        v1   = k*sin(ϕ)*sin(θ)/cl[1]
        v2   = k*cos(ϕ)*sin(θ)/cl[2]
        v3   = k*cos(θ)       /cl[3]
        a, b = randn(), randn()
        @parallel compute_1!(Yf, v1, v2, v3, a, b, dx, dy, dz, nx, ny, nz, co1, co2, co3)
    end
    @parallel compute_2!(Yf, c)
    # Performance
    wtime    = toc()
    A_eff    = 2/1e9*nx*ny*nz*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B has to be read: 1 whole-array memaccess)
    wtime_it = wtime/(nh-500)                      # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                      # Effective memory throughput [GB/s]
    if (me==0)  @printf("Total harmonic iters=%d, time=%1.3e sec (@ T_eff = %1.2f GB/s) \n", nh, wtime, round(T_eff, sigdigits=2))  end
    # Visualisation
    if do_viz
        Yf_inn .= Yf[2:end-1,2:end-1,2:end-1]; gather!(Yf_inn, Yf_v)
        if (me==0)
            # display(heatmap(X, Z, Array(Yf)[:,y_sl,:]', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Z[1],Z[end]), c=:inferno, title="3D RandomField (y-slice)"))
            heatmap(Xi_g, Zi_g, Yf_v[:,y_sl,:]', aspect_ratio=1, xlims=(Xi_g[1],Xi_g[end]), ylims=(Zi_g[1],Zi_g[end]), c=:inferno, title="3D RandomField (y-slice)"); frame(anim)
            gif(anim, "RandomFIeld3D_exp.gif", fps = 15)
        end
    end
    if do_save  file = matopen("Rnd3De.mat", "w"); write(file, "Rnd3D", Array(Yf_v)); close(file)  end
    finalize_global_grid()
    return
end

RandomField3D()
