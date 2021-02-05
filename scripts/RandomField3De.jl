const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
const GPU_ID  = 0
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
    CUDA.device!(GPU_ID) # select GPU
    macro sin(args...) esc(:(CUDA.sin($(args...)))) end
    macro cos(args...) esc(:(CUDA.cos($(args...)))) end
else
    @init_parallel_stencil(Threads, Float64, 3)
    macro sin(args...) esc(:(Base.sin($(args...)))) end
    macro cos(args...) esc(:(Base.cos($(args...)))) end
end
using MAT, Random, Plots, Printf, Statistics
##################################################
@parallel_indices (ix,iy,iz) function compute_1!(Yf::Data.Array, v1::Data.Number, v2::Data.Number, v3::Data.Number, a::Data.Number, b::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    if (ix<=size(Yf,1) && iy<=size(Yf,2) && iz<=size(Yf,3))  Yf[ix,iy,iz] = Yf[ix,iy,iz] + a*@sin( dx*(ix-0.5)*v1 + dy*(iy-0.5)*v2 + dz*(iz-0.5)*v3 ) + b*@cos( dx*(ix-0.5)*v1 + dy*(iy-0.5)*v2 + dz*(iz-0.5)*v3 )  end
    return
end

@parallel function compute_2!(Yf::Data.Array, c::Data.Number)
    @all(Yf) = @all(Yf)*c
    return
end
##################################################
@views function RandomField3De()
    do_viz  = false
    do_save = false
    Random.seed!(1234)                # Resetting the random seed
    # Physics
    lx, ly, lz = 100.0, 100.0, 100.0  # domain size
    sf         = 1.0                  # standard deviation
    cl         = (10.0, 8.0, 5.0)     # correlation lengths in [x,y,z]
    # Numerics
    nh         = 10000                # inner parameter, number of harmonics
    nx, ny, nz = 64, 64, 64           # numerical grid resolution
    # Derived numerics
    dx, dy, dz = lx/nx, ly/ny, lz/nz  # numerical grid step size
    c          = sf/sqrt(nh)
    # Array allocation
    Yf         = @zeros(nx, ny, nz)
    # Scalar allocations
    ϕ=0.0; k=0.0; d=0.0; θ=0.0; v1=0.0; v2=0.0; v3=0.0; a=0.0; b=0.0
    # Preparation of visualisation
    if do_viz
        ENV["GKSwstype"]="nul"; if isdir("viz3D_exp")==false mkdir("viz3D_exp") end; loadpath = "./viz3D_exp/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
        y_sl    = Int(ceil(ny/2))
        X, Y, Z = -lx/2:dx:lx/2, -ly/2:dy:ly/2, -lz/2:dz:lz/2
    end
    println("Starting 3D RandomField generation (anisotropic exponential covariance function)...")
    # Loop over nh harmonics
    for ih = 1:nh
        if (ih==501)  global wtime0 = Base.time()  end
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
        @parallel compute_1!(Yf, v1, v2, v3, a, b, dx, dy, dz)
    end
    @parallel compute_2!(Yf, c)
    # Performance
    wtime    = Base.time() - wtime0
    A_eff    = 2/1e9*nx*ny*nz*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B has to be read: 1 whole-array memaccess)
    wtime_it = wtime/(nh-500)                      # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                      # Effective memory throughput [GB/s]
    @printf("Total harmonic iters=%d, time=%1.3e sec (@ T_eff = %1.2f GB/s) \n", nh, wtime, round(T_eff, sigdigits=2))
    # Visualisation
    if do_viz
        # display(heatmap(X, Z, Array(Yf)[:,y_sl,:]', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Z[1],Z[end]), c=:hot, title="3D RandomField (y-slice)"))
        heatmap(X, Z, Array(Yf)[:,y_sl,:]', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Z[1],Z[end]), c=:hot, title="3D RandomField (y-slice)"); frame(anim)
        gif(anim, "RandomField3D_exp.gif", fps = 15)
    end
    if do_save  file = matopen("Rnd3De.mat", "w"); write(file, "Rnd3D", Array(Yf)); close(file)  end
    return
end

RandomField3De()
