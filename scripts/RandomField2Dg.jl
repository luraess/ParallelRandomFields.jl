const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
const GPU_ID  = 0
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(GPU_ID) # select GPU
    macro sin(args...) esc(:(CUDA.sin($(args...)))) end
    macro cos(args...) esc(:(CUDA.cos($(args...)))) end
else
    @init_parallel_stencil(Threads, Float64, 2)
    macro sin(args...) esc(:(Base.sin($(args...)))) end
    macro cos(args...) esc(:(Base.cos($(args...)))) end
end
using MAT, Random, Plots, Printf, Statistics
##################################################
@parallel_indices (ix,iy) function compute_1!(Yf::Data.Array, v1::Data.Number, v2::Data.Number, a::Data.Number, b::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    if (ix<=size(Yf,1) && iy<=size(Yf,2))  Yf[ix,iy] = Yf[ix,iy] + a*@sin( dx*(ix-0.5)*v1 + dy*(iy-0.5)*v2 ) + b*@cos( dx*(ix-0.5)*v1 + dy*(iy-0.5)*v2 )  end
    return
end

@parallel function compute_2!(Yf::Data.Array, c::Data.Number)
    @all(Yf) = @all(Yf)*c
    return
end
##################################################
@views function RandomField2Dg()
    do_viz  = false
    do_save = false
    Random.seed!(1234)     # Resetting the random seed
    # Physics
    lx, ly = 100.0, 100.0  # domain size
    sf     = 1.0           # standard deviation
    cl     = 5.0           # correlation length isotropic
    k_m    = 100.0         # maximum value of the wave number
    # Numerics
    nh     = 10000         # inner parameter, number of harmonics
    nx, ny = 128, 128      # numerical grid resolution
    # Derived numerics
    dx, dy = lx/nx, ly/ny  # numerical grid step size
    c      = sf/sqrt(nh)
    lf     = 2.0*cl/sqrt(pi)
    # Array allocation
    Yf     = @zeros(nx, ny)
    # Scalar allocations
    ϕ=0.0; k=0.0; d=0.0; θ=0.0; v1=0.0; v2=0.0; a=0.0; b=0.0
    # Preparation of visualisation
    if do_viz
        ENV["GKSwstype"]="nul"; if isdir("viz2D_gauss")==false mkdir("viz2D_gauss") end; loadpath = "./viz2D_gauss/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
        X, Y = -lx/2:dx:lx/2, -ly/2:dy:ly/2
    end
    println("Starting 2D RandomField generation (isotropic Gaussian covariance function)...")
    # Loop over nh harmonics
    for ih = 1:nh
        if (ih==501)  global wtime0 = Base.time()  end
        ϕ    = 2.0*pi*rand()
        # Gaussian spectrum
        flag = true
        while flag
            k  = k_m*rand()
            d  = k*k*exp(-0.5*k*k)
            if (rand()*2.0*exp(-1.0)<d)  flag = false  end
        end
        k    = k/lf*sqrt(2.0)
        θ    = acos(1.0-2.0*rand())
        v1   = k*sin(ϕ)*sin(θ)
        v2   = k*cos(ϕ)*sin(θ)
        a, b = randn(), randn()
        @parallel compute_1!(Yf, v1, v2, a, b, dx, dy)
    end
    @parallel compute_2!(Yf, c)
    # Performance
    wtime    = Base.time() - wtime0
    A_eff    = 2/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B has to be read: 1 whole-array memaccess)
    wtime_it = wtime/(nh-500)                   # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                   # Effective memory throughput [GB/s]
    @printf("Total harmonic iters=%d, time=%1.3e sec (@ T_eff = %1.2f GB/s) \n", nh, wtime, round(T_eff, sigdigits=2))
    # Visualisation
    if do_viz
        # display(heatmap(X, Y, Array(Yf)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:hot, title="2D RandomField"))
        heatmap(X, Y, Array(Yf)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:hot, title="2D RandomField"); frame(anim)
        gif(anim, "RandomField2D_gauss.gif", fps = 15)
    end
    if do_save  file = matopen("Rnd2Dg.mat", "w"); write(file, "Rnd2D", Array(Yf)); close(file)  end
    return
end

RandomField2Dg()
