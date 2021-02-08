const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
const GPU_ID  = 0
@static if USE_GPU
    macro sin(args...) esc(:(CUDA.sin($(args...)))) end
    macro cos(args...) esc(:(CUDA.cos($(args...)))) end
else
    macro sin(args...) esc(:(Base.sin($(args...)))) end
    macro cos(args...) esc(:(Base.cos($(args...)))) end
end
using Random, Printf, Statistics
##################################################
@parallel_indices (ix,iy) function compute_1!(Yf::Data.Array, v1::Data.Number, v2::Data.Number, a::Data.Number, b::Data.Number, dx::Data.Number, dy::Data.Number)
    if (ix<=size(Yf,1) && iy<=size(Yf,2))  Yf[ix,iy] = Yf[ix,iy] + a*@sin( dx*(ix-0.5)*v1 + dy*(iy-0.5)*v2 ) + b*@cos( dx*(ix-0.5)*v1 + dy*(iy-0.5)*v2 )  end
    return
end

@parallel function compute_2!(Yf::Data.Array, c::Data.Number)
    @all(Yf) = @all(Yf)*c
    return
end
##################################################
@views function RndField2D_expon!(Yf,::Data.Array, lx::Data.Number, ly::Data.Number, sf::Data.Number, cl, nh::Int, nx::Int, ny::Int, dx::Data.Number, dy::Data.Number; do_reset=true)
    # Resetting the random seed if needed
    if do_reset  Random.seed!(1234)  end
    # Derived numerics
    c = sf/sqrt(nh)
    # Scalar allocations
    ϕ=0.0; k=0.0; d=0.0; θ=0.0; v1=0.0; v2=0.0; a=0.0; b=0.0
    println("Starting 2D RandomField generation (anisotropic exponential covariance function)...")
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
    return wtime, T_eff
end
