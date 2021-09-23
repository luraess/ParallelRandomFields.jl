using Random, Printf, Statistics
# XPU kernels
macro coords() esc(:( dx*(co1*(nx-2) + ix-0.5)*v1 + dy*(co2*(ny-2) + iy-0.5)*v2 )) end
@parallel_indices (ix,iy) function compute_1!(Yf::Data.Array, v1::Data.Number, v2::Data.Number, a::Data.Number, b::Data.Number, dx::Data.Number, dy::Data.Number, nx::Int, ny::Int, co1::Int, co2::Int)
    if (ix<=size(Yf,1) && iy<=size(Yf,2))  Yf[ix,iy] = Yf[ix,iy] + a*sin( @coords() ) + b*cos( @coords() )  end
    return
end

@parallel function compute_2!(Yf::Data.Array, c::Data.Number)
    @all(Yf) = @all(Yf)*c
    return
end

# 2D Gaussian random field with exponnential covariance
@views function grf2D_expon!(Yf::Data.Array, sf::Data.Number, cl, nh::Int, nx::Int, ny::Int, dx::Data.Number, dy::Data.Number; me=0::Int, co1=0::Int, co2=0::Int, do_reset=true)
    # Resetting the random seed if needed
    if do_reset  Random.seed!(1234)  end
    # Derived numerics
    c = sf/sqrt(nh)
    # Scalar allocations
    ϕ=0.0; k=0.0; d=0.0; θ=0.0; v1=0.0; v2=0.0; a=0.0; b=0.0
    if (me==0) println("Starting 2D RandomField generation (anisotropic exponential covariance function)...") end
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
        @parallel compute_1!(Yf, v1, v2, a, b, dx, dy, nx, ny, co1, co2)
    end
    @parallel compute_2!(Yf, c)
    # Performance
    wtime    = Base.time() - wtime0
    wtime_it = wtime/(nh-500)                   # Execution time per iteration [s]
    if (me==0) @printf("Total harmonic iters=%d, time=%1.3e sec \n", nh, wtime) end
    return wtime_it
end

# 2D Gaussian random field with Gaussian covariance
@views function grf2D_gauss!(Yf::Data.Array, sf::Data.Number, cl, nh::Int, k_m::Data.Number, nx::Int, ny::Int, dx::Data.Number, dy::Data.Number; me=0::Int, co1=0::Int, co2=0::Int, do_reset=true)
    # Resetting the random seed if needed
    if do_reset  Random.seed!(1234)  end
    # Derived numerics
    c      = sf/sqrt(nh)
    lf     = 2.0*cl/sqrt(pi)
    # Scalar allocations
    ϕ=0.0; k=0.0; d=0.0; θ=0.0; v1=0.0; v2=0.0; a=0.0; b=0.0
    if (me==0) println("Starting 2D RandomField generation (isotropic Gaussian covariance function)...") end
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
        @parallel compute_1!(Yf, v1, v2, a, b, dx, dy, nx, ny, co1, co2)
    end
    @parallel compute_2!(Yf, c)
    # Performance
    wtime    = Base.time() - wtime0
    wtime_it = wtime/(nh-500)                   # Execution time per iteration [s]
    if (me==0) @printf("Total harmonic iters=%d, time=%1.3e sec \n", nh, wtime) end
    return wtime_it
end
