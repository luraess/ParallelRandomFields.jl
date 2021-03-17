using Random, Printf, Statistics
# XPU kernels
@parallel_indices (ix,iy,iz) function compute_1!(Yf::Data.Array, v1::Data.Number, v2::Data.Number, v3::Data.Number, a::Data.Number, b::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    if (ix<=size(Yf,1) && iy<=size(Yf,2) && iz<=size(Yf,3))  Yf[ix,iy,iz] = Yf[ix,iy,iz] + a*@sin( dx*(ix-0.5)*v1 + dy*(iy-0.5)*v2 + dz*(iz-0.5)*v3 ) + b*@cos( dx*(ix-0.5)*v1 + dy*(iy-0.5)*v2 + dz*(iz-0.5)*v3 )  end
    return
end

@parallel function compute_2!(Yf::Data.Array, c::Data.Number)
    @all(Yf) = @all(Yf)*c
    return
end
# 3D Gaussian random field with exponnential covariance
@views function grf3D_expon!(Yf::Data.Array, sf::Data.Number, cl, nh::Int, nx::Int, ny::Int, nz::Int, dx::Data.Number, dy::Data.Number, dz::Data.Number; do_reset=true)
    # Resetting the random seed if needed
    if do_reset  Random.seed!(1234)  end
    # Derived numerics
    c = sf/sqrt(nh)
    # Scalar allocations
    ϕ=0.0; k=0.0; d=0.0; θ=0.0; v1=0.0; v2=0.0; v3=0.0; a=0.0; b=0.0
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
    return wtime, T_eff
end

# 3D Gaussian random field with Gaussian covariance
@views function grf3D_gauss!(Yf::Data.Array, sf::Data.Number, cl, nh::Int, k_m::Data.Number, nx::Int, ny::Int, nz::Int, dx::Data.Number, dy::Data.Number, dz::Data.Number; do_reset=true)
    # Resetting the random seed if needed
    if do_reset  Random.seed!(1234)  end
    # Derived numerics
    c      = sf/sqrt(nh)
    lf     = 2.0*cl/sqrt(pi)
    # Scalar allocations
    ϕ=0.0; k=0.0; d=0.0; θ=0.0; v1=0.0; v2=0.0; v3=0.0; a=0.0; b=0.0
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
        v3   = k*cos(θ)
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
    return wtime, T_eff
end
