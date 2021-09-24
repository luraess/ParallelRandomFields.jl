const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false
const do_viz  = haskey(ENV, "DO_VIZ")  ? parse(Bool, ENV["DO_VIZ"])  : false
const do_save = haskey(ENV, "DO_SAVE") ? parse(Bool, ENV["DO_SAVE"]) : false
using ParallelRandomFields
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    using ParallelRandomFields.grf2D_CUDA
else
    @init_parallel_stencil(Threads, Float64, 2)
    using ParallelRandomFields.grf2D_Threads
end

using ImplicitGlobalGrid, MAT, Plots
import MPI

@views inn(A) = A[2:end-1,2:end-1]

@views function generate_grf2D_multixpu()
    cov_typ = "expon"
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
    nx, ny  = 32, 32        # numerical grid resolution
    me, dims, nprocs, coords, comm = init_global_grid(nx, ny, 1) # MPI initialisation
    @static if USE_GPU select_device() end    # select one GPU per MPI local rank (if >1 GPU per node)
    # Derived numerics
    dx, dy  = lx/nx_g(), ly/ny_g()    # cell sizes
    # Array allocation
    Yf      = @zeros(nx, ny)
    # Visu init
    if do_viz || do_save
        nx_v, ny_v = (nx-2)*dims[1], (ny-2)*dims[2]
        if (nx_v*ny_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
        Yf_v     = zeros(nx_v, ny_v) # global array for visu
        Yf_inn   = zeros(nx-2, ny-2) # no halo local array for visu
        y_sl     = Int(ceil((ny_g()-2)/2))
        Xi_g, Yi_g = LinRange(dx+dx/2, lx-dx-dx/2, nx_v), LinRange(dy+dy/2, ly-dy-dy/2, ny_v) # inner points only

    end
    if cov_typ=="expon"
        # Generate the 2D exponential covariance function
        grf2D_expon!(Yf, sf, cl_e, nh, nx, ny, dx, dy; me=me, co1=coords[1], co2=coords[2], do_reset=true)
    elseif cov_typ=="gauss"
        # Generate the 2D Gaussian covariance function
        grf2D_gauss!(Yf, sf, cl_g, nh, k_m, nx, ny, dx, dy; me=me, co1=coords[1], co2=coords[2], do_reset=true)
    else
        error("Undefined covariance function")
    end
    # Visualisation
    if do_viz || do_save
        Yf_inn .= inn(Yf); gather!(Yf_inn, Yf_v)
        if do_viz && me==0
            heatmap(Xi_g, Yi_g, Yf_v', aspect_ratio=1, xlims=(Xi_g[1],Xi_g[end]), ylims=(Yi_g[1],Yi_g[end]), c=:hot, title="2D RandomField")
            savefig("grf2D_multixpu_$(nx_v)x$(ny_v).png")
        end
        if do_save && me==0
            file = matopen("grf2D_multixpu_$(cov_typ).mat", "w"); write(file, "grf2D", Yf_v); close(file)
        end
    end
    return
end

generate_grf2D_multixpu()
