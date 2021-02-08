const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
const GPU_ID  = 0
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(GPU_ID) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using MAT, Plots
##################################################
@views function generate_RndField2D_expon(lx::Data.Number, ly::Data.Number, sf::Data.Number, cl, nh::Int, nx::Int, ny::Int, dx::Data.Number, dy::Data.Number; do_viz=false, do_save=false, do_reset=true)
    # Array allocation
    Yf = @zeros(nx, ny)
    # Preparation of visualisation
    if do_viz
        ENV["GKSwstype"]="nul"; if isdir("viz2D_exp")==false mkdir("viz2D_exp") end; loadpath = "./viz2D_exp/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
        X, Y = -lx/2:dx:lx/2, -ly/2:dy:ly/2
    end

    RndField2D_expon!(Yf, sf, cl, nh, nx, ny, dx, dy; do_reset=true)

    # Visualisation
    if do_viz
        display(heatmap(X, Y, Array(Yf)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:hot, title="3D RandomField (y-slice)"))
        # heatmap(X, Y, Array(Yf)', aspect_ratio=1, xlims=(X[1],X[end]), ylims=(Y[1],Y[end]), c=:hot, title="2D RandomField"); frame(anim)
        # gif(anim, "RandomField2D_exp.gif", fps = 15)
    end
    if do_save  file = matopen("RndF2De.mat", "w"); write(file, "RndF2D", Array(Yf)); close(file)  end
    return
end
