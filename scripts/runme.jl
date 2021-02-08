const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
const GPU_ID  = 0

using ParallelRandomFields

## Run 2D random field
# Physics
lx, ly = 100.0, 100.0  # domain size
sf     = 1.0           # standard deviation
# -- exponential setup
cl_e   = (10.0, 8.0)   # correlation lengths in [x,y]
# -- gaussian setup
cl_g   = 5.0           # correlation length isotropic
k_m    = 100.0         # maximum value of the wave number
# Numerics
nh     = 10000         # inner parameter, number of harmonics
nx, ny = 128, 128      # numerical grid resolution
# Derived numerics
dx, dy = lx/nx, ly/ny  # numerical grid step size

# Run the 2D exponential covariance function
generate_RndField2D_expon(lx, ly, sf, cl_e, nh, nx, ny, dx, dy; do_viz=true)


# # Run the 2D gaussian covariance function
# RandomField2D_gauss(lx, ly, sf, cl_g, k_m, nh, nx, ny, dx, dy; do_viz=false, do_save=false, do_reset=true)


# ## Run 3D random field
# # Physics
# lx, ly, lz = 100.0, 100.0, 100.0  # domain size
# sf         = 1.0                  # standard deviation
# cl_e       = (10.0, 8.0, 5.0)     # correlation lengths in [x,y,z]
# cl_g       = 5.0                  # correlation length isotropic
# k_m        = 100.0                # maximum value of the wave number
# # Numerics
# nh         = 10000                # inner parameter, number of harmonics
# nx, ny, nz = 64, 64, 64           # numerical grid resolution
# # Derived numerics
# dx, dy = lx/nx, ly/ny, lz/dz      # numerical grid step size

# # Run the 3D exponential covariance function
# RandomField3D_expon(lx, ly, lz, sf, cl_e, nh, nx, ny, nz, dx, dy, dz; do_viz=false, do_save=false, do_reset=true)
# # Run the 3D gaussian covariance function
# RandomField3D_gauss(lx, ly, lz, sf, cl_g, k_m, nh, nx, ny, nz, dx, dy, dz; do_viz=false, do_save=false, do_reset=true)
