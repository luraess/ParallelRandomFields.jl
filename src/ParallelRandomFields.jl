"""
Module ParallelRandomFields

Enables to sample spatial realisations of a 2-D and 3-D Gaussian random fields with given power spectrum: anisotropic exponential and isotropic Gaussian covariance functions. 

# General overview and examples
https://github.com/luraess/ParallelRandomFields.jl

# Functions
- to come

To see a description of a function or a macro type `?<functionname>`.
"""

module ParallelRandomFields

# 2D generator
include("generate_RndField2D_expon.jl")

# 2D random field
include("RndField2D_expon.jl")


# include("RandomField2D_gauss.jl")

# 3D generator
# include("RandomField3D_expon.jl")
# include("RandomField3D_gauss.jl")

# 3D MPI generator


end # Module ParallelRandomFields
