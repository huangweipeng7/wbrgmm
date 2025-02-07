module WassersteinBayesRepulsiveGMM

export blocked_gibbs, NormalInverseWishart

import Base.rand

using Distributions 
using LinearAlgebra
using PDMats
using ProgressBars
using Random 
using Statistics

include("niw.jl")
include("measure/wasserstein.jl")
include("mcmc/blocked_gibbs.jl")

end # module WassersteinBayesRepulsiveGMM
