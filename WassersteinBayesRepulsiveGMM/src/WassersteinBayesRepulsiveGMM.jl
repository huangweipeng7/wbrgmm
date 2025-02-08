module WassersteinBayesRepulsiveGMM

export blocked_gibbs, NormalInverseWishart

import Base.rand

using Distributions 
using LinearAlgebra
using PDMats
using ProgressBars
using Random 
using Statistics
using StatsBase

include("niw.jl")
include("measure/wasserstein.jl")
include("mcmc/blocked_gibbs.jl")

dlogpdf = Distributions.logpdf

const GUMBEL = Gumbel(0, 1)

end # module WassersteinBayesRepulsiveGMM
