module WassersteinBayesRepulsiveGMM

export blocked_gibbs, NormalInverseWishart

import Base.rand

using Profile
using Distributions 
using LinearAlgebra
using PDMats
using ProgressBars
using Random 
using SpecialFunctions
using Statistics
using StatsBase 

include("log_p_K.jl")
include("logV.jl")
include("mc_Zk.jl")
include("numerical_Zhat.jl")

include("measure/wasserstein.jl")
include("mcmc/blocked_gibbs.jl")
include("mcmc/prior_sampler.jl")


dlogpdf = Distributions.logpdf

const GUMBEL = Gumbel(0, 1)

end # module WassersteinBayesRepulsiveGMM
