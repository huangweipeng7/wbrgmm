module WassersteinBayesRepulsiveGMM

export wrbgmm_blocked_gibbs, EigBoundedNorInverseWishart

import Base.rand
 
using Distributions 
using FStrings
using LinearAlgebra 
using LoopVectorization
using PDMats
using ProgressBars
using Random 
using SpecialFunctions
using Statistics
using StatsBase 

include("log_p_K.jl")
include("logV.jl")
include("numerical_Zk.jl")
include("numerical_Zhat.jl")

include("measure/wasserstein.jl")
include("mcmc/niw.jl")
include("mcmc/blocked_gibbs.jl")
include("mcmc/prior_sampler.jl")


logpdf = Distributions.logpdf

const GUMBEL = Gumbel(0, 1)

gumbel_max_sample(logits) = argmax(logits + rand(GUMBEL, length(logits))) 

end # module WassersteinBayesRepulsiveGMM
