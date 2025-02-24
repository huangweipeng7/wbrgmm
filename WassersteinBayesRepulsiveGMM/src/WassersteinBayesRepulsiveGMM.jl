module WassersteinBayesRepulsiveGMM

export blocked_gibbs
 
using Distributions 
using FStrings
using LinearAlgebra 
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
include("mcmc/blocked_gibbs.jl")
include("mcmc/prior_sampler.jl")


dlogpdf = Distributions.logpdf

const GUMBEL = Gumbel(0, 1)

end # module WassersteinBayesRepulsiveGMM
