using CSV
using DataFrames
using Distributions
using JLD2 
using LinearAlgebra  
using Random
using StatsBase
using WassersteinBayesRepulsiveGMM  

include("./data.jl"); import .load_data
include("./parse_args.jl"); import .parse_cmd

Random.seed!(20)
 

function main(kwargs) 
    display(kwargs)
    
    dataname = kwargs["dataname"]
    method = kwargs["method"]

    n_burnin = kwargs["n_burnin"]
    n_iter = kwargs["n_iter"]
    thinning = kwargs["thinning"]

    g₀ = kwargs["g0"]
    τ = kwargs["tau"]
    ν₀ = kwargs["nu0"]
    β = 1
    θ = 1 
    l_σ2, u_σ2 = 1e-8, 1e8
    K = 1

    X = load_data(dataname)
    dim = size(X, 1) 

    prior = KernelPrior(
        τ, zeros(dim), τ^2*I(dim), l_σ2, u_σ2, ν₀, θ^2*I(dim))

    mc_samples = wrbgmm_blocked_gibbs(
        X; g₀=g₀, K=K, β=β, τ=τ, prior=prior, 
        t_max=5, method=method, l_σ2=l_σ2, u_σ2=u_σ2, 
        n_burnin=n_burnin, n_iter=n_iter, thinning=thinning)  

    @info "Saving the MCMC samples" 
    mkpath("./results/")
    jldsave("results/$(dataname)_$(method).jld2"; mc_samples)  
    @info "Process finished" 
end 
  

main(parse_cmd()) 