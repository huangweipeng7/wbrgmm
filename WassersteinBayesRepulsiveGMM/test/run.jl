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
    β = 1
    τ = kwargs["tau"]
    κ = 1 
    l_σ2 = 1e-8
    u_σ2 = 1e8
    K = 10 
    ν₀ = kwargs["nu0"]

    X = load_data(dataname)
    dim = size(X, 1) 

    prior = KernelPrior(
        τ, zeros(dim), τ^2*I(dim), l_σ2, u_σ2, ν₀, κ^2*I(dim))

    mc_samples = wrbgmm_blocked_gibbs(
        X; g₀=g₀, K=K, β=β, τ=τ, prior=prior, 
        t_max=5, method=method, l_σ2=l_σ2, u_σ2=u_σ2, 
        n_burnin=n_burnin, n_iter=n_iter, thinning=thinning) 

    # println(
    #     "Cluster distribution from the last iteration: \n", 
    #     countmap(mc_samples[end].C)) 

    @info "Saving the mcmc samples" 
    mkpath("results/")
    jldsave("results/$(dataname)_$(method).jld2"; mc_samples)  
    @info "Process finished" 
end 
 

if abspath(PROGRAM_FILE) == @__FILE__ 
    main(parse_cmd())
end 
