using CSV
using DataFrames
using Distributions
using JLD2 
using LinearAlgebra  
using Random
using StatsBase
using WassersteinBayesRepulsiveGMM  

include("./Data.jl"); import .Data
 
Random.seed!(20)
 

function main(datafile, method, kwargs...) 
    X = Data.load_data(datafile)
    println(size(X))
    dim = size(X, 1) 
    
    # Interesting hyper settings 
    g₀ = 1
    β = 1
    τ = 0.01
    κ = 1 
    l_σ2 = 1e-6
    u_σ2 = 1e6
    K = 10 
    ν₀ = 5

    prior = KernelPrior(
        τ, zeros(dim), τ^2*I(dim), l_σ2, u_σ2, ν₀, κ^2*I(dim))

    mc_samples = wrbgmm_blocked_gibbs(
        X; g₀=g₀, K=K, β=β, τ=τ, prior=prior, t_max=5, method=method,
        l_σ2=l_σ2, u_σ2=u_σ2, burnin=200, runs=300, thinning=1) 

    println(
        "Cluster distribution from the last iteration: \n", 
        countmap(mc_samples[end].C)) 

    mkpath("results/")
    jldsave("results/$(datafile)_$(method).jld2"; mc_samples))
 
    # p = plot_density_estimate(X, mc_samples)
    # # savefig(p, "$(datafile)_contour.pdf") 
    # draw(PDF("$(datafile)_contour.pdf", 7inch, 5inch), p) 
end 
 

if abspath(PROGRAM_FILE) == @__FILE__
    main("sim_data1", "mean")
end 
