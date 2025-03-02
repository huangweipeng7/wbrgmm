using CSV
using DataFrames
using Distributions
using LinearAlgebra  
using Plots, StatsPlots
using Random 
using StatsBase
using WassersteinBayesRepulsiveGMM  

using BenchmarkTools
using Profile

gr()
Random.seed!(250)


function main()
    data = CSV.File("./data/faithful_data.csv") |> DataFrame 
    X = Matrix(data[:, 2:3]) |> transpose  	 
    dim = size(X, 1) 
    
    # Interesting hyper settings
    # g₀ = 0.1
    # β = 15.
    # τ = 0.1
    # a₀ = 1.
    # b₀ = 1.
    # l_σ2 = 1e-8
    # u_σ2 = 1e8
    # K = 10
    # κ₀ = 1
    # ν₀ = 2

    g₀ = 0.5
    β = 20.
    τ = 0.1
    a₀ = 1.
    b₀ = 1.
    l_σ2 = 1e-6
    u_σ2 = 1e6
    K = 10
    κ₀ = 1
    ν₀ = 2

    prior = EigBoundedNorInverseWishart(
        l_σ2, u_σ2, κ₀, zeros(Real, dim), ν₀, τ^2*I(dim))

    C_mc, Mu_mc, Sigma_mc, K_mc, llhd_mc = wrbgmm_blocked_gibbs(
        X; g₀=g₀, K=K, β=β, τ=τ, a₀=a₀, b₀=b₀, prior=prior, 
        l_σ2=l_σ2, u_σ2=u_σ2, burnin=2500, runs=5000, thinning=1) 

    println(
        "Cluster distribution from the last iteration: \n", 
        countmap(C_mc[end])) 

    mkpath("results/old_faithful/")
    C_df = DataFrame(C_mc, :auto)
    CSV.write("results/old_faithful/C_mc.csv", C_df)
    K_df = DataFrame(K_mc', :auto)
    CSV.write("results/old_faithful/K_mc.csv", K_df)

    plots = [
        scatter(
            data[:, 2], data[:, 3], 
            legend=:none,
            cmap=:summer, 
            marker_z=C_mc[end-i],  
            markersize=2.5,
            alpha=0.5)
        for i in 0:3  
    ]
    p = plot(plots..., layout=4)
    savefig(p, "test.pdf") 
end 


main() 
