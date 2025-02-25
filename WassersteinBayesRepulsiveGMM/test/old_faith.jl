using CSV
using DataFrames
using LinearAlgebra 
using Plots, StatsPlots
using Random 
using StatsBase
using WassersteinBayesRepulsiveGMM  

using BenchmarkTools
using Profile

gr()
Random.seed!(200)


function main()
    data = CSV.File("./data/faithful_data.csv") |> DataFrame 
    X = Matrix(data[:, 2:3]) |> transpose  	 

    dim = size(X, 1) 	
    g₀ = 10
    β = 1.
    τ = 0.1
    a₀ = 1.
    b₀ = 1.
    l_σ2 = 0.001
    u_σ2 = 100.
    K = 5

    C_mc, Mu_mc, Sigma_mc, K_mc, llhd_mc = wrbgmm_blocked_gibbs(
        X; g₀=g₀, K=K, β=β, τ=τ, a₀=a₀, b₀=b₀, 
        l_σ2=l_σ2, u_σ2=u_σ2, burnin=2500, runs=5000, thinning=1) 

    println(
        "Cluster distribution from the last iteration: ", countmap(C_mc[end])) 

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
