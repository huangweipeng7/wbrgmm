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
	data = CSV.File("./data/joensuu.csv", delim=' ') |> DataFrame 
	X = Matrix(data[:, 1:2]) |> transpose 
	dim = size(X, 1) 	

	g₀ = 5
	β = 20.
	τ = 0.1
	a₀ = 1.
    b₀ = 1.
    l_σ2 = 1e-6
    u_σ2 = 1e6
    K = 5
    κ₀ = 1
    ν₀ = 2

    prior = EigBoundedNorInverseWishart(
        l_σ2, u_σ2, κ₀, zeros(Real, dim), ν₀, τ^2*I(dim))

    C_mc, Mu_mc, Sigma_mc, K_mc, llhd_mc = wrbgmm_blocked_gibbs(
        X; g₀=g₀, K=K, β=β, τ=τ, a₀=a₀, b₀=b₀, prior=prior, 
        l_σ2=l_σ2, u_σ2=u_σ2, burnin=100, runs=500, thinning=1) 

	println(
		"Cluster distribution from the last iteration: ", countmap(C_mc[end])) 
 
	mkpath("results/joensuu/")
	C_df = DataFrame(C_mc, :auto)
	CSV.write("results/joensuu/C_mc.csv", C_df)
	K_df = DataFrame(K_mc', :auto)
	CSV.write("results/joensuu/K_mc.csv", K_df) 
    
    plots = [
        scatter(
            data[:, 1], data[:, 2], 
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
