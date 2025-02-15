using CSV
using DataFrames
using LinearAlgebra 
using Random 
using Plots
using WassersteinBayesRepulsiveGMM  

using BenchmarkTools
using Profile

Random.seed!(100)


function main()
	data = CSV.File("faithful_data.csv") |> DataFrame 
	X = Matrix{Float32}(data[:, 2:3]) |> transpose |> Matrix

	dim = size(X, 1) 
	
	g₀ = 100f0  
	α = 10f0
	K = 5

	# Profile.init()
	# @profile begin 
	C_mc, Mu_mc, Sigma_mc, llhd_mc = 
		blocked_gibbs(
			X; g₀=g₀, K=K, α=α, 
			burnin=1000, runs=1500, thinning=10)
	# end 
	println(C_mc[end])
	# # println(Mu_mc[950:end])
	# # println(Sigma_mc[950:end])

	p = plot(1:length(llhd_mc), llhd_mc)
	savefig(p, "test.pdf")
    # Profile.print(format=:flat, groupby=:task)
end 


main() 
