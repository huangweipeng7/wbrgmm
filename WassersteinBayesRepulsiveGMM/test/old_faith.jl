using CSV
using DataFrames
using LinearAlgebra 
using Plots
using Random 
using StatsBase
using WassersteinBayesRepulsiveGMM  

using BenchmarkTools
using Profile

Random.seed!(100)


function main()
	data = CSV.File("faithful_data.csv") |> DataFrame 
	X = Matrix(data[:, 2:3]) |> transpose |> Matrix	 

	dim = size(X, 1) 
	
	g₀ = 10.
	β = 1.
	τ = 0.1
	a₀ = 1.
	b₀ = 1.
	K = 5

	# Profile.init()
	# @profile begin 
	C_mc, Mu_mc, Sigma_mc, llhd_mc = blocked_gibbs(
		X; g₀=g₀, K=K, β=β, τ=τ, a₀=a₀, b₀=b₀,
		burnin=5000, runs=7500, thinning=10)
	# end 
	println(C_mc[end])
	println(countmap(C_mc[end]))
	# # println(Mu_mc[950:end])
	# # println(Sigma_mc[950:end])

	p = plot(1:length(llhd_mc), llhd_mc)
	savefig(p, "test.pdf")
    # Profile.print(format=:flat, groupby=:task)
end 


main() 
