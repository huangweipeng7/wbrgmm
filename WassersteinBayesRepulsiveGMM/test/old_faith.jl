using CSV
using DataFrames
using LinearAlgebra 
using Plots
using Random 
using StatsBase
using WassersteinBayesRepulsiveGMM  

using BenchmarkTools
using Profile

Random.seed!(200)


function main()
	data = CSV.File("faithful_data.csv") |> DataFrame 
	X = Matrix(data[:, 2:3]) |> transpose |> Matrix	 

	dim = size(X, 1) 
	
	g₀ = 10.
	β = 1.
	τ = 0.1
	a₀ = 1.
	b₀ = 1.
	l_σ2 = 0.001
	u_σ2 = 1000.
	K = 5
 
	C_mc, Mu_mc, Sigma_mc, llhd_mc = blocked_gibbs(
		X; g₀=g₀, K=K, β=β, τ=τ, a₀=a₀, b₀=b₀, 
		l_σ2=l_σ2, u_σ2=u_σ2,
		burnin=5000, runs=7500, thinning=10) 

	println(C_mc[end])
	println(countmap(C_mc[end]))
	# # println(Mu_mc[950:end])
	# # println(Sigma_mc[950:end])

	# p = plot(1:length(llhd_mc), llhd_mc)
	plots = []
	for i in 0:3
		p = scatter(data[:, 2], data[:, 3], marker_z=C_mc[end-i])
		push!(plots, p)
	end 
	p = plot(plots..., layout=4)

	savefig(p, "test.pdf") 
end 


main() 
