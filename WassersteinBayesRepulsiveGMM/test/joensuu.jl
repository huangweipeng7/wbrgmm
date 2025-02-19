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
	data = CSV.File("Joensuu.csv", delim=' ') |> DataFrame 
	X = Matrix(data[:, 1:2]) |> transpose |> Matrix	 

	dim = size(X, 1) 	
	g₀ = 0.005
	β = 1.
	τ = 0.1
	a₀ = 1.
	b₀ = 1.
	l_σ2 = 0.001
	u_σ2 = 100.
	K = 10
 
	C_mc, Mu_mc, Sigma_mc, K_mc, llhd_mc = blocked_gibbs(
		X; g₀=g₀, K=K, β=β, τ=τ, a₀=a₀, b₀=b₀, 
		l_σ2=l_σ2, u_σ2=u_σ2,
		burnin=1000, runs=1500, thinning=10) 

	println(C_mc[end])
	println(countmap(C_mc[end]))
	# # println(Mu_mc[950:end])
	# # println(Sigma_mc[950:end])

	p = density(K_mc)
	savefig(p, "test_K.pdf")  

	plots = []
	for i in 0:3
		p = scatter(
			data[:, 1], data[:, 2], 
			legend=:none,
			# cmap=:summer, 
			marker_z=C_mc[end-i], 
			markersize=3)
		push!(plots, p)
	end 
	p = plot(plots..., layout=4)
	savefig(p, "test.pdf") 
end 


main() 
