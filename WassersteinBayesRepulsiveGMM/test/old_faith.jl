using CSV
using DataFrames
using LinearAlgebra
using MAT 
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
	X = Matrix(data[:, 2:3]) |> transpose |> Matrix	 

	dim = size(X, 1) 	
	g₀ = 10.
	β = 1.
	τ = 0.1
	a₀ = 1.
	b₀ = 1.
	l_σ2 = 0.001
	u_σ2 = 100.
	K = 5
 
	C_mc, Mu_mc, Sig_mc, K_mc, llhd_mc = blocked_gibbs(
		X; g₀=g₀, K=K, β=β, τ=τ, a₀=a₀, b₀=b₀, 
		l_σ2=l_σ2, u_σ2=u_σ2,
		burnin=2, runs=5, thinning=1) 

	println(C_mc[end])
	println(countmap(C_mc[end]))
	# # println(Mu_mc[950:end])
	# # println(Sigma_mc[950:end])

	mkpath("results/")
 	matwrite("results/old_faithful.mat", 
 		Dict(
			"K_mc" => K_mc,
			"C_mc" => C_mc,
			"Mu_mc" => Mu_mc,
			"Sig_mc" => Sig_mc
		); version=7.3
	)

	plots = [ 
		scatter(
			data[:, 2], data[:, 3], 
			legend=:none,
			cmap=:summer, 
			marker_z=C_mc[end-i], 
			markersize=3
		)
		for i in 0:3
	] 
	p = plot(plots..., layout=4)
	savefig(p, "test.pdf") 
end 


main() 
