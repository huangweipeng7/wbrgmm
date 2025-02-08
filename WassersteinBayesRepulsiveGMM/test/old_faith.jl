using CSV
using DataFrames
using LinearAlgebra 
using Random 
using Plots
using WassersteinBayesRepulsiveGMM  


Random.seed!(100)


function main()
	data = CSV.File("faithful_data.csv") |> DataFrame 
	X = Matrix{Float64}(data[:, 2:3]) |> transpose |> Matrix

	dim = size(X, 1)
	prior = NormalInverseWishart(
		2, zeros(dim), 2, Matrix(1.0I, dim, dim)) 

	g₀ = 100.  
	α = 10.
	K = 5
	C_mc, Mu_mc, Sigma_mc, llhd_mc = blocked_gibbs(
		X, prior; 
		g₀=g₀, K=K, α=α, 
		burnin=2500, runs=5000, thinning=10)

	println(C_mc[end])
	# println(Mu_mc[950:end])
	# println(Sigma_mc[950:end])

	p = plot(1:length(llhd_mc), llhd_mc)
	savefig(p, "test.pdf")
end 


main() 