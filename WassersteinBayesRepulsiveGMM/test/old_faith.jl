using CSV
using DataFrames
using LinearAlgebra 
using WassersteinBayesRepulsiveGMM  


function main()
	data = CSV.File("faithful_data.csv") |> DataFrame 
	X = Matrix{Float64}(data[:, 2:3]) |> transpose |> Matrix

	dim = size(X, 1)
	prior = NormalInverseWishart(
		2, zeros(dim), 2, Matrix(1.0I, dim, dim)) 

	g₀ = 100  
	K = 5
	C_mc, Mu_mc, Sigma_mc = blocked_gibbs(
		X, prior, g₀, K; 
    	burnin=2500, runs=5000, thinning=5)

	# println(C_mc[950:end])
	# println(Mu_mc[950:end])
	# println(Sigma_mc[950:end])
end 


main() 