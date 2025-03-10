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

	g₀ = 0.1
	β = 15.
	τ = 0.1
	a₀ = 1.
    b₀ = 1.
    l_σ2 = 1e-6
    u_σ2 = 1e6
    K = 5
    κ₀ = 1
    ν₀ = 3

    prior = EigBoundedNorInverseWishart(
        l_σ2, u_σ2, τ, zeros(Real, dim), τ^2*I(dim), ν₀, I(dim))

    C_mc, Mu_mc, Sigma_mc, K_mc, llhd_mc = wrbgmm_blocked_gibbs(
        X; g₀=g₀, K=K, β=β, τ=τ, a₀=a₀, b₀=b₀, prior=prior, 
        l_σ2=l_σ2, u_σ2=u_σ2, burnin=100, runs=300, thinning=1) 

    println(
        "Cluster distribution from the last iteration: ", countmap(C_mc[end])) 
 
    mkpath("results/joensuu/")
    C_df = DataFrame(C_mc, :auto)
    CSV.write("results/joensuu/C_mc.csv", C_df)
    K_df = DataFrame(K_mc', :auto)
    CSV.write("results/joensuu/K_mc.csv", K_df) 
    
    @inbounds plots = [
        begin
            println("Drawing the MCMC sample $(i+1)")
            println("The unique classes are $(unique(C_mc[end-i]))")

            p = scatter(
                data[:, 1], data[:, 2], 
                legend=:none,    
                marker_z=C_mc[end-i],  
                markersize=3,
                m=:+,
                title="MCMC Sample $(i+1)",
                alpha=0.8)

            for k in unique(C_mc[end-i])
                covellipse!(
                    Mu_mc[end-i][:, k], Sigma_mc[end-i][:, :, k], 
                    alpha=0.2, label=[k])
            end
            p 
        end 
        for i in 0:3
    ] 
    p = plot(plots..., layout=4, xlims=(60, 66), ylims=(26, 32))
    savefig(p, "test.pdf") 
end 


main() 
