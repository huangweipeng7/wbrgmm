using CSV
using DataFrames
using Distributions
using LinearAlgebra  
# using Gadfly
using CodecBzip2
using RDatasets
using Random 
using StatsBase, StatsFuns
using WassersteinBayesRepulsiveGMM  

using Plots, StatsPlots
# import Cairo, Fontconfig
# import ColorSchemes as cs
 
# gr()
# theme(:wong2)
Random.seed!(20)


function main()
    # data = CSV.File("./data/faithful_data.csv") |> DataFrame 
    # X = Matrix(data[!, 2:3]) |> transpose      
   
    data = dataset("datasets", "a1")
    X = Matrix(data[!, 1:2]) |> transpose  	
    X = X ./ 1000 
    dim = size(X, 1) 
    
    # Interesting hyper settings 
    g₀ = 0.1
    β = 1
    τ = 1
    κ = 1 
    l_σ2 = 1e-6
    u_σ2 = 1e6
    K = 30 
    ν₀ = 3

    prior = KernelPrior(
        τ, zeros(dim), τ^2*I(dim), l_σ2, u_σ2, ν₀, κ^2*I(dim))

    C_mc, Mu_mc, Sig_mc, K_mc, llhd_mc = wrbgmm_blocked_gibbs(
        X; g₀=g₀, K=K, β=β, τ=τ, prior=prior, t_max=5,
        l_σ2=l_σ2, u_σ2=u_σ2, burnin=200, runs=300, thinning=1) 

    println(
        "Cluster distribution from the last iteration: \n", 
        countmap(C_mc[end])) 

    mkpath("results/old_faithful/")
    C_df = DataFrame(C_mc, :auto)
    CSV.write("results/old_faithful/C_mc.csv", C_df)
    K_df = DataFrame(K_mc', :auto)
    CSV.write("results/old_faithful/K_mc.csv", K_df)
 
    p = plot_density_estimate(X, C_mc, Mu_mc, Sig_mc)
    savefig(p, "test_mean.pdf") 
end 


using Plots, Distributions


function plot_density_estimate(X, C_mc, Mu_mc, Sig_mc)
    # Generate grid
    x_min, x_max = minimum(X[1, :]) - 1, maximum(X[1, :]) + 1
    y_min, y_max = minimum(X[2, :]) - 1, maximum(X[2, :]) + 1
    x_grid = range(x_min, x_max, length=200)
    y_grid = range(y_min, y_max, length=200)
    xx = repeat(x_grid', length(y_grid), 1)
    yy = repeat(y_grid, 1, length(x_grid))
    grid_points = hcat(vec(xx), vec(yy)) 

    # Compute density for each grid point
    function compute_density(grid_point)
        total = 0.0
        for i in eachindex(Mu_mc)
            cnt = countmap(C_mc[i]) 
            π = [cnt[j] for j in 1:length(unique(C_mc[i]))]
            logπ = log.(π ./ sum(π))
            component_densities = [
                logπ[k] + logpdf(
                    MvNormal(Mu_mc[i][:, k], Sig_mc[i][:, :, k]), grid_point) 
                for k in eachindex(logπ)]
            total += exp(logsumexp(component_densities))
        end
        return total / length(Mu_mc)
    end

    density = zeros(size(grid_points, 1)) 
    Threads.@threads for i in 1:size(grid_points, 1)
        density[i] = compute_density(grid_points[i, :]) 
    end 
    density_matrix = reshape(density, (length(y_grid), length(x_grid)))
    println("Finish processing the DE computation")

    # Plot
    p = scatter(X[1, :], X[2, :],  
        color=:black, alpha=0.3, markersize=2, label="Data")
    contour!(x_grid, y_grid, density_matrix, 
        levels=20, c=:viridis, linewidth=1, alpha=0.8)
    title!("Density Estimate by Mean Repulsion")
    xlabel!("X"); ylabel!("Y")
    p 
end 


main() 
