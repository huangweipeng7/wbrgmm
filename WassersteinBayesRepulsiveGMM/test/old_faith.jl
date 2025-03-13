using CSV
using DataFrames
using Distributions
using JLD2 
using LinearAlgebra  
using Gadfly  
using Random 
using StatsBase, StatsFuns
using WassersteinBayesRepulsiveGMM  

# using Plots, StatsPlots
import Cairo, Fontconfig
import ColorSchemes as cs
 
# gr()
# theme(:wong2)
Random.seed!(20)


function load_data(datafile) 
    df = "./data/$(datafile).csv" |> CSV.File |> DataFrame
    
    df_ = nothing
    if datafile == "faithful_data.csv"
        df_ = df[!, 2:3]
    else 
        df_ = df[!, 1:2]
    end 

    return df_ |> Matrix |> transpose 
end 


function main(datafile, method, kwargs...) 
    X = load_data(datafile)
    println(size(X))
    dim = size(X, 1) 
    
    # Interesting hyper settings 
    g₀ = 1
    β = 1
    τ = 0.01
    κ = 1 
    l_σ2 = 1e-6
    u_σ2 = 1e6
    K = 10 
    ν₀ = 5

    prior = KernelPrior(
        τ, zeros(dim), τ^2*I(dim), l_σ2, u_σ2, ν₀, κ^2*I(dim))

    mc_samples = wrbgmm_blocked_gibbs(
        X; g₀=g₀, K=K, β=β, τ=τ, prior=prior, t_max=5, method=method,
        l_σ2=l_σ2, u_σ2=u_σ2, burnin=200, runs=300, thinning=1) 

    println(
        "Cluster distribution from the last iteration: \n", 
        countmap(mc_samples[end].C)) 

    mkpath("results/")
    jldsave("results/$(datafile)_$(method).jld2"; mc_samples)
 
    # p = plot_density_estimate(X, mc_samples)
    # # savefig(p, "$(datafile)_contour.pdf") 
    # draw(PDF("$(datafile)_contour.pdf", 7inch, 5inch), p) 
end 
 

function plot_density_estimate(X, mc_samples)
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
        for i in eachindex(mc_samples)
            cnt = countmap(mc_samples[i].C) 
            π = [cnt[j] for j in 1:length(unique(mc_samples[i].C))]
            logπ = log.(π ./ sum(π))
            component_densities = [
                logπ[k] + logpdf(
                    MvNormal(mc_samples[i].Mu[:, k], mc_samples[i].Sig[:, :, k]), 
                    grid_point) 
                for k in eachindex(logπ)]
            total += exp(logsumexp(component_densities))
        end
        return total / length(mc_samples)
    end

    density = zeros(size(grid_points, 1)) 
    Threads.@threads for i in 1:size(grid_points, 1)
        density[i] = compute_density(grid_points[i, :]) 
    end
    density_matrix = reshape(density, (length(y_grid), length(x_grid)))
    println("Finish processing the DE computation")

    # Plot
    # p = scatter(X[1, :], X[2, :],  
    #     color=:black, alpha=0.3, markersize=2, label="Data")
    # contour!(x_grid, y_grid, density_matrix, 
    #     levels=20, linewidth=1, alpha=0.8)
    # title!("Density Estimate by Mean Repulsion")
    # xlabel!("X"); ylabel!("Y")
 
    p = Gadfly.plot(  
        Coord.cartesian(
            xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max), 
        layer(x=X[1, :], y=X[2, :], Geom.point, alpha=[0.5],
            Theme(default_color="black")),
        layer(z=density_matrix, x=x_grid, y=y_grid, 
            Geom.contour(levels=15)), 
        Guide.ylabel(nothing), Guide.xlabel(nothing), 
        Scale.color_discrete(n -> get(cs.linear_tritanopic_krjcw_5_98_c46_n256, range(0, 1, length=n)))
        )
    println("Finish plotting")
    p 
end 


# if abspath(PROGRAM_FILE) == @__FILE__
#     main("sim_data1", "mean")
# end 

function load_and_plot(datafile, method)
    mc_samples = load(
        "results/$(datafile)_$(method).jld2", "mc_samples")  
    X = load_data(datafile)
    p = plot_density_estimate(X, mc_samples)
    draw(PDF("$(datafile)_$(method)_contour.pdf", 7inch, 5inch), p) 
end 


load_and_plot("sim_data1", "mean")