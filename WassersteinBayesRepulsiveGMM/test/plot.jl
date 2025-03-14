using Gadfly 
using JLD2   
using Distributions,StatsBase, StatsFuns
using WassersteinBayesRepulsiveGMM 

import Cairo, Fontconfig
import ColorSchemes as cs

include("./Data.jl"); import .Data


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
    println("Finish processing the density estimation computation")

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
        Theme(key_position = :none), 
        Scale.color_discrete(
            n -> get(cs.linear_tritanopic_krjcw_5_98_c46_n256, 
                range(0, 1, length=n))))
    println("Finish plotting")
    p 
end 


function load_and_plot(datafile, method)
    mc_samples = load(
        "results/$(datafile)_$(method).jld2", "mc_samples")  
    X = Data.load_data(datafile)
    p = plot_density_estimate(X, mc_samples)
    draw(PDF("$(datafile)_$(method)_contour.pdf", 6.3inch, 5inch), p) 
end 


load_and_plot("sim_data1", "mean")