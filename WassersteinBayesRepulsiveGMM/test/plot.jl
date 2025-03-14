using Gadfly 
# using Plots, StatsPlots 
using JLD2   
using MLStyle
using Distributions, StatsBase, StatsFuns
using WassersteinBayesRepulsiveGMM 

import Cairo, Compose, Fontconfig
import ColorSchemes as cs

include("./data.jl"); import .load_data
include("./parse_args.jl"); import .parse_plot_cmd
 
   
function plot_density_estimate(X, mc_samples, kwargs)
    method = kwargs["method"]

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
        # total = 0.0
        logp = zeros(length(mc_samples))
        Threads.@threads for i in eachindex(mc_samples)
            cnt = countmap(mc_samples[i].C) 
            π = [cnt[j] for j in 1:length(unique(mc_samples[i].C))]
            logπ = log.(π ./ sum(π))
            component_densities = [
                logπ[k] + logpdf(
                    MvNormal(mc_samples[i].Mu[:, k], mc_samples[i].Sig[:, :, k]), 
                    grid_point) 
                for k in eachindex(logπ)]
            # total += exp(logsumexp(component_densities))
            logp[i] = exp(logsumexp(component_densities))
        end
        # return total / length(mc_samples)
        return mean(logp)
    end

    density = zeros(size(grid_points, 1)) 
    for i in 1:size(grid_points, 1)
        density[i] = compute_density(grid_points[i, :]) 
    end
    density_matrix = reshape(density, (length(y_grid), length(x_grid)))
    println("Finish processing the density estimation computation")

    # # Plot
    # p = scatter(X[1, :], X[2, :],  
    #      markercolor=:white,
    #     # color=:black, alpha=0.3, 
    #     markersize=2.5, label="Data")
    # contour!(x_grid, y_grid, density_matrix, cmap=:linear_tritanopic_krjcw_5_98_c46_n256,
    #     levels=15, linewidth=1, alpha=0.8)
    # title!("Density Estimate by Mean Repulsion")
    # xlabel!("X"); ylabel!("Y")
 
    colorkey = @match method begin
        "mean" => "MRGM"
        "wasserstein" => "WRGM"
        "no" => "DPGM"
    end 
    method = uppercase(method[1]) * method[2:end]
    p = Gadfly.plot(  
        Coord.cartesian(
            xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max), 
        layer(x=X[1, :], y=X[2, :], Geom.point, size=2.5, 
            Theme(
                default_color="white", 
                discrete_highlight_color=c->["black"], 
                minor_label_font_size=16pt)),
        layer(z=density_matrix, x=x_grid, y=y_grid, 
            Geom.contour(levels=15)), 
        Guide.ylabel(nothing), Guide.xlabel(nothing),
        Theme(key_position=:none),   
        Scale.color_discrete(
            n -> get(cs.linear_tritanopic_krjcw_5_98_c46_n256, 
                range(0, 1, length=n))))
    p = title(
        render(p), 
        "Density Estimation with $(method) Repulsion", 
        Compose.fontsize(17pt))
    println("Finish plotting")
    p 
end 


function load_and_plot(kwargs)
    display(kwargs)  

    dataname = kwargs["dataname"]
    method = kwargs["method"]
    
    mc_samples = JLD2.load(
        "results/$(dataname)_$(method).jld2", "mc_samples")  
    X = load_data(dataname)
    p = plot_density_estimate(X, mc_samples, kwargs)
    draw(PDF("$(dataname)_$(method)_contour.pdf", 6.3inch, 5inch), p) 
    # savefig(p, "test.pdf")
end 


load_and_plot(parse_plot_cmd())