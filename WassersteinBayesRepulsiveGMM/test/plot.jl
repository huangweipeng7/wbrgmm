# using Gadfly 
using Plots, StatsPlots 
using JLD2   
using MLStyle
using Distributions, StatsBase, StatsFuns
using WassersteinBayesRepulsiveGMM 

# import Cairo, Compose, Fontconfig
# import ColorSchemes as cs

include("./data.jl"); import .load_data
include("./parse_args.jl"); import .parse_plot_cmd
 
   
function plot_density_estimate(X, mc_samples, kwargs)
    method = kwargs["method"]

    # Generate grid
    x_min, x_max = minimum(X[1, :]) - 1, maximum(X[1, :]) + 1
    y_min, y_max = minimum(X[2, :]) - 1, maximum(X[2, :]) + 1
    x_grid = range(x_min, x_max, length=100)
    y_grid = range(y_min, y_max, length=100)
    xx = repeat(x_grid', length(y_grid), 1)
    yy = repeat(y_grid, 1, length(x_grid))
    grid_points = hcat(vec(xx), vec(yy)) 

    # Compute density for each grid point
    function compute_density(grid_point)
        # total = 0.0
        p = zeros(length(mc_samples))
        Threads.@threads for i in eachindex(mc_samples)
            cnt = countmap(mc_samples[i].C) 
            pi = [cnt[j] for j in 1:length(unique(mc_samples[i].C))]
            pi = pi ./ sum(pi) 
            component_densities = [
                pi[k] * pdf(
                    MvNormal(
                        mc_samples[i].Mu[:, k], 
                        mc_samples[i].Sig[:, :, k]), 
                        grid_point) 
                for k in eachindex(pi)]
            # total += exp(logsumexp(component_densities))
            p[i] = sum(component_densities) 
        end
        # return total / length(mc_samples)
        return mean(p)
    end

    density = zeros(size(grid_points, 1)) 
    for i in 1:size(grid_points, 1)
        density[i] = compute_density(grid_points[i, :]) 
    end
    density_matrix = reshape(density, (length(y_grid), length(x_grid)))
    println("Finish processing the density estimation computation")


    rep_type = @match method begin
        "mean" => "MRGM"
        "wasserstein" => "WRGM"
        "no" => "DPGM"
    end 

    # Plot
    logcpo = round(
        mean([mc_sample.llhd for mc_sample in mc_samples]), 
        digits=3)
    p = scatter(X[1, :], X[2, :],  
         markercolor=:white,
        # color=:black, alpha=0.3, 
        markersize=2.5, label="log-CPO: $(logcpo)")
    contour!(x_grid, y_grid, density_matrix, cmap=:viridis,
        levels=20, linewidth=1, alpha=0.8)
    title!("Density Estimate by $(rep_type)")
    xlabel!("X"); ylabel!("Y")

    # method = uppercase(method[1]) * method[2:end]
    # p = Gadfly.plot(  
    #     Coord.cartesian(
    #         xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max), 
    #     layer(z=density_matrix, x=x_grid, y=y_grid, alpha=[0.8],
    #         Geom.contour(levels=15), Theme(minor_label_font_size=16pt)), 
    #     layer(x=X[1, :], y=X[2, :], Geom.point, 
    #         Theme(
    #             default_color="white", 
    #             discrete_highlight_color=c->["black"])),
    #     Guide.ylabel(nothing), Guide.xlabel(nothing),
    #     Theme(key_position=:none),   
    #     Scale.color_discrete(
    #         n -> get(cs.linear_tritanopic_krjcw_5_98_c46_n256, 
    #             range(0, 1, length=n))))
    # p = title(
    #     render(p), 
    #     "Density Estimation with $(method) Repulsion", 
    #     Compose.fontsize(17pt))
    println("Finish plotting\n\n\n")
    p 
end 


function load_and_plot(kwargs)
    display(kwargs)  

    dataname = kwargs["dataname"]
    method = kwargs["method"]
    
    mc_samples = JLD2.load(
        "results/$(dataname)_$(method).jld2", "mc_samples")  
    display(mc_samples[end].Mu)
    display(mc_samples[end].Sig)
    X = load_data(dataname)
    p = plot_density_estimate(X, mc_samples, kwargs)
    # draw(PDF("$(dataname)_$(method)_contour.pdf", 6.3inch, 5inch), p) 
    savefig(p, "$(dataname)_$(method)_contour.pdf")
end 


load_and_plot(parse_plot_cmd())