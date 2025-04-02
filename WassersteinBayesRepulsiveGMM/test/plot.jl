using OrderedCollections
using DataFrames 
using Gadfly  
import Plots, StatsPlots   
using JLD2   
using MLStyle
using Distributions, StatsBase, StatsFuns
using WassersteinBayesRepulsiveGMM 
 
import PlotlyKaleido   
import Cairo, Compose, Fontconfig
import ColorSchemes as cs

include("./data.jl"); import .load_data
include("./parse_args.jl"); import .parse_plot_cmd
 
   
function plot_density_estimate(X, mc_samples, kwargs)
    Plots.gr()

    dataname = kwargs["dataname"]
    method = kwargs["method"]

    # Generate grid
    x_min, x_max = minimum(X[1, :]) - 1, maximum(X[1, :]) + 1
    y_min, y_max = minimum(X[2, :]) - 1, maximum(X[2, :]) + 1
    x_grid = range(x_min, x_max, length=120)
    y_grid = range(y_min, y_max, length=120) 
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
            p[i] = sum(component_densities) 
        end 
        return mean(p)
    end

    density = zeros(size(grid_points, 1)) 
    for i in 1:size(grid_points, 1)
        density[i] = compute_density(grid_points[i, :]) 
    end
    density_matrix = reshape(density, (length(y_grid), length(x_grid)))
    println("Finish processing the density estimation computation")
 
    method = uppercase(method)
    # Plot
    logcpo = round(
        mean([mc_sample.llhd for mc_sample in mc_samples]), 
        digits=3) 
    p = Plots.scatter(X[1, :], X[2, :],  
        markercolor=:white,
        # color=:black, alpha=0.3,  
        markersize=2, label="log-CPO: $(logcpo)")
    Plots.contour!(
        x_grid, y_grid, density_matrix, 
        cmap=:linear_tritanopic_krjcw_5_98_c46_n256,
        levels=50, linewidth=0.7, alpha=0.9)
 
    Plots.title!("Density Estimate by $(method)") 
    
    println("Finish plotting\n\n\n") 
    Plots.savefig(p, "./plots/$(dataname)_$(method)_contour.pdf") 
end 


function plot_min_d_all(X, mc_sample_dict, kwargs)
    function compute(mc_samples)
        min_d_vec = zeros(length(mc_samples))
        for (k, mc_sample) in enumerate(mc_samples)
            K = size(mc_sample.Mu, 2)  
            d_mat = fill(Inf, K, K)
            indices = filter(c -> c[1] < c[2], CartesianIndices((1:K, 1:K)))
            Threads.@threads for (i, j) in Tuple.(indices) 
                d = (mc_sample.Mu[:, i] .- mc_sample.Mu[:, j]) .^2 |> sum |> sqrt
                d_mat[i, j] = d 
            end 
            min_d_vec[k] = minimum(d_mat)  
        end
        min_d_vec 
    end 

    # Plots.plotlyjs()
    Plots.gr()
    is_first = true
    p = nothing 
    ls = [:solid, :dash, :dot, :dashdotdot]
    for (i, (method, mc_samples)) in enumerate(mc_sample_dict)  
        method = uppercase(method)
        df = DataFrame(x=compute(mc_samples), method=method) 
    
        if is_first
            p = Plots.density(df.x, label=method,
                color=:black, tickfontsize=11, lw=1.5, 
                top_margin=5Plots.mm, linestyle=ls[i],
                title="Density of minimal mean distance")
            is_first = false 
        else
            Plots.density!(df.x, label=method, 
                color=:black, tickfontsize=11, lw=1.5,
                linestyle=ls[i])
        end 
    end 

    dataname = kwargs["dataname"]
    # draw(PDF("$(dataname)_min_dist_kde.pdf", 4inch, 3inch), p) 
    println("Finish plotting\n\n\n") 
    # PlotlyKaleido.start()
    Plots.savefig("./plots/$(dataname)_min_dist_kde.pdf")
end 


function plot_min_d(X, mc_samples, kwargs)
    dataname = kwargs["dataname"]
    method = kwargs["method"]

    rep_type = @match method begin
        "mean"          => "MRGM"
        "brgm"          => "BRGM"
        "wasserstein"   => "WRGM"
        "no"            => "DPGM"
    end 

    min_d_vec = zeros(length(mc_samples))
    for (k, mc_sample) in enumerate(mc_samples)
        K = size(mc_sample.Mu, 2)  
        d_mat = fill(Inf, K, K)
        indices = filter(c -> c[1] < c[2], CartesianIndices((1:K, 1:K)))
        Threads.@threads for (i, j) in Tuple.(indices) 
            d = (mc_sample.Mu[:, i] .- mc_sample.Mu[:, j]) .^2 |> sum |> sqrt
            d_mat[i, j] = d 
        end 
        min_d_vec[k] = minimum(d_mat)  
    end 
 
    # Plot 
    p = plot(x=min_d_vec, Theme(alphas=[0.6]),
        Stat.density, 
        Guide.title("KDE of minimal mean distance with $(method) Repulsion")) 
 
    draw(PDF("$(dataname)_min_dist.pdf", 4inch, 3inch), p)
    println("Finish plotting\n\n\n")
end 


function load_and_plot(kwargs)
    display(kwargs)  

    dataname = kwargs["dataname"]
    method = kwargs["method"]
    X = load_data(dataname) 
    
    mkpath("./plots/") 
         
    if method == "all"
        mc_sample_dict = OrderedDict( 
            (
                method, 
                JLD2.load("results/$(dataname)_$(method).jld2", "mc_samples")
            )
            for method in ["mean", "wasserstein", "no", "brgm"] 
        )
        plot_min_d_all(X, mc_sample_dict, kwargs) 
    else
        mc_samples = JLD2.load(
            "results/$(dataname)_$(method).jld2", "mc_samples") 
        plot_density_estimate(X, mc_samples, kwargs)
    end   
end 


load_and_plot(parse_plot_cmd())
