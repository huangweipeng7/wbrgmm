using OrderedCollections
using DataFrames 
using Gadfly  
import Plots, StatsPlots
using LinearAlgebra   
using JLD2   
using MLStyle
using Distributions, StatsBase, StatsFuns
using WassersteinBayesRepulsiveGMM 
 
import PlotlyKaleido   
import Cairo, Compose, Fontconfig
import ColorSchemes as cs

include("./data.jl"); import .load_data
include("./parse_args.jl"); import .parse_plot_cmd



function wass_dist(μ₁, Σ₁, μ₂, Σ₂) 
    Σ₂_sqrt = sqrt(Σ₂) 
    Σ_part_sqrt = 2 * sqrt(Float64.(Σ₂_sqrt * Σ₁ * Σ₂_sqrt))   
    μ_part = (μ₁ .- μ₂) .^ 2 
    Σ_part = Σ₁ .+ Σ₂ .- Σ_part_sqrt 
    return sqrt(sum(μ_part) + tr(Σ_part)) 
end


function getellipsepoints(cx, cy, rx, ry, θ)
    t = range(0, 2*pi, length=100)
    ellipse_x_r = @. rx * cos(t)
    ellipse_y_r = @. ry * sin(t)
    R = [cos(θ) sin(θ); -sin(θ) cos(θ)]
    r_ellipse = [ellipse_x_r ellipse_y_r] * R
    x = @. cx + r_ellipse[:,1]
    y = @. cy + r_ellipse[:,2]
    (x, y)
end


function getellipsepoints(μ, Σ, confidence=0.95)
    quant = quantile(Chisq(2), confidence) |> sqrt
    cx = μ[1]
    cy = μ[2]
    
    egvs = eigvals(Σ)
    if egvs[1] > egvs[2]
        idxmax = 1
        largestegv = egvs[1]
        smallesttegv = egvs[2]
    else
        idxmax = 2
        largestegv = egvs[2]
        smallesttegv = egvs[1]
    end

    rx = quant*sqrt(largestegv)
    ry = quant*sqrt(smallesttegv)
    
    eigvecmax = eigvecs(Σ)[:,idxmax]
    θ = atan(eigvecmax[2]/eigvecmax[1])
    if θ < 0
        θ += 2*π
    end

    getellipsepoints(cx, cy, rx, ry, θ)
end

   
function plot_density_estimate(X, mc_samples, kwargs)
    
    dataname = kwargs["dataname"]
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

    Plots.plotlyjs()
    Plots.theme(:dao)

    logcpo = round(
        mean([mc_sample.llhd for mc_sample in mc_samples]), 
        digits=3) 
    p = Plots.scatter(X[1, :], X[2, :],  
        markercolor=:gray,
        markerstrokewidth=0,
        alpha=0.7,  
        tickfontsize=10,
        xlabel=ifelse(dataname=="GvHD", "CD8", "x"), 
        ylabel=ifelse(dataname=="GvHD", "CD4", "y"),
        markersize=2, label="log-CPO: $(logcpo)")
    Plots.contour!(
        x_grid, y_grid, density_matrix, 
        cmap=:bone, #:linear_tritanopic_krjcw_5_98_c46_n256,
        levels=35, linewidth=0.7, alpha=0.9, cbar=false)
 
    Plots.title!("Density Estimate by $(method)") 
    
    println("Finish plotting\n\n\n") 
    mkpath("./plots/$(dataname)")

    PlotlyKaleido.start()
    Plots.savefig(p, "./plots/$(dataname)/$(dataname)_$(method)_contour.pdf") 
end 


function plot_map_estimate(X, mc_samples, kwargs) 
    Plots.theme(:default)

    dataname = kwargs["dataname"]
    method = kwargs["method"]

    map_est_ind = map(x -> x.lpost, mc_samples) |> argmax
    mc_sample = mc_samples[map_est_ind] 
  
    method = uppercase(method)
    Plots.plotlyjs()
    p = Plots.scatter(X[1, :], X[2, :],  
        # markercolor=:white, 
        framestyle=:grid,
        markersize=2, 
        label=nothing, 
        tickfontsize=10,
        xlabel=ifelse(dataname=="GvHD", "CD8", "x"), 
        ylabel=ifelse(dataname=="GvHD", "CD4", "y"),
        color=mc_sample.C)
    for k in unique(mc_sample.C) 
        Plots.plot!(
            getellipsepoints(       
                mc_sample.Mu[:, k], mc_sample.Sig[:, :, k], 0.95
            ),
            color=:black, 
            label=nothing, 
        )
    end 

    Plots.title!("MAP Component Estimate by $(method)") 
    
    println("Finish plotting\n\n\n") 
    mkpath("./plots/$(dataname)")
    PlotlyKaleido.start()
    Plots.savefig(p, "./plots/$(dataname)/$(dataname)_$(method)_map.pdf") 
end 


function plot_min_d_all(X, mc_sample_dict, kwargs)
    Plots.theme(:dao)

    dist_type = kwargs["dist_type"] 
    
    function compute(mc_samples)
        min_d_vec = zeros(length(mc_samples))
        for (k, mc_sample) in enumerate(mc_samples)
            K = size(mc_sample.Mu, 2)  
            d_mat = fill(Inf, K, K)
            indices = filter(c -> c[1] < c[2], CartesianIndices((1:K, 1:K)))
            Threads.@threads for (i, j) in Tuple.(indices) 
                if dist_type == "Mean"
                    d = (mc_sample.Mu[:, i] .- mc_sample.Mu[:, j]) .^2 |> sum |> sqrt
                elseif dist_type == "Wasserstein"  
                    d = wass_dist(
                        mc_sample.Mu[:, i], mc_sample.Sig[:, :, i],
                        mc_sample.Mu[:, j], mc_sample.Sig[:, :, j])
                else 
                    throw("Distance type not supported")
                end 
                d_mat[i, j] = d 
            end 
            min_d_vec[k] = minimum(d_mat)  
        end
        min_d_vec 
    end 

    Plots.plotlyjs()
    # Plots.gr()
    is_first = true
    p = nothing 
    ls = [:solid, :dash, :dot, :dashdot]
    for (i, (method, mc_samples)) in enumerate(mc_sample_dict)  
        method = uppercase(method)
        df = DataFrame(x=compute(mc_samples), method=method) 
    

        ls = Dict(
            "DPGM-FULL" => :solid, 
            "DPGM-DIAG" => :solid,
            "RGM-FULL"  => :dot, 
            "RGM-DIAG"  => :dot, 
            "WRGM-DIAG" => :dashdot, 
            "WRGM-FULL" => :dashdot,
        )
        if is_first
            p = Plots.density(df.x, label=method, 
                tickfontsize=10, lw=2, linestyle=ls[method], 
                top_margin=5Plots.mm,  
                title="Density of Minimal Inter-component $(dist_type) Distance")
            is_first = false 
        else
            Plots.density!(
                df.x, label=method, linestyle=ls[method], tickfontsize=10, lw=2       
            )
        end 
    end 

    dataname = kwargs["dataname"]
    println("Finish plotting\n\n\n") 

    mkpath("./plots/$(dataname)")
    PlotlyKaleido.start()
    Plots.savefig("./plots/$(dataname)/$(dataname)_$(dist_type)_min_dist_kde.pdf")
end 


function plot_min_d(X, mc_samples, kwargs)
    dataname = kwargs["dataname"]
    method = kwargs["method"] 

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
 
    mkpath("./plots/$(dataname)")
    draw(PDF("./plots/$(dataname)/$(dataname)_min_dist.pdf", 4inch, 3inch), p)
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
            for method in [
                "dpgm-diag", "dpgm-full", "rgm-diag", "rgm-full","wrgm-diag", "wrgm-full"
            ] 
        )
        plot_min_d_all(X, mc_sample_dict, kwargs) 
    else
        mc_samples = JLD2.load(
            "results/$(dataname)_$(method).jld2", "mc_samples") 
        plot_density_estimate(X, mc_samples, kwargs)
        plot_map_estimate(X, mc_samples, kwargs)
    end   
end 


load_and_plot(parse_plot_cmd())
