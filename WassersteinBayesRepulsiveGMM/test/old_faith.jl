using CSV
using DataFrames
using Distributions
using LinearAlgebra  
# using Gadfly
using Random 
using StatsBase
using WassersteinBayesRepulsiveGMM  

using Plots, StatsPlots
# import Cairo, Fontconfig
# import ColorSchemes as cs
 
# gr()
# theme(:wong2)
Random.seed!(250)


function main()
    data = CSV.File("./data/faithful_data.csv") |> DataFrame 
    X = Matrix(data[!, 2:3]) |> transpose  	 
    dim = size(X, 1) 
    
    # Interesting hyper settings 
    g₀ = 50
    β = 1
    τ = 0.1 
    l_σ2 = 1e-6
    u_σ2 = 1e6
    K = 5 
    ν₀ = 10 

    prior = KernelPrior(
        τ, zeros(Real, dim), τ^2*I(dim), l_σ2, u_σ2, ν₀, I(dim))

    C_mc, Mu_mc, Sig_mc, K_mc, llhd_mc = wrbgmm_blocked_gibbs(
        X; g₀=g₀, K=K, β=β, τ=τ, prior=prior, t_max=5,
        l_σ2=l_σ2, u_σ2=u_σ2, burnin=2500, runs=5000, thinning=10) 

    println(
        "Cluster distribution from the last iteration: \n", 
        countmap(C_mc[end])) 

    mkpath("results/old_faithful/")
    C_df = DataFrame(C_mc, :auto)
    CSV.write("results/old_faithful/C_mc.csv", C_df)
    K_df = DataFrame(K_mc', :auto)
    CSV.write("results/old_faithful/K_mc.csv", K_df)

    # coord = Coord.cartesian(xmin=1, ymin=40, ymax=100)
    # plots = [
    #     begin
    #         println("Drawing the MCMC sample $(i+1)")
    #         println("The unique classes are $(unique(C_mc[end-i]))")

    #         p = scatter(
    #             data[:, 2], data[:, 3], 
    #             legend=:none,    
    #             marker_z=C_mc[end-i],  
    #             markersize=3,
    #             cmap=:viridis, 
    #             title="MCMC Sample $(i+1)",
    #             alpha=0.6)

    #         for k in unique(C_mc[end-i])
    #             covellipse!(
    #                 Mu_mc[end-i][:, k], Sig_mc[end-i][:, :, k], 
    #                 alpha=0.5, label=[k], color=:silver)
    #         end

    #         # tmp_df = copy(data)
    #         # tmp_df.cluster = C_mc[end-i] 
    #         # p = Gadfly.plot(
    #         #     tmp_df, coord, x=:eruptions, y=:waiting, color=:cluster,
    #         #     alpha=[0.95], Geom.point, 
    #         #     Geom.ellipse(distribution=
    #         #         MvNormal(Mu_mc[end-i][:, 1], Sig_mc[end-i][:, :, 1])),
    #         #     layer(style(line_style=[:dot])),
    #         #     Guide.ylabel(nothing), Guide.xlabel(nothing),
    #         #     Theme(key_position=:none),
    #         #     Scale.color_discrete(
    #         #         n->get(cs.viridis, range(0, 1, length=n))
    #         #     )
    #         # )
    #         p 
    #     end 
    #     for i in 0:2  
    # ]
    # p = Plots.plot(
    #     plots..., sharey=true, sharex=true, 
    #     layout=(1, 3), size=(800, 230))
    
    # p = hstack(plots...)
    # draw(PDF("test_wass.pdf", 13inch, 3inch), p)

    p = plot_density_estimate(X, C_mc, Mu_mc, Sig_mc)
    savefig(p, "test_wass.pdf") 
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
    function compute_density(grid_point, Mu_mc, Sig_mc)
        total = 0.0
        for i in eachindex(Mu_mc)
            cnt = countmap(C_mc[i]) 
            π = [cnt[j] for j in 1:length(unique(C_mc[i]))]
            π = π ./ sum(π)
            component_densities = [
                π[k] * pdf(MvNormal(Mu_mc[i][:, k], Sig_mc[i][:, :, k]), grid_point) 
                for k in eachindex(π)]
            total += sum(component_densities)
        end
        return total / length(Mu_mc)
    end

    density = [compute_density(grid_points[i, :], Mu_mc, Sig_mc) 
               for i in 1:size(grid_points, 1)]
    density_matrix = reshape(density, (length(y_grid), length(x_grid)))

    # Plot
    p = scatter(X[1, :], X[2, :],  
        color=:black, alpha=0.75, markersize=2, label="Data")
    contour!(x_grid, y_grid, density_matrix, 
        levels=30, c=:viridis, linewidth=1, alpha=0.7)
    title!("Density Estimate by Wass Repulsion")
    xlabel!("X"); ylabel!("Y")
    p 
end 


main() 
