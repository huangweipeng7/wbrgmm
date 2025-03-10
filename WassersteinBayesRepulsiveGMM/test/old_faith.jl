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
    g₀ = 20
    β = 50
    τ = 0.1 
    l_σ2 = 1e-6
    u_σ2 = 1e6
    K = 5 
    ν₀ = 3 

    prior = EigBoundedNorInverseWishart(
        l_σ2, u_σ2, τ, zeros(Real, dim), τ^2*I(dim), ν₀, I(dim))

    C_mc, Mu_mc, Sigma_mc, K_mc, llhd_mc = wrbgmm_blocked_gibbs(
        X; g₀=g₀, K=K, β=β, τ=τ, prior=prior, 
        l_σ2=l_σ2, u_σ2=u_σ2, burnin=2000, runs=5000, thinning=5) 

    println(
        "Cluster distribution from the last iteration: \n", 
        countmap(C_mc[end])) 

    mkpath("results/old_faithful/")
    C_df = DataFrame(C_mc, :auto)
    CSV.write("results/old_faithful/C_mc.csv", C_df)
    K_df = DataFrame(K_mc', :auto)
    CSV.write("results/old_faithful/K_mc.csv", K_df)

    # coord = Coord.cartesian(xmin=1, ymin=40, ymax=100)
    plots = [
        begin
            println("Drawing the MCMC sample $(i+1)")
            println("The unique classes are $(unique(C_mc[end-i]))")

            p = scatter(
                data[:, 2], data[:, 3], 
                legend=:none,    
                marker_z=C_mc[end-i],  
                markersize=3,
                cmap=:viridis, 
                title="MCMC Sample $(i+1)",
                alpha=0.6)

            for k in unique(C_mc[end-i])
                covellipse!(
                    Mu_mc[end-i][:, k], Sigma_mc[end-i][:, :, k], 
                    alpha=0.5, label=[k], color=:silver)
            end

            # tmp_df = copy(data)
            # tmp_df.cluster = C_mc[end-i] 
            # p = Gadfly.plot(
            #     tmp_df, coord, x=:eruptions, y=:waiting, color=:cluster,
            #     alpha=[0.95], Geom.point, 
            #     Geom.ellipse(distribution=
            #         MvNormal(Mu_mc[end-i][:, 1], Sigma_mc[end-i][:, :, 1])),
            #     layer(style(line_style=[:dot])),
            #     Guide.ylabel(nothing), Guide.xlabel(nothing),
            #     Theme(key_position=:none),
            #     Scale.color_discrete(
            #         n->get(cs.viridis, range(0, 1, length=n))
            #     )
            # )
            p 
        end 
        for i in 0:2  
    ]
    p = Plots.plot(
        plots..., sharey=true, sharex=true, 
        layout=(1, 3), size=(800, 230))
    
    # p = hstack(plots...)
    # draw(PDF("test_wass.pdf", 13inch, 3inch), p)
    savefig(p, "test_wass.pdf") 
end 


main() 
