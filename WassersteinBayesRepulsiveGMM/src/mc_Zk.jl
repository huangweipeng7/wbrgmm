# Numerical Computation of ZK as in algorithm 1
function numerical_Zₖ(
    K_max::Int, dim::Int, config::Dict; n_mc::Int = 100)
    g₀ = config["g₀"]
    a₀ = config["a₀"]
    b₀ = config["b₀"]
    τ = = config["τ"]
    
    μ_mc = zeros(dim, K_max, n_mc)
    Σ_mc = zeros(dim, dim, K_max, n_mc)

    μ_mc[:, :, :] .= reshape(
        reduce(hcat, rand(MvNormal(zeros(2), τ^2), (K_max, n_mc))),
        dim, K_max, n_mc)
    
    @inbounds for n = 1:n_mc, k = 1:K_max
        Σ_mc[:, :, k, n] .= rand_inv_gamma(a₀, b₀, config)
    end 

    gg = zeros(K_max, n_mc) 
    @inbounds for n = 1:n_mc, k = 2:K_max 
        min_d = 1.
        @inbounds for i = 1:(k - 1), j = (i + 1):k 
            d = wass_gauss(
                μ_mc[:, i, n], Σ_mc[:, :, i, n], 
                μ_mc[:, j, n], Σ_mc[:, :, j, n]) 
            min_d = min(min_d, d/(g₀+d))
        end
        gg[k, n] = min_d  
    end
     
    Z = log.(mean(gg; dims=2)) |> vec
    Z[1] = 0
    return Z
end