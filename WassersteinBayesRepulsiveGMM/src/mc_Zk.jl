# Numerical Computation of ZK as in algorithm 1
function numerical_ZK(K_max, dim, config)
    g₀ = config["g₀"]
    a₀ = config["a₀"]
    b₀ = config["b₀"]

    n_mc = 2000
    μ_mc = randn(dim, K_max, n_mc)
    Σ_mc = zeros(dim, dim, K_max, n_mc)
    @inbounds for n = 1:n_mc, k = 1:K_max
        Σ_mc[:, :, k, n] .= rand_inv_gamma(a₀, b₀, config)
    end 

    g = zeros(K_max, n_mc)
    # g_tmp = ones(K, K)
    @inbounds for n = 1:n_mc, k = 2:K_max
        # fill!(g_tmp, 1)
        min_d = 1.
        @inbounds for i = 1:(k - 1), j = (i + 1):k
            d = wass_gauss(
                μ_mc[:, i, n], Σ_mc[:, :, i, n], 
                μ_mc[:, j, n], Σ_mc[:, :, j, n])
            # g_tmp[i, j] = d / (g₀ + d) 
            min_d = min(min_d, d / (g₀ + d))
        end
        g[k, n] = min_d # minimum(g_tmp)
    end
     
    Z = log.(mean(g; dims=2)) |> vec
    Z[1] = 0
    return Z
end