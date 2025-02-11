// Numerical Computation of ZK as in algorithm 1
function numerical_ZK(K_max, dim, g₀)
    n_mc = 2000
    μ_mc = randn(dim, K_max, n_mc)
    Σ_mc = randn(dim, dim, K_max, n_mc)
    g = zeros(K_max, n_mc)
    g_tmp = ones(K, K)
    @inbounds for n = 1:n_mc, K = 2:K_max
        fill!(g_tmp, 1)
        @inbounds for i = 1:(K - 1), j = (i + 1):K
            d = wass_gauss(
                μ_mc[:, i, n], Σ_mc[:, :, i, n], 
                μ_mc[:, j, n], Σ_mc[:, :, j, n])
            g_tmp[i, j] = d / (g₀ + d) 
        end
        g[K, m] = minimum(g_tmp)
    end
    
    # Z[1] = 0
    # for K = 2:K_max
    #     Z[K] = log(mean(g[K, :]))
    # end
    Z = log.(mean(g; dims=2)) |> vec
    Z[1] = 1
    return Z
end