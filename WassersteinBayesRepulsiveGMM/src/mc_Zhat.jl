function numerical_Zhat(μ_mc, Σ_mc, g₀, t, m)::Vector{Float64}
    n_mc = size(μ_mc, 3)
    g_tmp = Matrix{Float64}(undef, K, K)
    gg = zeros(Float64, m, n_mc)

    @inbounds for n = 1:n_mc, K = t:(t+m-1)
        fill!(g_tmp, 1) 
        @inbounds for i = 1:(K-1), j = (i+1):K
            d = wass_gauss(
                μ_mc[:, i, n], Σ_mc[:, :, i, n], 
                μ_mc[:, j, n], Σ_mc[:, :, j, n])
            g_tmp[i, j] = d / (g₀ + d); 
        end
        gg[K-t+1, n] = minimum(g_tmp)
    end 

    Ẑ = log.(mean(gg; dims=2)) |> vec
    return Ẑ
end 