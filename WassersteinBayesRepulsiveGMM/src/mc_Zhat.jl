function numerical_Zhat(μ_mc, Σ_mc, g₀, ℓ, t_max)::Vector{Float64}
    n_mc = size(μ_mc, 3)
    # g_tmp = Matrix{Float64}(undef, K, K)
    gg = zeros(Float64, t_max, n_mc)

    @inbounds for n = 1:n_mc, k = ℓ:(ℓ+t_max-1)
        # fill!(g_tmp, 1) 
        min_d = 1.
        @inbounds for i = 1:(k-1), j = (i+1):k
            d = wass_gauss(
                μ_mc[:, i, n], Σ_mc[:, :, i, n], 
                μ_mc[:, j, n], Σ_mc[:, :, j, n])
            # g_tmp[i, j] = d / (g₀ + d)
            min_d = min(min_d, d/(g₀+d)) 
        end
        gg[k-ℓ+1, n] = min_d # minimum(g_tmp)
    end 

    Ẑ = log.(mean(gg; dims=2)) |> vec
    return Ẑ
end 