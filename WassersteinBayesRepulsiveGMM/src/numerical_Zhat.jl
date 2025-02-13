function numerical_Zhat(μ_mc, Σ_mc, g₀, ℓ, t_max)::Vector{Float64}
    n_mc = size(μ_mc, 3) 
    gg = zeros(Float64, t_max, n_mc)

    @inbounds for n = 1:n_mc, k = ℓ:(ℓ+t_max-1)  
        min_d = 1.
        @inbounds for i = 1:(k-1), j = (i+1):k
            d = wass_gauss(
                μ_mc[:, i, n], Σ_mc[:, :, i, n], 
                μ_mc[:, j, n], Σ_mc[:, :, j, n]) 
            min_d = min(min_d, d/(g₀+d)) 
        end
        gg[k-ℓ+1, n] = min_d  
    end 
    Ẑ = log.(mean(gg; dims=2)) |> vec
    return Ẑ
end 