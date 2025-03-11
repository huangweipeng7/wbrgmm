# Numerical Computation of ZK as in algorithm 1
@inline function numerical_Zₖ(K_max, dim, config, k_prior; n_mc=1000)
    g₀ = config["g₀"]
    a₀ = config["a₀"]
    b₀ = config["b₀"]
    τ = config["τ"]
    
    μ_mc = zeros(dim, K_max, n_mc)
    Σ_mc = zeros(dim, dim, K_max, n_mc)
 
    @inbounds μ_mc[:, :, :] .= randn((dim, K_max, n_mc)) * τ
    
    Threads.@threads for n = 1:n_mc
        for k = 1:K_max
            Σ_mc[:, :, k, n] .= rand(k_prior.biw)
        end 
    end 

    gg = zeros(K_max, n_mc) 
    Threads.@threads for n = 1:n_mc
        for k = 2:K_max 
            min_d = 1.
            @inbounds for i = 1:(k - 1), j = (i + 1):k 
                d = wass_gauss(
                    μ_mc[:, i, n], Σ_mc[:, :, i, n], 
                    μ_mc[:, j, n], Σ_mc[:, :, j, n]) 
                min_d = min(min_d, d/(g₀+d))
            end
            gg[k, n] = min_d  
        end
    end 

    Z = log.(mean(gg; dims=2)) |> vec
    Z[1] = 0
    return Z
end