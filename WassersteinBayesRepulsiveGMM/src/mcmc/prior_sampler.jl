function rand_inv_gamma(a, b, config::Dict)::Diagonal 
    dim = config["dim"]
    return rand_inv_gamma(a, fill(b, dim), config)
end 

 
function rand_inv_gamma(a, bb::Vector, config::Dict)::Diagonal 
    dim = config["dim"]
    l_σ2 = config["l_σ2"]
    u_σ2 = config["u_σ2"]

    Λ = Diagonal(zeros(dim))  
    @inbounds for p in 1:dim 
        # Using gamma to sample inverse gamma R.V. is always more robust in Julia
        σ2 = truncated(Gamma(a, 1/bb[p]), 1/u_σ2, 1/l_σ2) |> rand 
        @assert σ2 > 0
        Λ[p, p] = 1 / σ2
    end  
    return Λ
end 


function rand_inv_gamma_mc(a, b, n_mc, config::Dict)::Diagonal 
    dim = config["dim"]
    return rand_inv_gamma_mc(a, fill(b, dim), n_mc, config)
end 

 
function rand_inv_gamma_mc(a, bb::Vector, n_mc, config::Dict) 
    dim = config["dim"]
    l_σ2 = config["l_σ2"]
    u_σ2 = config["u_σ2"]

    Λ = zeros(dim, dim, n_mc, ) 
    @inbounds for p in 1:dim
        # Using gamma to sample inverse gamma R.V. is always more robust in Julia
        σ2 = rand(truncated(Gamma(a, 1/bb[p]), 1/u_σ2, 1/l_σ2), n_mc) 
        @assert all(σ2 .> 0)
        Λ[p, p, :] .= 1 ./ σ2
    end  
    return Λ
end 


function sample_repulsive_gauss!(X, Mu, Sig, ℓ, config::Dict)
    g₀ = config["g₀"] 
    a₀ = config["a₀"] 
    b₀ = config["b₀"]
    τ = config["τ"] 

    dim, K = size(Mu)
    normal = MvNormal(zeros(dim), τ^2)

    min_d = 0.     # min wasserstein distance 
    while rand() > min_d   
        @inbounds for k in ℓ+1:K 
            Mu[:, k] .= rand(normal)
            Sig[:, :, k] .= rand_inv_gamma(a₀, b₀, config)
        end 
        min_d = min_wass_distance(Mu, Sig, g₀)
    end 
end 
