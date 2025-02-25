@inline rand_inv_gamma(a, b::Real, config)::Diagonal = 
    rand_inv_gamma(a, fill(b, config["dim"]), config)

 
@inline function rand_inv_gamma(a, b::Vector, config)::Diagonal 
    dim = config["dim"]
    l_σ2 = config["l_σ2"]
    u_σ2 = config["u_σ2"]

    Λ = Diagonal(zeros(dim))  
    @inbounds for p in 1:dim 
        # Using gamma to sample inverse gamma R.V. is always more robust in Julia
        σ2 = truncated(Gamma(a, 1/b[p]), 1/u_σ2, 1/l_σ2) |> rand 
        @assert σ2 > 0
        Λ[p, p] = 1 / σ2
    end  
    return Λ
end 


@inline rand_inv_gamma_mc(a, b::Real, n_mc, config)::Array = 
    rand_inv_gamma_mc(a, fill(b, config["dim"]), n_mc, config)

 
@inline function rand_inv_gamma_mc(a, b::Vector, n_mc, config) 
    dim = config["dim"]
    l_σ2 = config["l_σ2"]
    u_σ2 = config["u_σ2"]

    Λ = zeros(dim, dim, n_mc) 
    @inbounds for p in 1:dim
        # Using gamma to sample inverse gamma R.V. is always more robust in Julia
        σ2 = rand(truncated(Gamma(a, 1/b[p]), 1/u_σ2, 1/l_σ2), n_mc) 
        @assert all(σ2 .> 0)
        Λ[p, p, :] .= 1 ./ σ2
    end  
    return Λ
end 


@inline function sample_gauss(config::Dict)   
    μ = randn(config["dim"]) * config["τ"] 
    Σ = rand_inv_gamma(config["a₀"], config["b₀"], config)
    return μ, Σ
end 


@inline function sample_repulsive_gauss!(X, Mu, Sig, ℓ, config)
    g₀ = config["g₀"] 
    K = size(Mu, 2) 

    min_d = 0.     # min wasserstein distance 
    while rand() > min_d   
        @inbounds for k in ℓ+1:K 
            μ, Σ = sample_gauss(config)
            Mu[:, k] .= μ
            Sig[:, :, k] .= Σ
        end 
        min_d = min_wass_distance(Mu, Sig, g₀)
    end 
end 


#
# Code for sampling using a Normal-Inverse-Wishart prior.
#

@inline sample_gauss(g_prior::NorInvWishart) = rand(g_prior) 


@inline function sample_repulsive_gauss!(X, Mu, Sig, ℓ, config, g_prior)
    g₀ = config["g₀"]  
    K = size(Mu, 2)

    min_d = 0.     # min wasserstein distance 
    while rand() > min_d   
        @inbounds for k in ℓ+1:K 
            μ, Σ = sample_gauss(g_prior)
            Mu[:, k] .= μ
            Sig[:, :, k] .= Σ 
        end 
        min_d = min_wass_distance(Mu, Sig, g₀)
    end 
end 
  