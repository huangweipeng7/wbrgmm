abstract type KernelPrior end 

struct WGRMPrior <: KernelPrior
    dim::Int
    smvn::ScaleMvNormal
    biw::EigBoundedIW
end 


struct ScaleMvNormal
    τ::Float64
    mvn::MvNormal
end 


struct EigBoundedIW
    l_σ2::Float64
    u_σ2::Float64
    iw::InverseWishart 
end 


EigBoundedIW(l_σ2, u_σ2, ν₀, Φ₀) = EigBoundedIW(
    l_σ2, u_σ2, InverseWishart(ν₀, round.(Φ₀, digits=10) |> Matrix))


KernelPrior(τ, μ₀, Σ₀, l_σ2, u_σ2, ν₀, Φ₀) = KernelPrior(
    size(μ₀, 1), 
    ScaleMvNormal(τ, MvNormal(μ₀, Σ₀)), 
    EigBoundedIW(l_σ2, u_σ2, ν₀, round.(Φ₀, digits=10) |> Matrix)) 


clogpdf(prior::KernelPrior, μ, Σ) =
    logpdf(prior.smvn.mvn, μ) + logpdf(prior.biw.iw, Σ)


@inline function rand(biw::EigBoundedIW; max_cnt=2000, approx=true)  
    dim = size(biw.iw.Ψ, 1) 
    
    Σ = zeros(dim, dim)
    if !approx
        eig_v_Σ = nothing 

        l_σ2, u_σ2 = biw.l_σ2, biw.u_σ2
        @inbounds for c = 1:max_cnt 
            Σ .= rand(biw.iw) 
            Σ .= round.(Σ, digits=10)    
            eig_v_Σ = eigvals(Σ)
            
            (first(eig_v_Σ) > l_σ2 && last(eig_v_Σ) < u_σ2) && break  
        end  
        throw("Sampling from the prior takes too long. 
            Check if the bounds are set properly")
    else 
        Σ .= rand(biw.iw) 
    end 
    return Σ
end 


function rand(prior::WGRMPrior; max_cnt=2000)  
    Σ = rand(prior.biw; max_cnt=max_cnt)
    μ = rand(prior.smvn.mvn)   
    return μ, Σ
end 


@inline function post_sample_gauss!(
    X, Mu, Sig, C, k_prior::WGRMPrior)

    K = size(Mu, 2)
    @inbounds for k in 1:K
        Xₖ = X[:, C .== k] 
        n = size(Xₖ, 2)  
        
        μ, Σ = n == 0 ? 
            rand(k_prior) : 
            post_sample_gauss(Xₖ, Mu[:, :, k], Sig[:, :, k], k_prior)
         
        Mu[:, k] .= μ[:] 
        Sig[:, :, k] .= Σ[:, :]
    end 
end 


@inline function post_sample_gauss(X, μ, Σ, k_prior::WGRMPrior)
    """ A function for the posterior sampling of Gaussian kernels.
        μ is currently not in use.
    """
    # An inner function for computing the covariance
    cov_(X) = cov(X; dims=2, corrected=false) * n  

    dim, n = size(X) 
  
    x̄ = mean(X; dims=2)

    νₙ = k_prior.biw.iw.df + n 
    Ψₙ = Matrix(k_prior.biw.iw.Ψ) + cov_(X) 
    Ψₙ = round.(Ψₙ, digits=8)
    biw_p = EigBoundedIW(k_prior.biw.l_σ2, k_prior.biw.u_σ2, νₙ, Ψₙ)
    Σ = rand(biw_p)

    Σ₀ = inv(inv(k_prior.smvn.mvn.Σ) + n * inv(Σ))
    Σ₀ = round.(Σ₀, digits=8)
    μ₀ = Σ₀ * (n * inv(Σ) * x̄) |> vec 
    normal = try 
        MvNormal(μ₀, Σ₀) 
    catch LoadError
        Σ₀ = round.(Σ₀, digits=6); 
        MvNormal(μ₀, Σ₀)  
    end 
    μ = rand(normal)

    return μ, Σ
end 


##################################################

struct BGRMPrior <: KernelPrior
    dim::Int
    a::Real
    b::Vector{Real}
    l_σ2::Real
    u_σ2::Real
    τ::Real
end 


function rand(prior::BGRMPrior; max_cnt=2000)  
    Σ = randn(prior.dim) * prior.τ
    μ = rand_inv_gamma(prior)   
    return μ, Σ
end 


function post_sample_gauss_kernels_mc(X, ℓ, t_max, Mu, Sig, C, config::Dict; n_mc=20) 
    g₀ = config["g₀"] 
    a₀ = config["a₀"] 
    b₀ = config["b₀"]
    τ = config["τ"]

    dim = size(Mu, 1)  
    K = ℓ + t_max - 1
    Mu_mc = zeros(Float64, dim, K, n_mc)
    Sig_mc = zeros(Float64, dim, dim, K, n_mc)
    
    normal = MvNormal(zeros(dim), τ^2) 
    @inbounds for k in 1:K
        X_tmp = X[:, C.==k] 
        n = size(X_tmp, 2) 

        if n == 0 
            @inbounds Mu_mc[:, k, :] .= rand(normal, n_mc)
            for mc = 1:n_mc
                @inbounds Sig_mc[:, :, k, mc] .= rand_inv_gamma(a₀, b₀, config)
            end
        else
            x_sum = sum(X_tmp; dims=2)  
            Σ₀ = inv(τ^2*I + n * inv(Sig[:, :, k]))
            μ₀ = Σ₀ * (inv(Sig[:, :, k]) * x_sum) |> vec 
            Mu_mc[:, k, :] .= rand(MvNormal(μ₀, Σ₀), n_mc)

            aₖ = a₀ + n / 2 
            bₖ = b₀ .+ sum((X_tmp .- Mu[:, k]).^2; dims=2) / 2 |> vec
            Sig_mc[:, :, k, :] .= rand_inv_gamma(aₖ, bₖ, config; n=n_mc) 
        end  
    end 
    return Mu_mc, Sig_mc
end 


function rand_inv_gamma(prior::BGRMPrior; n=1) 
    dim = prior.dim
    l_σ2 = prior.l_σ2
    u_σ2 = prior.u_σ2
    a = prior.a 
    b = prior.b 

    Λ = n == 1 ? Diagonal(zeros(Float64, dim)) : zeros(Float64, dim, dim, n) 
    @inbounds for p in 1:dim
        if n == 1
            # Using gamma to sample inverse gamma R.V. is always more robust in Julia
            σ2 = truncated(Gamma(a, 1/bb[p]), 1/u_σ2, 1/l_σ2) |> rand 
            @assert σ2 > 0
            Λ[p, p] = 1 / σ2
        else 
            # Using gamma to sample inverse gamma R.V. is always more robust in Julia
            σ2 = rand(truncated(Gamma(a, 1/bb[p]), 1/u_σ2, 1/l_σ2), n) 
            @assert all(σ2 .> 0)
            Λ[p, p, :] .= 1 ./ σ2
        end  
    end  
    return Λ
end 
