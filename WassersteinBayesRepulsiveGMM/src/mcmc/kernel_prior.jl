mutable struct ScaleMvNormal
    τ::Float64
    mvn::MvNormal
end 


mutable struct EigBoundedIW
    l_σ2::Float64
    u_σ2::Float64
    iw::InverseWishart 
end 


mutable struct KernelPrior
    dim::Int
    smvn::ScaleMvNormal
    biw::EigBoundedIW
end 


EigBoundedIW(l_σ2, u_σ2, ν₀, Φ₀) = EigBoundedIW(
    l_σ2, u_σ2, InverseWishart(ν₀, round.(Φ₀, digits=10) |> Matrix))


KernelPrior(τ, μ₀, Σ₀, l_σ2, u_σ2, ν₀, Φ₀) = KernelPrior(
    size(μ₀, 1), 
    ScaleMvNormal(τ, MvNormal(μ₀, Σ₀)), 
    EigBoundedIW(l_σ2, u_σ2, ν₀, round.(Φ₀, digits=10) |> Matrix)) 


@inline function rand(biw::EigBoundedIW; max_cnt=2000)  
    dim = size(biw.iw.Ψ, 1) 
    
    Σ = zeros(dim, dim)
    eig_v_Σ = nothing 

    l_σ2, u_σ2 = biw.l_σ2, biw.u_σ2
    @inbounds for c = 1:max_cnt 
        Σ .= rand(biw.iw) 
        Σ .= round.(Σ, digits=10)    
        eig_v_Σ = eigvals(Σ)
        
        (first(eig_v_Σ) > l_σ2 && last(eig_v_Σ) < u_σ2) && break  
        
        c < max_cnt ||
            throw("Sampling from the prior takes too long. 
                   Check if the bounds are set properly")
    end  
    return Σ
end 

function rand(prior::KernelPrior; max_cnt=2000)  
    Σ = rand(prior.biw; max_cnt=max_cnt)
    μ = rand(prior.smvn.mvn)   
    return μ, Σ
end 


@inline function post_sample_gauss!(
    X, Mu, Sig, C, k_prior::KernelPrior)

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


@inline function post_sample_gauss(X, μ, Σ, k_prior::KernelPrior)
    """ A function for the posterior sampling of Gaussian kernels.
        μ is currently not in use.
    """
    # An inner function for computing the covariance
    cov_(X) = cov(X; dims=2, corrected=false) * n  

    dim, n = size(X) 
  
    x̄ = mean(X; dims=2)

    νₙ = k_prior.biw.iw.df + n 
    Ψₙ = Matrix(k_prior.biw.iw.Ψ) + cov_(X) 
    Ψₙ = round.(Ψₙ, digits=10)
    biw_p = EigBoundedIW(k_prior.biw.l_σ2, k_prior.biw.u_σ2, νₙ, Ψₙ)
    Σ = rand(biw_p)

    Σ₀ = inv(inv(k_prior.smvn.mvn.Σ) + n * inv(Σ))
    Σ₀ = round.(Σ₀, digits=10)
    μ₀ = Σ₀ * (n * inv(Σ) * x̄) |> vec 
    μ = MvNormal(μ₀, Σ₀) |> rand 

    return μ, Σ
end 