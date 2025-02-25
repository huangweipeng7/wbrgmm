mutable struct NorInvWishart
    κ₀::Float64
    μ₀::Vector{Float64} 
    iw::InverseWishart 
end 


NorInvWishart(κ₀, μ₀, ν₀, Φ₀) =
    NorInvWishart(κ₀, μ₀, InverseWishart(ν₀, Φ₀)) 


logpdf(niw::NorInvWishart, μ, Σ) = 
    logpdf(MvNormal(niw.μ₀, Σ/niw.κ₀), μ) + logpdf(niw.iw, Σ)


function rand(niw::NormalInverseWishart)  
    Σ = rand(niw.iw)   
    μ = MvNormal(niw.μ₀, Σ./niw.κ₀) |> rand  
    return μ, Σ
end 


function reset!(
    niw::NorInvWishart, 
    κ₀::Float64, μ₀::AbstractArray,
    ν₀::Float64, Φ₀::AbstractArray)

    niw.κ₀ = κ₀
    niw.μ₀ .= μ₀ 
    niw.iw = InverseWishart(ν₀, Φ₀)
end 


function post_sample_gauss_kernels!(X, Mu, Sig, C, g_prior::NorInvWishart)
    K = size(Mu, 1)

    @inbounds for k in 1:K
        X_tmp = X[:, C.==k] 
        n = size(X_tmp, 2)  
        μ, Σ = n == 0 ? 
            rand(g_prior) : post_sample_gauss_kernel(X_tmp, g_prior)
        Mu[k, :] .= μ[:] 
        Sig[k, :, :] .= Σ[:, :]
    end 
end 


function post_sample_gauss_kernel(X, g_prior::NorInvWishart)
    # An inner function for computing the covariance
    cov2(X) = cov(X; dims=2, corrected=false)

    x̄ = mean(X; dims=2)
    κ₀ = g_prior.κ₀ + n  
    ν₀ = g_prior.iw.ν₀ + n 
    μ₀ = (g_prior.κ₀ * g_prior.μ₀ + n * x̄) / κ₀ 
    Φ₀ = g_prior.iw.Φ₀ + cov2(X) * n + g_prior.κ₀ * n / κ₀ * 
        (x̄ - g_prior.μ₀) * transpose(x̄ - g_prior.μ₀) 

    μ, Σ = NorInvWishart(κ₀, μ₀, ν₀, Φ₀) |> rand
    return μ, Σ
end 