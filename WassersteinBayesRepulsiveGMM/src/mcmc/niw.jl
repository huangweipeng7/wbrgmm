mutable struct EigBoundedNorInverseWishart
    l_σ2::Float64
    u_σ2::Float64
    τ::Float64
    μ₀::Vector{Float64} 
    Σ₀::Union{Matrix, Diagonal}
    iw::InverseWishart 
end 


EigBoundedNorInverseWishart(l_σ2, u_σ2, τ, μ₀, Σ₀, ν₀, Φ₀) =
    EigBoundedNorInverseWishart(
        l_σ2, u_σ2, τ, μ₀, Σ₀, InverseWishart(ν₀, round.(Φ₀, digits=10) |> PDMat)) 


# logpdf(niw::EigBoundedNorInverseWishart, μ, Σ) = 
#     logpdf(MvNormal(niw.μ₀, Σ/niw.κ₀), μ) + logpdf(niw.iw, Σ)


# @inline function sample_gauss(niw::EigBoundedNorInverseWishart)  
#     Σ = rand(niw.iw)
#     eig_v_Σ = eigvals(Σ) 

#     l_σ2, u_σ2 = niw.l_σ2, niw.u_σ2
#     count = 0
#     while first(eig_v_Σ) < l_σ2 || last(eig_v_Σ) > u_σ2 
#         Σ .= rand(niw.iw)      
#         eig_v_Σ = eigvals(Σ)

#         count += 1
#         count <= 2000 ||
#             throw("Sampling from the prior takes too long. 
#                    Check if the bounds are set properly")
#     end 
    
#     μ = MvNormal(niw.μ₀, Σ./niw.κ₀) |> rand   
#     return μ, Σ
# end 


@inline function sample_gauss(niw::EigBoundedNorInverseWishart)  
    dim = length(niw.μ₀)
    max_cnt = 2000
    
    Σ = zeros(dim, dim)
    eig_v_Σ = nothing 

    l_σ2, u_σ2 = niw.l_σ2, niw.u_σ2
    @inbounds for c = 1:max_cnt 
        Σ .= rand(niw.iw)      
        eig_v_Σ = eigvals(Σ)
         
        (first(eig_v_Σ) < l_σ2 || last(eig_v_Σ) > u_σ2) || break  
        
        c < max_cnt ||
            throw("Sampling from the prior takes too long. 
                   Check if the bounds are set properly")
    end 
    
    μ = MvNormal(niw.μ₀, niw.Σ₀) |> rand   
    return μ, Σ
end 


@inline function post_sample_gauss!(X, Mu, Sig, C, g_prior::EigBoundedNorInverseWishart)
    K = size(Mu, 2)

    @inbounds for k in 1:K
        Xₖ = X[:, C .== k] 
        n = size(Xₖ, 2)  
        
        μ, Σ = n == 0 ? 
            sample_gauss(g_prior) : 
            post_sample_gauss(Xₖ, Mu[:, :, k], Sig[:, :, k], g_prior)
        
        Mu[:, k] .= μ[:] 
        Sig[:, :, k] .= Σ[:, :]
    end 
end 


@inline function post_sample_gauss(X, μ, Σ, niw::EigBoundedNorInverseWishart)
    # An inner function for computing the covariance
    cov2(X) = cov(X; dims=2, corrected=false)

    dim, n = size(X)
    τ = niw.τ
  
    x̄ = mean(X; dims=2)

    Σ₀ = inv(τ^2 * I + n * inv(Σ))
    Σ₀ = round.(Σ₀, digits=10)
    μ₀ = Σ₀ * (n * inv(Σ) * x̄) |> vec 

    νₙ = niw.iw.df + n 
    Φₙ = Matrix(niw.iw.Ψ) + cov2(X) * n 
    Φₙ .= round.(Φₙ, digits=10)

    niw_p = EigBoundedNorInverseWishart(niw.l_σ2, niw.u_σ2, τ, μ₀, Σ₀, νₙ, Φₙ)
    μ, Σ = sample_gauss(niw_p) 
    return μ, Σ
end 