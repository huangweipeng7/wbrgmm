mutable struct EigBoundedNorInverseWishart
    l_σ2::Float64
    u_σ2::Float64
    κ₀::Float64
    μ₀::Vector{Float64} 
    iw::InverseWishart 
end 


EigBoundedNorInverseWishart(l_σ2, u_σ2, κ₀, μ₀, ν₀, Φ₀) =
    EigBoundedNorInverseWishart(
        l_σ2, u_σ2, κ₀, μ₀, InverseWishart(ν₀, round.(Φ₀, digits=10) |> PDMat)) 


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
    
    μ = MvNormal(niw.μ₀, Σ./niw.κ₀) |> rand   
    return μ, Σ
end 


function reset!(
    niw::EigBoundedNorInverseWishart, 
    κ₀::Float64, μ₀::AbstractArray,
    ν₀::Float64, Φ₀::AbstractArray)

    niw.κ₀ = κ₀
    niw.μ₀ .= μ₀ 
    niw.iw = InverseWishart(ν₀, Φ₀)
end 


@inline function post_sample_gauss!(X, Mu, Sig, C, g_prior::EigBoundedNorInverseWishart)
    K = size(Mu, 2)

    @inbounds for k in 1:K
        Xₖ = X[:, C .== k] 
        n = size(Xₖ, 2)  
        
        μ, Σ = n == 0 ? 
            sample_gauss(g_prior) : 
            post_sample_gauss(Xₖ, g_prior)
        
        Mu[:, k] .= μ[:] 
        Sig[:, :, k] .= Σ[:, :]
    end 
end 


@inline function post_sample_gauss(X, niw::EigBoundedNorInverseWishart)
    # An inner function for computing the covariance
    cov2(X) = cov(X; dims=2, corrected=false)

    dim, n = size(X)
  
    x̄ = mean(X; dims=2)
    κₙ = niw.κ₀ + n  
    μₙ = vec(niw.κ₀ * niw.μ₀ + n * x̄) / κₙ  
    νₙ = niw.iw.df + n 
    Φₙ = Matrix(niw.iw.Ψ) + cov2(X) * n + (niw.κ₀ * n / κₙ) * 
        (x̄ - niw.μ₀) * transpose(x̄ - niw.μ₀)    
    Φₙ .= round.(Φₙ, digits=10)

    niw_p = EigBoundedNorInverseWishart(niw.l_σ2, niw.u_σ2, κₙ, μₙ, νₙ, Φₙ)
    μ, Σ = sample_gauss(niw_p)
    # count = 0
    # # Numerical instability will fail the hermitain check
    # iw = InverseWishart(νₙ, round.(Φₙ, digits=10))  
    # Σ = rand(iw)      
    # eig_v_Σ = eigvals(Σ)

    # while first(eig_v_Σ) < niw.l_σ2 || last(eig_v_Σ) > niw.u_σ2  
    #     Σ[:, :] .= rand(iw)      
    #     eig_v_Σ = eigvals(Σ)

    #     count += 1

    #     count <= 1000 || 
    #         throw("Posterior sampling of the kernels takes too long.")
    # end 
    # μ = MvNormal(μₙ, Φₙ/κₙ) |> rand 

    # ν = νₙ - dim + 1
    # μ = MvTDist(ν, μₙ, Φₙ/(κₙ*ν)) |> rand 

    # println(iw)
    # println(μ, " ", Σ)
    # println(x̄, " ", cov2(X), "\n")
    return μ, Σ
end 