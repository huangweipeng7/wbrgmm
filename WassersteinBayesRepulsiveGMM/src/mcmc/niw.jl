mutable struct EigBoundedNorInverseWishart
    l_σ2::Float64
    u_σ2::Float64
    κ₀::Float64
    μ₀::Vector{Float64} 
    iw::InverseWishart 
end 


EigBoundedNorInverseWishart(l_σ2, u_σ2, κ₀, μ₀, ν₀, Φ₀) =
    EigBoundedNorInverseWishart(
        l_σ2, u_σ2, κ₀, μ₀, InverseWishart(ν₀, PDMat(Φ₀))) 


# logpdf(niw::EigBoundedNorInverseWishart, μ, Σ) = 
#     logpdf(MvNormal(niw.μ₀, Σ/niw.κ₀), μ) + logpdf(niw.iw, Σ)


@inline function rand(niw::EigBoundedNorInverseWishart)  
    Σ = rand(niw.iw)      
    eig_v_Σ = eigvals(Σ)

    count = 0
    while first(eig_v_Σ) < niw.l_σ2 || last(eig_v_Σ) > niw.u_σ2 
        println(eig_v_Σ, " ", niw.l_σ2, " ", niw.u_σ2, "  prior  ", Σ)
        println(niw.iw)
        Σ[:, :] .= rand(niw.iw)      
        eig_v_Σ = eigvals(Σ)

        count += 1
        count <= 2000 ||
            throw("Sampling from the prior takes too long. Check if the bounds are set properly")
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
    K = size(Mu, 1)

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
    νₙ = niw.iw.df + n 
    μₙ = vec(niw.κ₀ * niw.μ₀ + n * x̄) / κₙ 

    println(X, "  ", cov2(X) * n, "  ", (niw.κ₀ * n / κₙ), 
        "   ", 
        (x̄ - niw.μ₀) * transpose(x̄ - niw.μ₀))
    Φₙ = Matrix(niw.iw.Ψ) + cov2(X) * n + (niw.κ₀ * n / κₙ) * 
        (x̄ - niw.μ₀) * transpose(x̄ - niw.μ₀)   
    println(inv(Φₙ))

    count = 0
    iw = InverseWishart(νₙ, inv(Φₙ))
    Σ = rand(iw)      
    eig_v_Σ = eigvals(Σ)

    while first(eig_v_Σ) < niw.l_σ2 || last(eig_v_Σ) > niw.u_σ2  
        # println(eig_v_Σ, " ", niw.l_σ2, " ", niw.u_σ2, "  post ", Σ)
        # println(niw.iw)

        Σ[:, :] .= rand(iw)      
        eig_v_Σ = eigvals(Σ)

        count += 1

        count <= 2000 || 
            throw("Posterior sampling of the kernels takes too long.")
    end 

    ν = νₙ - dim + 1
    μ = MvTDist(ν, μₙ, Φₙ/(κₙ*ν)) |> rand 
    return μ, Σ
end 