""" Sum of squares 
"""
@inline sum_sq(x) = sum(@turbo x.^2) 


@inline function wass_gauss(μ₁, Σ₁, μ₂, Σ₂) # = (μ₁ .- μ₂) .^ 2 |> sum_sq
    Σ₂_sqrt = sqrt(Σ₂) 
    μ_part = (μ₁ .- μ₂) .^ 2 
    Σ_part = Σ₁ .+ Σ₂ .- 2 * sqrt(Σ₂_sqrt * Σ₁ * Σ₂_sqrt)   
    return sum(μ_part) + tr(Σ_part)  
end


@inline function min_wass_distance(Mu, Sig, g₀)
    return Threads.nthreads() == 1 ?
        single_min_wass_distance(Mu, Sig, g₀) :
        threaded_min_wass_distance(Mu, Sig, g₀)
end 


@inline function single_min_wass_distance(Mu, Sig, g₀)
    size(Mu, 2) == size(Sig, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))

    K = size(Mu, 2)   
    hₖ = 1.
    @inbounds for i in 1:K-1, j in i+1:K
        d = wass_gauss(
            Mu[:, i], Sig[:, :, i], Mu[:, j], Sig[:, :, j])
        hₖ = min(hₖ, d / (d + g₀)) 
    end   
    return hₖ
end 


@inline function threaded_min_wass_distance(Mu, Sig, g₀)
    size(Mu, 2) == size(Sig, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))

    K = size(Mu, 2)  
    hₖ = 1. 
    indices = filter(c -> c[1] < c[2], CartesianIndices((1:K, 1:K)))
    Threads.@threads for (i, j) in Tuple.(indices)
        @inbounds d = wass_gauss(
            Mu[:, i], Sig[:, :, i], Mu[:, j], Sig[:, :, j])  
        hₖ = min(hₖ, d / (d + g₀)) 
    end  
    return hₖ
end 