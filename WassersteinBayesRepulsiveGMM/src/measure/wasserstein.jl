""" Sum of squares 
"""
@inline sum_sq(x) = sum(@turbo x.^2) 


@inline wass_gauss(μ₁, Σ₁, μ₂, Σ₂) =   
    sqrt(sum_sq(μ₁ .- μ₂) + sum_sq(sqrt(Σ₁) .- sqrt(Σ₂)))  


@inline function min_wass_distance(Mu, Sig, g₀)
    return Threads.nthreads() == 1 ?
        single_min_wass_distance(Mu, Sig, g₀) :
        threaded_min_wass_distance(Mu, Sig, g₀)
end 


@inline function single_min_wass_distance(Mu, Sig, g₀)
    size(Mu, 2) == size(Sig, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))

    K = size(Mu, 2)  
    min_d = 1.
    for i in 1:K-1, j in i+1:K
        @inbounds d = wass_gauss(
            Mu[:, i], Sig[:, :, i], Mu[:, j], Sig[:, :, j])  
        min_d = min(min_d, d/(d+g₀))
    end  
    return min_d
end 


@inline function threaded_min_wass_distance(Mu, Sig, g₀)
    size(Mu, 2) == size(Sig, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))

    K = size(Mu, 2)  
    min_d = 1. 
    indices = filter(c -> c[1] < c[2], CartesianIndices((1:K, 1:K)))
    Threads.@threads for (i, j) in Tuple.(indices)
        @inbounds d = wass_gauss(
            Mu[:, i], Sig[:, :, i], Mu[:, j], Sig[:, :, j])  
        min_d = min(min_d, d/(d+g₀))
    end  
    return min_d
end 