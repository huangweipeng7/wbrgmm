wass_gauss(μ₁, Σ₁, μ₂, Σ₂) =   
    sum((μ₁ .- μ₂) .^ 2) + sum((sqrt(Σ₁) - sqrt(Σ₂)) .^ 2)  
 

function min_wass_distance(Mu, Sig, g₀)
    size(Mu, 2) == size(Sig, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))

    K = size(Mu, 2)  
    min_d = 1.
    @inbounds for i = 1:K-1, j = i+1:K   
        d = wass_gauss(
            Mu[:, i], Sig[:, :, i], Mu[:, j], Sig[:, :, j])  
        min_d = min(min_d, d/(d+g₀))
    end  
    return min_d
end 
