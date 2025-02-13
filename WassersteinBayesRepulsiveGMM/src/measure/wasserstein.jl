function wass_gauss(μ₁, Σ₁, μ₂, Σ₂)::Float64
    """ Wassertein distance between Normal(μ₁, Σ₁) and Normal(μ₂, Σ₂)
    """ 
    Σ₂_sqrt = sqrt(Σ₂) 
    μ_part = @. (μ₁ - μ₂)^2 
    Σ_part = Σ₁ .+ Σ₂ .- 2 * sqrt(Σ₂_sqrt * Σ₁ * Σ₂_sqrt) 
    return sum(μ_part) + tr(Σ_part) 
end
