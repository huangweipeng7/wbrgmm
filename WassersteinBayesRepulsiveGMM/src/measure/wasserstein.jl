@fastmath function wass_gauss(μ₁, Σ₁, μ₂, Σ₂)::Float64
    """ Wassertein distance between Normal(μ₁, Σ₁) and Normal(μ₂, Σ₂)
    """   
    d = sum((μ₁ .- μ₂) .^ 2) + sum((sqrt(Σ₁) - sqrt(Σ₂)) .^ 2)    
    return sqrt(d) 
end


@fastmath function wass_gauss2(μ₁, Σ₁, μ₂, Σ₂)::Float64
    """ Wassertein distance between Normal(μ₁, Σ₁) and Normal(μ₂, Σ₂)
    """  
    Σ₁_sqrt = sqrt(Σ₁) 
    d = sum((μ₁ .- μ₂) .^ 2) 
    d += tr(Σ₁) + tr(Σ₂) - 2 * tr(sqrt(Σ₁_sqrt * Σ₂ * Σ₁_sqrt))   
    return sqrt(d) 
end
