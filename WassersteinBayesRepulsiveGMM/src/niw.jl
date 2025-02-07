mutable struct NormalInverseWishart
    κ₀::Float64
    μ₀::Vector{Float64}
    ν₀::Float64
    Φ₀::Matrix{Float64}
    iw::InverseWishart 
end 


function NormalInverseWishart(κ₀, μ₀, ν₀, Φ₀)
    return NormalInverseWishart(
        κ₀, μ₀, ν₀, Φ₀, InverseWishart(ν₀, Φ₀)) 
end


logpdf(niw::NormalInverseWishart, μ, Σ)::Float64 = 
    Distributions.logpdf(MvNormal(niw.μ₀, Σ/niw.κ₀), μ) + 
    Distributions.logpdf(niw.iw, Σ) 


function rand(niw::NormalInverseWishart)  
    Σ = rand(niw.iw)   
    μ = MvNormal(niw.μ₀, Σ/niw.κ₀) |> rand  
    return μ, Σ
end 


function reset!(
    niw::NormalInverseWishart, 
    κ₀::Float64, μ₀::AbstractArray,
    ν₀::Float64, Φ₀::AbstractArray)
    niw.κ₀ = κ₀
    niw.μ₀ .= μ₀
    niw.ν₀ = ν₀
    niw.Φ₀ .= Φ₀
    niw.iw = InverseWishart(ν₀, Φ₀)
end 