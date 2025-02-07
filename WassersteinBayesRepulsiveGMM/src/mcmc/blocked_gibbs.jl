function blocked_gibbs(
    X::Matrix{Float64}, prior::NormalInverseWishart, g₀::Real, K::Int; 
    burnin::Int = 2000, runs::Int = 3000, thinning::Int = 1)

    C_mc = Vector()
    Mu_mc = Vector()
    Sigma_mc = Vector()

    dim, n = size(X)
 
    C = zeros(Int, n) 
    Mu = zeros(Float64, K, dim)
    Sigma = zeros(Float64, K, dim, dim)
    initialize!(Mu, Sigma, prior, g₀) 

    inds = 1:n 
    @inbounds for run in ProgressBar(1:(burnin+runs))
        @inbounds for i in 1:n
            C[i] = post_sample_C(X[:, i], K, Mu, Sigma, prior) 
        end 

        post_sample_repulsive_gauss!(X, Mu, Sigma, C, g₀, prior)
        # post_sample_K()
        
        if run > burnin && run % thinning == 0 
            append!(C_mc, deepcopy(C))
            append!(Mu_mc, deepcopy(Mu))
            append!(Sigma_mc, deepcopy(Sigma))
        end
    end 
    return C_mc, Mu_mc, Sigma_mc
end 


gumbel_max_sample(lp)::Int = argmax(lp + rand(Gumbel(0, 1), size(lp)))


function post_sample_C(x, K, Mu, Sigma, prior)::Int
    lp = zeros(Float64, K)
    @inbounds for k in 1:K 
        lp[k] = Distributions.logpdf(MvNormal(Mu[k, :], Sigma[k, :, :]), x) +
            logpdf(prior, Mu[k, :], Sigma[k, :, :])
    end 
    return gumbel_max_sample(lp)
end  


function post_sample_gauss_kernels!(X, Mu, Sigma, C, prior)
    niw_tmp = deepcopy(prior)

    K = size(Mu, 1)
    @inbounds for k in 1:K
        X_tmp = X[:, C.==k] 
        n = size(X_tmp, 2) 

        if n == 0 
            μ, Σ = rand(prior)
        else 
            x̄ = mean(X_tmp; dims=2)
            κ₀ = prior.κ₀ + n  
            ν₀ = prior.ν₀ + n 
            μ₀ = (prior.κ₀ * prior.μ₀ + n * x̄) / κ₀ 
            Φ₀ = prior.Φ₀ + cov(X_tmp; dims=2, corrected=false) * n + 
                prior.κ₀ * n / κ₀ * (x̄ - prior.μ₀) * transpose(x̄ - prior.μ₀) 

            reset!(niw_tmp, κ₀, μ₀, ν₀, Φ₀)
            μ, Σ = rand(niw_tmp)  
        end 
        Mu[k, :] .= μ[:] 
        Sigma[k, :, :] .= Σ[:, :]
    end 
end 
 
 
function post_sample_K()
    return 0
end 


function post_sample_repulsive_gauss!(X, Mu, Sigma, C, g₀, prior)::Int
    min_d = 0.     # min wasserstein distance
    reject_counts = 0 
    while rand() > min_d 
        post_sample_gauss_kernels!(X, Mu, Sigma, C, prior)
        reject_counts += 1 
        min_d = min_wass_distance(Mu, Sigma, g₀)
    end
    return reject_counts 
end 
 

function initialize!(Mu, Sigma, prior, g₀) 
    size(Mu, 1) == size(Sigma, 1) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    
    K = size(Mu, 1) 
    min_d = 0.
    while rand() > min_d   
        @inbounds for k in 1:K
            μ, Σ = rand(prior)  
            Mu[k, :] .= μ[:] 
            Sigma[k, :, :] .= Σ[:, :] 
        end
        min_d = min_wass_distance(Mu, Sigma, g₀) 
    end 
end 


function min_wass_distance(Mu, Sigma, g₀)::Float64 
    size(Mu, 1) == size(Sigma, 1) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))

    K = size(Mu, 1)
    min_d = Inf
    @inbounds for i = 1:K, j = (i+1):K  
        d = wass_gauss(
            Mu[i, :], Sigma[i, :, :], Mu[j, :], Sigma[j, :, :])
        d = d / (d + g₀)
        min_d = min(min_d, d)
    end  
    return min_d
end 