function blocked_gibbs(
    X::Matrix{Float64}, prior::NormalInverseWishart;
    g₀::Float64 = 100., K::Int = 5, α::Float64 = 1., 
    burnin::Int = 2000, runs::Int = 3000, thinning::Int = 1)

    C_mc = Vector()
    Mu_mc = Vector()
    Sigma_mc = Vector()
    llhd_mc = Vector{Float64}()

    dim, n = size(X)

    C = zeros(Int, n) 
    Mu = zeros(Float64, K, dim)
    Sigma = zeros(Float64, K, dim, dim)
    initialize!(Mu, Sigma, C, prior, g₀) 

    iter = ProgressBar(1:(burnin+runs))
    @inbounds for run in iter
        llhd = post_sample_C!(X, α, Mu, Sigma, C, prior)
        set_description(iter, "loglikelihood: $(round(llhd, sigdigits=3))")

        post_sample_repulsive_gauss!(X, Mu, Sigma, C, g₀, prior)
        # post_sample_K()
        
        if run % thinning == 0 
            push!(C_mc, deepcopy(C))
            push!(Mu_mc, deepcopy(Mu))  
            push!(Sigma_mc, deepcopy(Sigma))
            push!(llhd_mc, llhd)
        end
    end 
    return C_mc, Mu_mc, Sigma_mc, llhd_mc
end 


gumbel_max_sample(lp)::Int = argmax(lp + rand(GUMBEL, size(lp)))


function post_sample_C!(X, α, Mu, Sigma, C, prior)::Float64
    size(Mu, 1) == size(Sigma, 1) ||
        throw(DimensionMismatch("Inconsistent array dimensions.")) 

    n = size(X, 2)
    K = size(Mu, 1)    
    lp = Vector{Float64}(undef, n)
    n_z = K - (length ∘ unique)(C)
 
    C_prime = Vector{Int}(undef, n)
    llhd = 0. 
    @inbounds for i in 1:n 
        fill!(lp, -Inf)
        x = X[:, i]
        @inbounds for k in 1:K 
            lp[k] = dlogpdf(MvNormal(Mu[k, :], Sigma[k, :, :]), x) 
            n_k = sum(C .== K) - (C[i] == K)   
            lp[k] += log(n_k == 0 ? α / n_z : n_k)  
        end  
        C_prime[i] = gumbel_max_sample(lp)
        llhd += lp[C_prime[i]]
    end 
    C[:] .= C_prime[:]
    return llhd
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
 

function initialize!(Mu, Sigma, C, prior, g₀) 
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

    C[:] .= sample(1:K, length(C), replace=true)
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