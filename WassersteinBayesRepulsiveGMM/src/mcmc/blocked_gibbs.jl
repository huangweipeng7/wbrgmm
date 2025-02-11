function blocked_gibbs(
    X::Matrix{Float64};
    g₀::Float64 = 100., K::Int = 5, α::Float64 = 1., 
    a₀::Float64 = 1., b₀::Float64 = 1.,
    l_σ2::Float64 = 0.001, u_σ2::Float64 = 10000.,
    burnin::Int = 2000, runs::Int = 3000, thinning::Int = 1)

    dim, n = size(X)

    config = Dict(
        "g₀" => g₀, "α" => α, "a₀" => a₀, "b₀" => b₀,
        "dim" => dim, "l_σ2" => l_σ2, "u_σ2" => u_σ2)

    C_mc = Vector()
    Mu_mc = Vector()
    Sigma_mc = Vector()
    llhd_mc = Vector{Float64}()


    C = zeros(Int, n) 
    Mu = zeros(Float64, dim, K)
    Sigma = zeros(Float64, dim, dim, K)
    initialize!(Mu, Sigma, C, config) 

    iter = ProgressBar(1:(burnin+runs))
    @inbounds for run in iter
        llhd = post_sample_C!(X, Mu, Sigma, C, config)
        set_description(iter, "loglikelihood: $(round(llhd, sigdigits=3))")

        post_sample_repulsive_gauss!(X, Mu, Sigma, C, config)
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


function post_sample_C!(X, Mu, Sigma, C, config)::Float64
    size(Mu, 2) == size(Sigma, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions.")) 

    α = get(config, "α", missing)

    n = size(X, 2)
    K = size(Mu, 1)    
    lp = Vector{Float64}(undef, n)
    n_z = K - (length ∘ unique)(C)
 
    C_prime = Vector{Int}(undef, n)
    llhd = 0.           # Log likelihood
    c_ = -1             # a fake index value for c underscore 
    @inbounds for i in 1:n 
        fill!(lp, -Inf)
        x = X[:, i]
        @inbounds for k in 1:K 
            n_k = sum(C .== K) - (C[i] == K)   
            
            if n_k == 0 || c_ == -1 
                lp[k] = dlogpdf(MvNormal(Mu[:, k], Sigma[:, :, k]), x) 
                lp[k] += (n_k == 0 ? 
                    log(α) + log_V_nt(ℓ+1) - log_V_nt(ℓ) : log(n_k+α)) 
            end 

            if n_k == 0 && c_ != -1
                c_ = k 
            end            
        end  
        C_prime[i] = gumbel_max_sample(lp)
        llhd += lp[C_prime[i]]
    end 
    C[:] .= C_prime[:]
    return llhd
end   
 

function post_sample_gauss_kernels!(X, Mu, Sigma, C, config::Dict) 
    g₀ = get(config, "g₀", missing)
    a₀ = get(config, "a₀", missing) 
    b₀ = get(config, "b₀", missing)

    dim, K = size(Mu)  
    gauss = MvNormal(zeros(dim), Matrix(I, dim, dim)) 
    
    @inbounds for k in 1:K
        X_tmp = X[:, C.==k] 
        n = size(X_tmp, 2) 

        if n == 0 
            Mu[:, k] .= rand(gauss)
            Sigma[:, :, k] .= rand_inv_gamma(a₀, b₀, config)
        else 
            x̄ = mean(X_tmp; dims=2)  
            Σ₀ = pinv(I + n * pinv(Sigma[:, :, k]))
            μ₀ = Σ₀ * (n * pinv(Sigma[:, :, k]) * x̄) |> vec 
            Mu[:, k] .= MvNormal(μ₀, Σ₀) |> rand

            aₖ = a₀ + n / 2 
            bₖ = b₀ .+ sum((X_tmp .- Mu[:, k]).^2; dims=2) / 2 |> vec
            Sigma[:, :, k] .= rand_inv_gamma(aₖ, bₖ, config)
        end  
    end 
end 
 

function rand_inv_gamma(a::Float64, b::Float64, config)::Diagonal
    dim = get(config, "dim", missing)
    return rand_inv_gamma(a, fill(b, dim), config)
end 

 
function rand_inv_gamma(a::Float64, bb::Vector{Float64}, config)::Diagonal
    dim = get(config, "dim", missing)
    l_σ2 = get(config, "l_σ2", missing)
    u_σ2 = get(config, "u_σ2", missing)

    λs = zeros(Float64, dim)
    @inbounds for p in 1:dim
        σ2 = InverseGamma(a, bb[p]) |> rand 
        while σ2 > u_σ2 || σ2 < l_σ2
            σ2 = InverseGamma(a, bb[p]) |> rand 
        end 
        λs[p] = σ2
    end   
    return Diagonal(λs)
end 


function post_sample_repulsive_gauss!(X, Mu, Sigma, C, config)::Int
    min_d = 0.     # min wasserstein distance
    reject_counts = 0 

    g₀ = get(config, "g₀", missing)
    while rand() > min_d 
        post_sample_gauss_kernels!(X, Mu, Sigma, C, config)
        reject_counts += 1 
        min_d = min_wass_distance(Mu, Sigma, g₀)
    end
    return reject_counts 
end 
 

function initialize!(Mu, Sigma, C, config)  
    size(Mu, 2) == size(Sigma, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    
    g₀ = get(config, "g₀", missing)
    a₀ = get(config, "a₀", missing) 
    b₀ = get(config, "b₀", missing)

    dim, K = size(Mu) 
    min_d = 0.
    gauss = MvNormal(zeros(dim), Matrix(I, dim, dim))
    while rand() > min_d   
        @inbounds for k in 1:K 
            Mu[:, k] .= rand(gauss)
            Sigma[:, :, k] .= rand_inv_gamma(a₀, b₀, config)
        end
        min_d = min_wass_distance(Mu, Sigma, g₀) 
    end 
    C[:] .= sample(1:K, length(C), replace=true)    # Random assignments of clustering
end 


function min_wass_distance(Mu, Sigma, g₀)::Float64 
    size(Mu, 2) == size(Sigma, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))

    K = size(Mu, 1)
    min_d = Inf
    @inbounds for i = 1:K, j = (i+1):K  
        d = wass_gauss(
            Mu[:, i], Sigma[:, :, i], Mu[:, j], Sigma[:, :, j])
        d = d / (d + g₀)
        min_d = min(min_d, d)
    end  
    return min_d
end 
