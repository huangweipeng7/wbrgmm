function blocked_gibbs(
    X::Matrix{Float64};
    g₀::Float64 = 100., α::Float64 = 1., 
    a₀::Float64 = 1., b₀::Float64 = 1., 
    l_σ2::Float64 = 0.001, u_σ2::Float64 = 1000.,
    K::Int = 5, t_max::Int = 2,
    burnin::Int = 2000, runs::Int = 3000, thinning::Int = 1)

    dim, n = size(X)

    config = Dict(
        "g₀" => g₀, "α" => α, "a₀" => a₀, "b₀" => b₀,
        "t_max" => t_max, "dim" => dim, 
        "l_σ2" => l_σ2, "u_σ2" => u_σ2)

    C_mc = Vector()
    Mu_mc = Vector()
    Sig_mc = Vector()
    llhd_mc = Vector()
  
    C = zeros(Int, n) 
    Mu = zeros(Float64, dim, K+1)
    Sig = zeros(Float64, dim, dim, K+1)
    initialize!(X, Mu, Sig, C, config) 

    logV = logV_nt(n, t_max)
    Zₖ = numerical_ZK(4K, dim, config)
    
    iter = ProgressBar(1:(burnin+runs))
    @inbounds for run in iter
        llhd = post_sample_C!(
            X, Mu, Sig, C, logV, Zₖ, config)
        set_description(
            iter, "loglikelihood: $(round(llhd, sigdigits=3))")

        # post_sample_repulsive_gauss!(X, Mu, Sig, C, config) 
        
        if run % thinning == 0 
            push!(C_mc, deepcopy(C))
            push!(Mu_mc, deepcopy(Mu))  
            push!(Sig_mc, deepcopy(Sig))
            push!(llhd_mc, llhd)
        end
    end 
    return C_mc, Mu_mc, Sig_mc, llhd_mc
end 


gumbel_max_sample(lp)::Int = lp + rand(GUMBEL, size(lp)) |> argmax


function post_sample_C!(X, Mu, Sig, C, logV, Zₖ, config)::Float64
    size(Mu, 2) == size(Sig, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions.")) 

    dim = config["dim"]
    t_max = config["t_max"]
    α = config["α"] 
    g₀ = config["g₀"]

    n = size(X, 2)
    K = size(Mu, 2)     

    lp = Vector{Float64}(undef, K+1)
    llhd = 0.           # Log likelihood
    @inbounds for i in 1:n  
        x = X[:, i]

        # Some bugs here
        n_map = countmap(C) 
        ℓ = length(n_map) - (sum(C .== C[i]) == 1) 
        n_map[C[i]] -= 1

        # Sample K 
        logpdf_K = log_p_K(ℓ+t_max-1, ℓ, n)
        K = sample_K(logpdf_K, ℓ, t_max, n) 

        # Always leave a place for a new cluster 
        # Therefore, we use K+1
        Mu_ = zeros(Float64, dim, K+1)
        Sig_ = zeros(Float64, dim, dim, K+1)
        nz_K = keys(n_map) |> collect  

        ℓ = length(nz_K)
        Mu_[:, 1:ℓ] .= Mu[:, nz_K]
        Sig_[:, :, 1:ℓ] .= Sig[:, :, nz_K]

        # Mu_[:, end] .= Mu[:, end]
        # Sig_[:, :, end] .= Sig[:, :, end]
        @inbounds for (i_nz, nz_k) in enumerate(nz_K)
            C[C .== nz_k] .= -1 * i_nz  # ensure there is no index collision
        end 
        C[i] = 0
        C .*= -1
        sample_repulsive_gauss!(X, Mu_, Sig_, C, ℓ, config)
        # println(nz_K, " ", K, " ", Sig, " ", Sig_)

        resize!(lp, K+1) 
        Mu, Sig = Mu_, Sig_ 
        @inbounds for k in 1:K+1 
            n_k = sum(C .== k) - (C[i] == k) 
            lp[k] = dlogpdf(MvNormal(Mu[:, k], Sig[:, :, k]), x) 
            lp[k] += (k != K+1 ? log(n_k+α) : log(α) + logV[ℓ+1] - logV[ℓ])             
        end  
        # println(lp, "  ", exp.(lp .- maximum(lp)) / sum(exp.(lp .- maximum(lp))))
        C[i] = gumbel_max_sample(lp) 
 
        n_map = countmap(C)    
        nz_K = keys(n_map) |> collect  
        ℓ = length(nz_K)

        # Mu_mc, Sig_mc = post_sample_gauss_kernels_mc(X, ℓ, t_max, Mu, Sig, C, config)
        # Ẑ = numerical_Zhat(Mu_mc, Sig_mc, g₀, ℓ, t_max)
        logpdf_K = log_p_K(ℓ+t_max-1, ℓ, n)
        # K_ = post_sample_K(logpdf_K, Ẑ, Zₖ, ℓ, t_max) 
        K_ = sample_K(logpdf_K, ℓ, t_max, n) 
        println("$K $K_  iter: $i")

        Mu_ = zeros(dim, K_+1)
        Sig_ = zeros(dim, dim, K_+1)  

        @inbounds for (i_nz, nz_k) in enumerate(nz_K)
            C[C .== nz_k] .= -1 * i_nz  # ensure there is no index collision
        end 
        C .*= -1
        Mu_[:, 1:ℓ] .= Mu[:, nz_K]
        Sig_[:, :, 1:ℓ] .= Sig[:, :, nz_K]
        post_sample_repulsive_gauss!(X, Mu_, Sig_, C, config)

        Mu, Sig = Mu_, Sig_ 
 
        llhd += lp[C[i]]
    end  
    return llhd
end   


function sample_K(logpdf_K, ℓ, t_max, n)::Int
    # println(length(logpdf_K), "  ", ℓ, "  ", t_max)
    # if length(logpdf_K) < ℓ + t_max
    #     log_p_k_extend!(logpdf_K, ℓ+t_max-length(logpdf_K), n)
    # end   
    # kk = deepcopy(logpdf_K)
    # kk .-= maximum(logpdf_K)
    # kk .= exp.(kk) 
    # kk ./= sum(kk)

    # k = sample(1:length(kk) |> Vector, Weights(kk)) # 
    k = gumbel_max_sample(logpdf_K)
    # println(logpdf_K, "  prior  ", ℓ, "  ", k)
    # println("sampled k: $k  ", sample(1:length(kk) |> Vector, Weights(kk)))
    return ℓ + k - 1
end 


function post_sample_K(logpdf_K, Ẑ, Zₖ, ℓ, t_max)::Int
    # if length(logpdf_K) < ℓ + t_max
    #     log_p_k_extend!(logpdf_K, ℓ + t_max - length(logpdf_K))
    # end 
    k = gumbel_max_sample(
        logpdf_K .+ Ẑ .- Zₖ[ℓ:ℓ+t_max-1]
    )
    
    # println("ℓ: $ℓ, t_max: $t_max")
    # println("wah...", size(logpdf_K), " ", size(Ẑ), " ", t_max)
    # println(logpdf_K .+ Ẑ .- Zₖ[ℓ:ℓ+t_max-1], "  post   ", logpdf_K)
    # println("k: $k \n\n")
    # Note that logpdf_K and Zₖ are precomputed, while Ẑ will be updated accordingly 
    return ℓ + k - 1
end 


function post_sample_gauss_kernels!(X, Mu, Sig, C, config::Dict) 
    g₀ = config["g₀"] 
    a₀ = config["a₀"] 
    b₀ = config["b₀"]

    dim, K = size(Mu)  
    gauss = MvNormal(zeros(dim), Matrix(I, dim, dim)) 
    
    @inbounds for k in 1:K
        X_tmp = X[:, C.==k] 
        n = size(X_tmp, 2) 

        if n == 0 
            Mu[:, k] .= rand(gauss)
            Sig[:, :, k] .= rand_inv_gamma(a₀, b₀, config)
        else 
            aₖ = a₀ + n / 2 
            bₖ = b₀ .+ sum((X_tmp .- Mu[:, k]).^2; dims=2) / 2 |> vec
            Sig[:, :, k] .= rand_inv_gamma(aₖ, bₖ, config)

            x̄ = mean(X_tmp; dims=2)  
            Σ₀ = inv(I + n * inv(Sig[:, :, k]))
            μ₀ = Σ₀ * (n * inv(Sig[:, :, k]) * x̄) |> vec 

            print(Sig[:, :, k], "  ", Σ₀, " ", (n * inv(Sig[:, :, k]) * x̄))
            println(x̄, "  ", μ₀)
            Mu[:, k] .= MvNormal(μ₀, Σ₀) |> rand
        end  
    end 
end 
 

function post_sample_gauss_kernels_mc(X, ℓ, t_max, Mu, Sig, C, config::Dict; n_mc=100) 
    g₀ = config["g₀"] 
    a₀ = config["a₀"] 
    b₀ = config["b₀"]

    dim = size(Mu, 1)  
    K = ℓ + t_max - 1
    Mu_mc = zeros(dim, K, n_mc)
    Sig_mc = zeros(dim, dim, K, n_mc)
    
    gauss = MvNormal(zeros(dim), Matrix(I, dim, dim)) 
    
    @inbounds for k in 1:K
        X_tmp = X[:, C.==k] 
        n = size(X_tmp, 2) 

        if n == 0 
            Mu_mc[:, k, :] .= rand(gauss, n_mc)
            @inbounds for mc = 1:n_mc
                Sig_mc[:, :, k, mc] .= rand_inv_gamma(a₀, b₀, config)
            end 
        else 
            x̄ = mean(X_tmp; dims=2)  
            
            Σ₀ = pinv(I + n * pinv(Sig[:, :, k]))
            μ₀ = Σ₀ * (n * pinv(Sig[:, :, k]) * x̄) |> vec 
            Mu_mc[:, k, :] .= rand(MvNormal(μ₀, Σ₀), n_mc)

            aₖ = a₀ + n / 2 
            bₖ = b₀ .+ sum((X_tmp .- Mu[:, k]).^2; dims=2) / 2 |> vec
            @inbounds for mc = 1:n_mc 
                Sig_mc[:, :, k, mc] .= rand_inv_gamma(aₖ, bₖ, config; n=n_mc) 
            end
        end  
    end 
    return Mu_mc, Sig_mc
end 


function rand_inv_gamma(a::Float64, b::Float64, config; n=1)::Diagonal
    dim = get(config, "dim", missing)
    return rand_inv_gamma(a, fill(b, dim), config; n=n)
end 

 
function rand_inv_gamma(a::Float64, bb::Vector{Float64}, config; n=1)::Diagonal
    dim = config["dim"]
    l_σ2 = config["l_σ2"]
    u_σ2 = config["u_σ2"]

    λs = zeros(dim)
    @inbounds for p in 1:dim
        # σ2 = InverseGamma(a, bb[p]) |> rand 
        # while σ2 > u_σ2 || σ2 < l_σ2
        #     σ2 = InverseGamma(a, bb[p]) |> rand 
        # end 
        # λs[p] = σ2
        σ2 = truncated(Gamma(a, 1/bb[p]), l_σ2, u_σ2) |> rand
        λs[p] = 1/σ2
    end   
    return Diagonal(λs)
end 


function sample_repulsive_gauss!(X, Mu, Sig, C, ℓ, config) 
    min_d = 0.     # min wasserstein distance 
    g₀ = config["g₀"] 
    a₀ = config["a₀"] 
    b₀ = config["b₀"]
    dim = config["dim"]

    K = size(Mu, 2)
    N = MvNormal(zeros(dim), I(dim))
    while rand() > min_d   
        @inbounds for k in ℓ+1:K 
            Mu[:, k] .= rand(N)
            Sig[:, :, k] .= rand_inv_gamma(a₀, b₀, config)
        end
        min_d = min_wass_distance(Mu, Sig, g₀)  
    end 
end 


function post_sample_repulsive_gauss!(X, Mu, Sig, C, config)::Int
    min_d = 0.     # min wasserstein distance
    reject_counts = 0 

    g₀ = config["g₀"]
    while rand() > min_d 
        post_sample_gauss_kernels!(X, Mu, Sig, C, config)
        reject_counts += 1 
        min_d = min_wass_distance(Mu, Sig, g₀)
    end
    return reject_counts 
end 
 

function initialize!(X, Mu, Sig, C, config)  
    size(Mu, 2) == size(Sig, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    
    min_d = 0.
    n = size(X, 2)

    g₀ = get(config, "g₀", missing)
    a₀ = get(config, "a₀", missing) 
    b₀ = get(config, "b₀", missing)

    dim, K = size(Mu)  
    gauss = MvNormal(zeros(dim), Matrix(I, dim, dim)) 
    while rand() > min_d   
        @inbounds for k = 1:K
            Mu[:, k] .= rand(gauss)
            Sig[:, :, k] .= rand_inv_gamma(a₀, b₀, config)
        end
        min_d = min_wass_distance(Mu, Sig, g₀) 
    end 
    C[:] .= sample(1:K-1, n, replace=true)
 
    # N(k) = MvNormal(Mu[:, k], Sig[:, :, k])
    # @inbounds for i = 1:n 
    #     # Make sure that the last cluster is not assigned anything 
    #     C[i] = map(k -> dlogpdf(N(k), X[:, i]), 1:K-1) |> gumbel_max_sample
    # end 
end 


function min_wass_distance(Mu, Sig, g₀)
    size(Mu, 2) == size(Sig, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))

    K = size(Mu, 2)  
    min_d = 1.
    @inbounds for i = 1:K, j = i+1:K-1   
        d = wass_gauss(
            Mu[:, i], Sig[:, :, i], Mu[:, j], Sig[:, :, j]) 
        min_d = min(min_d, d/(d + g₀))
    end  
    return min_d
end 
