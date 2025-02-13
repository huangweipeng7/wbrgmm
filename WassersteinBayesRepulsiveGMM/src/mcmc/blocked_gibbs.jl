function blocked_gibbs(
    X::Matrix{Float64};
    g₀::Float64 = 100., K::Int = 5, α::Float64 = 1., 
    a₀::Float64 = 1., b₀::Float64 = 1., 
    l_σ2::Float64 = 0.001, u_σ2::Float64 = 10000.,
    t_max::Int = 2,
    burnin::Int = 2000, runs::Int = 3000, thinning::Int = 1)

    dim, n = size(X)

    config = Dict(
        "g₀" => g₀, "α" => α, "a₀" => a₀, "b₀" => b₀,
        "t_max" => t_max, "dim" => dim, 
        "l_σ2" => l_σ2, "u_σ2" => u_σ2)

    C_mc = Vector()
    Mu_mc = Vector()
    Sigma_mc = Vector()
    llhd_mc = Vector()
  
    C = zeros(Int, n) 
    Mu = zeros(Float64, dim, K)
    Sigma = zeros(Float64, dim, dim, K)
    initialize!(X, Mu, Sigma, C, config) 

    logV = logV_nt(n, t_max)
    Zₖ = numerical_ZK(4K, dim, config)
    
    iter = ProgressBar(1:(burnin+runs))
    @inbounds for run in iter
        llhd = post_sample_C!(
            X, Mu, Sigma, C, logV, Zₖ, config)
        set_description(
            iter, "loglikelihood: $(round(llhd, sigdigits=3))")

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


function post_sample_C!(X, Mu, Sig, C, logV, Zₖ, config)::Float64
    size(Mu, 2) == size(Sig, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions.")) 

    dim = config["dim"]
    t_max = config["t_max"]
    α = config["α"] 

    n = size(X, 2)
    K = size(Mu, 2)    
    lp = Vector{Float64}(undef, n) 

    n_map = countmap(C) 
    llhd = 0.           # Log likelihood
    c_ = -1             # a fake index value for c underscore 
    @inbounds for i in 1:n 
        fill!(lp, -Inf)
        x = X[:, i]

        ℓ = (length ∘ unique)(C) - (sum(C .== C[i]) == 1) 
        n_map[C[i]] -= 1

        # Sample K 
        logpdf_K = log_p_K(ℓ+t_max, ℓ, n)
        old_K = K
        K = sample_K(logpdf_K, ℓ, t_max, n) 

        Mu_ = zeros(dim, K+1)
        Sig_ = zeros(dim, dim, K+1)
        nz_K = keys(n_map) |> collect |> sort

        @inbounds for k_ind in 1:old_K 
            if !haskey(n_map, k_ind)
                z_k = k_ind 
                break 
            end 
        end 

        if ℓ != K
            ℓ = length(nz_K)
            Mu_[:, 1:end-1] .= Mu[:, nz_K]
            Sig_[:, :, 1:end-1] .= Sig[:, :, nz_K]

            Mu_[:, end] .= Mu[:, z_k]
            Sig_[:, :, end] .= Sig[:, :, z_k]

            for (i_nz, nz_k) in enumerate(nz_K)
                C[C .== nz_k] .= -1 * i_nz  # ensure there is no index collision
            end 
            C[i] = 0
            C .*= -1
        end 
        sample_repulsive_gauss!(X, Mu_, Sig_, C, ℓ, config)

        Mu, Sig = Mu_, Sig_ 

        @inbounds for k in 1:K 
            n_k = sum(C .== K) - (C[i] == K) 
            if n_k == 0 || c_ == -1 
                lp[k] = dlogpdf(MvNormal(Mu[:, k], Sig[:, :, k]), x) 
                lp[k] += (n_k == 0 ? log(α) + logV[ℓ+1] - logV[ℓ] : log(n_k+α)) 
            end 

            if n_k == 0 && c_ != -1
                c_ = k 
            end            
        end  
        kᵢ = gumbel_max_sample(lp)
        C[i] = kᵢ
        n_map[kᵢ] = get(n_map, kᵢ, 0) + 1

        old_K = K
        K = sample_K(logpdf_K, ℓ, t_max, n) 

        Mu_ = zeros(dim, K+1)
        Sig_ = zeros(dim, dim, K+1)
        nz_K = keys(n_map) |> collect |> sort

        ℓ = length(nz_K)
        # Mu_[:, 1:end-1] .= Mu[:, nz_K]
        # Sig_[:, :, 1:end-1] .= Sig[:, :, nz_K]

        # Mu_[:, end] .= Mu[:, z_k]
        # Sig_[:, :, end] .= Sig[:, :, z_k]

        for (i_nz, nz_k) in enumerate(nz_K)
            C[C .== nz_k] .= -1 * i_nz  # ensure there is no index collision
        end 
        C .*= -1
        post_sample_repulsive_gauss!(X, Mu_, Sig_, C, ℓ, config)

        Mu, Sig = Mu_, Sig_ 


        llhd += lp[kᵢ]
    end  
    return llhd
end   


function sample_K(logpdf_K, ℓ, t_max, n)::Int
    # println(length(logpdf_K), "  ", ℓ, "  ", t_max)
    # if length(logpdf_K) < ℓ + t_max
    #     log_p_k_extend!(logpdf_K, ℓ+t_max-length(logpdf_K), n)
    # end  
    ss = logpdf_K[ℓ:ℓ+t_max]
    ss .-= maximum(ss)
    println(exp.(ss), ", ", logpdf_K[ℓ:ℓ+t_max])
    return ℓ + gumbel_max_sample(logpdf_K[ℓ:ℓ+t_max]) - 1
end 


function post_sample_K(logpdf_K, Ẑ, Zₖ, ℓ, t_max)::Int
    # if length(logpdf_K) < ℓ + t_max
    #     log_p_k_extend!(logpdf_K, ℓ + t_max - length(logpdf_K))
    # end 

    # Note that logpdf_K and Zₖ are precomputed, while Ẑ will be updated accordingly 
    return ℓ + gumbel_max_sample(
        logpdf_K[ℓ:ℓ+t_max] .+ Ẑ .- Zₖ[ℓ:ℓ+t_max]
    ) - 1
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


function sample_repulsive_gauss!(X, Mu, Sigma, C, ℓ, config) 
    min_d = 0.     # min wasserstein distance 
    g₀ = config["g₀"] 
    a₀ = config["a₀"] 
    b₀ = config["b₀"]
    dim = config["dim"]

    K = size(Mu, 2)
    gauss = MvNormal(zeros(dim), Matrix(I, dim, dim))
    while rand() > min_d   
        @inbounds for k in ℓ+1:K 
            Mu[:, k] .= rand(gauss)
            Sigma[:, :, k] .= rand_inv_gamma(a₀, b₀, config)
        end
        min_d = min_wass_distance(Mu, Sigma, g₀)  
    end 
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
 

function initialize!(X, Mu, Sigma, C, config)  
    size(Mu, 2) == size(Sigma, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    
    n = size(X, 2)

    g₀ = get(config, "g₀", missing)
    a₀ = get(config, "a₀", missing) 
    b₀ = get(config, "b₀", missing)

    dim, K = size(Mu) 
    min_d = 0.
    gauss = MvNormal(zeros(dim), Matrix(I, dim, dim))
    # @inbounds for k in 1:K 
    #     Mu[:, k] .= rand(gauss)
    #     Sigma[:, :, k] .= rand_inv_gamma(a₀, b₀, config)
    # end
    while rand() > min_d   
        @inbounds for k in 1:K 
            Mu[:, k] .= rand(gauss)
            Sigma[:, :, k] .= rand_inv_gamma(a₀, b₀, config)
        end
        min_d = min_wass_distance(Mu, Sigma, g₀) 
    end 
 
    @inbounds for i = 1:n 
        C[i] = map(
            k -> dlogpdf(MvNormal(Mu[:, k], Sigma[:, :, k]), X[:, i]),
            1:K
        ) |> argmax
    end 
end 


function min_wass_distance(Mu, Sigma, g₀)::Float64 
    size(Mu, 2) == size(Sigma, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))

    K = size(Mu, 1)
    min_d = Inf
    @inbounds for i = 1:K, j = i+1:K-1  
        println(Mu[:, i], Mu[:, j])
        d = wass_gauss(
            Mu[:, i], Sigma[:, :, i], Mu[:, j], Sigma[:, :, j])
        d = d / (d + g₀)
        min_d = min(min_d, d)
    end  
    return min_d
end 
