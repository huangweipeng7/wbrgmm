function wrbgmm_blocked_gibbs(
    X; g₀ = 100., β = 1., a₀ = 1., b₀ = 1., 
    l_σ2 = 0.001, u_σ2 = 10000., τ = 1., K = 5, 
    t_max = 2, burnin = 2000, runs = 3000, thinning = 1)

    dim, n = size(X)

    config = Dict(
        "g₀" => g₀, "β" => β, "a₀" => a₀, "b₀" => b₀,
        "t_max" => t_max, "dim" => dim, "τ" => τ,
        "l_σ2" => l_σ2, "u_σ2" => u_σ2)

    C_mc = Vector{Vector}()
    Mu_mc = Vector{Array}()
    Sig_mc = Vector{Array}()
    K_mc = Vector{Int}()
    llhd_mc = Vector{Float64}()
  
    C = zeros(Int, n) 
    Mu = zeros(dim, K+1)
    Sig = zeros(dim, dim, K+1)
    initialize!(X, Mu, Sig, C, config) 

    logV = logV_nt(n, t_max)
    Zₖ = numerical_Zₖ(2K, dim, config)
    
    pbar = ProgressBar(1:(burnin+runs))
    @inbounds for iter in pbar
        C, Mu, Sig, llhd = post_sample_C!(
            X, Mu, Sig, C, logV, Zₖ, config)
        
        C, Mu, Sig, _ = post_sample_K_and_swap_indices(
            X, C, Mu, Sig, Zₖ, t_max; approx=true) 
        
        post_sample_repulsive_gauss!(X, Mu, Sig, C, config)

        set_description(pbar, f"loglikelihood: {llhd:.3f}")
        
        if iter > burnin && iter % thinning == 0 
            push!(C_mc, deepcopy(C))
            push!(Mu_mc, deepcopy(Mu))  
            push!(Sig_mc, deepcopy(Sig))
            push!(K_mc, size(Mu, 2)-1)
            push!(llhd_mc, llhd)
        end
    end 
    return C_mc, Mu_mc, Sig_mc, K_mc, llhd_mc
end 


function post_sample_C!(X, Mu, Sig, C, logV, Zₖ, config) 

    size(Mu, 2) == size(Sig, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions.")) 
 
    t_max = config["t_max"]
    β = config["β"]  

    n = size(X, 2)
    K = size(Mu, 2)     
 
    lp = Vector()
    lc = Vector()
    llhd = 0.  
    for i = 1:n  
        x = X[:, i] 

        C_, Mu_, Sig_, ℓ = sample_K_and_swap_indices(
            i, n, C, Mu, Sig, t_max; exclude_i=true)

        C .= C_ 
        # pseudo component assignment
        C[i] = -1                     
        @assert maximum(C) <= ℓ

        # K plus 1
        Kp1 = size(Mu_, 2)  

        sample_repulsive_gauss!(X, Mu_, Sig_, ℓ, config)
        @inbounds for k = 1:Kp1
            @assert all(diag(Sig_[:, :, k]) .> 0) Sig_[:, :, k]
        end  
        
        # resize the vectors for loglikelihood of data and the corresponding coefficients
        resize!(lp, Kp1) 
        resize!(lc, Kp1) 
        @inbounds for k = 1:Kp1 
            n_k = sum(C .== k) 
            lp[k] = logpdf(MvNormal(Mu_[:, k], Sig_[:, :, k]), x) 
            lc[k] = k != Kp1 ? log(n_k+β) : log(β) + logV[ℓ+1] - logV[ℓ]  
        end
        Mu, Sig = Mu_, Sig_ 
        C[i] = categorical_from_log(lp .+ lc)
        llhd += lp[C[i]] 
    end  
    return C, Mu, Sig, llhd
end   


function sample_K_and_swap_indices(
    i, n, C, Mu, Sig, t_max; 
    exclude_i=false, approx=true, X=nothing, 
    Zₖ=nothing, config=nothing, n_mc=2000) 
    
    dim = size(Mu, 1)
    C_ = similar(C)
    inds = 1:n

    nz_k_inds = exclude_i ? unique(C[inds .!= i]) : unique(C)  
    ℓ = length(nz_k_inds) 

    # Sample K  
    log_p_K = log_prob_K(ℓ, t_max, n)
    if approx 
        K = sample_K(log_p_K, ℓ, t_max, n) 
    else
        Zₖ != nothing || 
            throw("In the non-approximation version, X cannot be nothing.")
        X != nothing ||
            throw("In the non-approximation version, Zₖ cannot be nothing.")
        config != nothing ||
            throw("In the non-approximation version, config cannot be nothing.")
        
        Mu_mc, Sig_mc = post_sample_gauss_kernels_mc(
            X, ℓ, t_max, Mu, Sig, C, config; n_mc=n_mc)

        Ẑ = numerical_Zhat(Mu_mc, Sig_mc, config["g₀"], ℓ, t_max)
        K = post_sample_K(log_p_K, Ẑ, Zₖ, ℓ, t_max)  
    end 

    # Always leave a place for a new cluster. Therefore, we use K+1
    Mu_ = zeros(dim, K+1)
    Sig_ = zeros(dim, dim, K+1)

    @inbounds Mu_[:, 1:ℓ] .= Mu[:, nz_k_inds]
    @inbounds Sig_[:, :, 1:ℓ] .= Sig[:, :, nz_k_inds]  
    
    @inbounds for (i_nz, nz_k_ind) in enumerate(nz_k_inds)
        C_[C .== nz_k_ind] .= i_nz
    end   
    return C_, Mu_, Sig_, ℓ  
end 


function post_sample_K_and_swap_indices(
    X, C, Mu, Sig, Zₖ, t_max; 
    approx=true, config=nothing, n_mc=nothing) 

    n = size(X, 2)
    return sample_K_and_swap_indices(
        nothing, n, C, Mu, Sig, t_max; 
        exclude_i=true, approx=approx, X=X, Zₖ=Zₖ, 
        config=config, n_mc=n_mc)
end 


sample_K(log_p_K, ℓ, t_max, n) = ℓ + categorical_from_log(log_p_K) - 1


post_sample_K(log_p_K, Ẑ, Zₖ, ℓ, t_max) = 
    ℓ + categorical_from_log(log_p_K .+ Ẑ .- Zₖ[ℓ:ℓ+t_max-1]) - 1


function post_sample_gauss_kernels!(X, Mu, Sig, C, config) 
    g₀ = config["g₀"] 
    a₀ = config["a₀"] 
    b₀ = config["b₀"]
    τ = config["τ"] 

    dim, K = size(Mu)  
    normal = MvNormal(zeros(dim), τ^2)  
    @inbounds for k = 1:K
        X_tmp = X[:, C.==k] 
        n = size(X_tmp, 2) 

        if n == 0 
            Mu[:, k] .= rand(normal)
            Sig[:, :, k] .= rand_inv_gamma(a₀, b₀, config)
        else  
            x_sum = sum(X_tmp; dims=2)  
            Σ₀ = inv(τ^2 * I + n * inv(Sig[:, :, k]))
            μ₀ = Σ₀ * (inv(Sig[:, :, k]) * x_sum) 
            Mu[:, k] .= MvNormal(μ₀|> vec, Σ₀) |> rand

            aₖ = a₀ + n / 2. 
            bₖ = b₀ .+ sum((X_tmp .- Mu[:, k]).^2; dims=2) / 2. |> vec
            Sig[:, :, k] .= rand_inv_gamma(aₖ, bₖ, config)
        end  
    end 
end 
 

function post_sample_gauss_kernels_mc(
    X, ℓ, t_max, Mu, Sig, C, config; n_mc=20) 
    
    g₀ = config["g₀"] 
    a₀ = config["a₀"] 
    b₀ = config["b₀"]
    τ = config["τ"]

    dim = size(Mu, 1)  
    K = ℓ + t_max - 1
    Mu_mc = zeros(dim, K, n_mc)
    Sig_mc = zeros(dim, dim, K, n_mc)
    
    normal = MvNormal(zeros(dim), τ^2) 
    @inbounds for k in 1:K
        X_tmp = X[:, C.==k] 
        n = size(X_tmp, 2) 

        if n == 0 
            Mu_mc[:, k, :] .= rand(normal, n_mc)
            Sig_mc[:, :, k, :] .= rand_inv_gamma_mc(a₀, b₀, n_mc, config)
        else
            x_sum = sum(X_tmp; dims=2)  
            Σ₀ = inv(τ^2*I + n * inv(Sig[:, :, k]))
            μ₀ = Σ₀ * (inv(Sig[:, :, k]) * x_sum) 
            Mu_mc[:, k, :] .= rand(MvNormal(μ₀ |> vec, Σ₀), n_mc)

            aₖ = a₀ + n / 2. 
            bₖ = b₀ .+ sum((X_tmp .- Mu[:, k]).^2; dims=2) / 2. |> vec
            Sig_mc[:, :, k, :] .= rand_inv_gamma_mc(aₖ, bₖ, n_mc, config) 
        end  
    end 
    return Mu_mc, Sig_mc
end 


function post_sample_repulsive_gauss!(X, Mu, Sig, C, config)
    min_d = 0.              # min wasserstein distance
    reject_counts = 0 

    g₀ = config["g₀"]
    while rand() > min_d 
        reject_counts += 1 
        post_sample_gauss_kernels!(X, Mu, Sig, C, config)
        min_d = min_wass_distance(Mu, Sig, g₀)
    end
    return reject_counts 
end 
 

function initialize!(X, Mu, Sig, C, config) 
    size(Mu, 2) == size(Sig, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    
    dim, K = size(Mu)

    sample_repulsive_gauss!(X, Mu, Sig, 0, config)

    # function inside another function
    normal_k(k) = MvNormal(Mu[:, k], Sig[:, :, k])
    @inbounds for i in eachindex(C)
        # Make sure that the last cluster is not assigned anything
        C[i] = (argmax ∘ map)(
            k -> logpdf(normal_k(k), X[:, i]), 1:K-1)
    end
end 
 