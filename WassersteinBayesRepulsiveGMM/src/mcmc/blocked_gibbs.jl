@inline function wrbgmm_blocked_gibbs(
    X; g₀ = 100., β = 1., 
    a₀ = 1., b₀ = 1., τ = 1.,
    l_σ2 = 0.001, u_σ2 = 10000.,   
    prior = nothing,
    K = 5, t_max = 2, burnin = 2000, runs = 3000, thinning = 1)

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
    initialize!(X, Mu, Sig, C, config, prior)  

    logV = logV_nt(n, β, 2K)
    Zₖ = nothing # numerical_Zₖ(2K, dim, config, prior)
    
    pbar = ProgressBar(1:(burnin+runs))
    @inbounds for iter in pbar #1:(burnin+runs)
        # println(iter)
        C, Mu, Sig, llhd = post_sample_C!(
            X, Mu, Sig, C, logV, Zₖ, config, prior) 

        C, Mu, Sig, _ = post_sample_K_and_swap_indices(
            X, C, Mu, Sig, Zₖ, t_max; approx=true)  
        
        post_sample_repulsive_gauss!(X, Mu, Sig, C, config, prior) 

        set_description(pbar, f"log-likelihood: {llhd:.3f}")
        
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


@inline function post_sample_C!(X, Mu, Sig, C, logV, Zₖ, config, prior) 

    size(Mu, 2) == size(Sig, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions.")) 
 
    t_max = config["t_max"]
    β = config["β"]  
    logβ = log(β)

    n = size(X, 2)
    K = size(Mu, 2)     
 
    lp = Vector()
    lc = Vector()
    llhd = 0.  
    for i = 1:n 
       @inbounds x = X[:, i] 

        C_, Mu_, Sig_, ℓ = sample_K_and_swap_indices(
            i, n, C, Mu, Sig, t_max; exclude_i=true)

        C .= C_ 
        # pseudo component assignment
        C[i] = -1                     
        @assert maximum(C) <= ℓ

        # K plus 1
        Kp1 = size(Mu_, 2)  

        sample_repulsive_gauss!(X, Mu_, Sig_, ℓ, config, prior)
 
        # resize the vectors for loglikelihood of data and the corresponding coefficients
        resize!(lp, Kp1) 
        resize!(lc, Kp1) 
        for k = 1:Kp1 
            n_k = sum(C .== k) 
            lp[k] = logpdf(MvNormal(Mu_[:, k], Sig_[:, :, k]), x) 
            lc[k] = k != Kp1 ? log(n_k + β) : logβ + logV[ℓ+1] - logV[ℓ]  
        end
        Mu, Sig = Mu_, Sig_ 
        C[i] = gumbel_max_sample(lp .+ lc)
        llhd += lp[C[i]] 
    end  
    return C, Mu, Sig, llhd
end   


@inline function sample_K_and_swap_indices(
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


@inline function post_sample_K_and_swap_indices(
    X, C, Mu, Sig, Zₖ, t_max; 
    approx=true, config=nothing, n_mc=nothing) 

    n = size(X, 2)
    return sample_K_and_swap_indices(
        nothing, n, C, Mu, Sig, t_max; 
        exclude_i=true, approx=approx, X=X, Zₖ=Zₖ, 
        config=config, n_mc=n_mc)
end 


@inline sample_K(log_p_K, ℓ, t_max, n) = 
    ℓ + gumbel_max_sample(log_p_K) - 1


@inline post_sample_K(log_p_K, Ẑ, Zₖ, ℓ, t_max) = 
    ℓ + gumbel_max_sample(log_p_K .+ Ẑ .- Zₖ[ℓ:ℓ+t_max-1]) - 1
 

@inline function post_sample_repulsive_gauss!(X, Mu, Sig, C, config, k_prior)
    min_d = 0.              # min wasserstein distance
    reject_counts = 0 

    g₀ = config["g₀"]
    while rand() > min_d
        reject_counts += 1 
        post_sample_gauss!(X, Mu, Sig, C, k_prior)
        min_d = min_wass_distance(Mu, Sig, g₀)
    end
    return reject_counts 
end 
 

# @inline function initialize!(X, Mu, Sig, C, config) 
#     size(Mu, 2) == size(Sig, 3) ||
#         throw(DimensionMismatch("Inconsistent array dimensions."))
    
#     K = size(Mu, 2)

#     sample_repulsive_gauss!(X, Mu, Sig, 0, config)

#     # Inner functions: Assign the component index to each 
#     # observation according to their max log-likelihoods
#     log_i_k(i, k) = logpdf(MvNormal(Mu[:, k], Sig[:, :, k]), X[:, i])
 
#     if Threads.nthreads() > 1
#         Threads.@threads for i in eachindex(C)
#             # Make sure that the last cluster is not assigned anything
#             C[i] = log_i_k.(Ref(i), 1:K-1) |> argmax
#         end
#     else     
#         for i in eachindex(C)
#             # Make sure that the last cluster is not assigned anything
#             C[i] = log_i_k.(Ref(i), 1:K-1) |> argmax
#         end
#     end 
# end 


@inline function initialize!(X, Mu, Sig, C, config, prior) 
    size(Mu, 2) == size(Sig, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    
    K = size(Mu, 2)

    sample_gauss!(X, Mu, Sig, 0, config, prior)

    # Inner functions: Assign the component index to each 
    # observation according to their max log-likelihoods
    log_i_k(i, k) = logpdf(MvNormal(Mu[:, k], Sig[:, :, k]), X[:, i])
 
    # C .= sample(1:K-1, length(C), replace=true)
    if Threads.nthreads() > 1
        Threads.@threads for i in eachindex(C)
            # Make sure that the last cluster is not assigned anything
            C[i] = log_i_k.(Ref(i), 1:K-1) |> argmax
        end
    else     
        for i in eachindex(C)
            # Make sure that the last cluster is not assigned anything
            C[i] = log_i_k.(Ref(i), 1:K-1) |> argmax
        end
    end 
end 
 
 
@inline function sample_repulsive_gauss!(X, Mu, Sig, ℓ, config, k_prior)
    g₀ = config["g₀"]  
    K = size(Mu, 2)
    μ, Σ = nothing, nothing 
    min_d = 0.     # min wasserstein distance 
    while rand() > min_d   
        @inbounds for k in ℓ+1:K 
            μ, Σ = rand(k_prior)
            Mu[:, k] .= μ
            Sig[:, :, k] .= Σ 
        end 
        min_d = min_wass_distance(Mu, Sig, g₀) 
    end 
end 


@inline function sample_gauss!(X, Mu, Sig, ℓ, config, k_prior)
    g₀ = config["g₀"]  
    K = size(Mu, 2) 
    @inbounds for k in ℓ+1:K 
        μ, Σ = rand(k_prior)
        Mu[:, k] .= μ
        Sig[:, :, k] .= Σ 
    end 
end 