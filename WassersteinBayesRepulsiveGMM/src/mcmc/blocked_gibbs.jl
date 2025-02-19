function blocked_gibbs(
    X::Matrix{Float64};
    g₀ = 100., β = 1., a₀ = 1., b₀ = 1., 
    l_σ2 = 0.001, u_σ2 = 10000., τ = 1.,
    K::Int = 5, t_max::Int = 2,
    burnin::Int = 2000, runs::Int = 3000, thinning::Int = 1)

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
    Mu = zeros(Float64, dim, K+1)
    Sig = zeros(Float64, dim, dim, K+1)
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

        set_description(
            pbar, "loglikelihood: $(round(llhd, sigdigits=3))")
        
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


gumbel_max_sample(lp)::Int = lp + rand(GUMBEL, size(lp)) |> argmax


function post_sample_C!(
    X::Matrix, Mu::Matrix, Sig::Array, 
    C::Vector, logV::Vector, Zₖ::Vector, config::Dict) 

    size(Mu, 2) == size(Sig, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions.")) 
 
    t_max = config["t_max"]
    β = config["β"]  

    n = size(X, 2)
    K = size(Mu, 2)     
 
    lp = Vector{Float64}()
    lc = Vector{Float64}()
    llhd = 0.  
    for i = 1:n  
        x = vec(X[:, i])

        C_, Mu_, Sig_, ℓ = sample_K_and_swap_indices(
            i, n, C, Mu, Sig, t_max; exclude_i=true)
        @inbounds C .= C_ 
        @inbounds C[i] = -1       # pseudo component assignment
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
            lp[k] = dlogpdf(MvNormal(Mu_[:, k], Sig_[:, :, k]), x) 
            lc[k] = k != Kp1 ? log(n_k+β) : log(β) + logV[ℓ+1] - logV[ℓ]  
        end
        Mu, Sig = Mu_, Sig_ 
        C[i] = gumbel_max_sample(lp .+ lc) 
        llhd += lp[C[i]] 
    end  
    return C, Mu, Sig, llhd
end   


function sample_K_and_swap_indices(
    i, n, C, Mu, Sig, t_max; 
    exclude_i=false, approx=true, X=nothing, 
    Zₖ=nothing, config=nothing, n_mc=2000)::Tuple
    
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
    Mu_ = zeros(Float64, dim, K+1)
    Sig_ = zeros(Float64, dim, dim, K+1)

    Mu_[:, 1:ℓ] .= Mu[:, nz_k_inds]
    Sig_[:, :, 1:ℓ] .= Sig[:, :, nz_k_inds]  
    
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


sample_K(log_p_K, ℓ, t_max, n) = ℓ + gumbel_max_sample(log_p_K) - 1


post_sample_K(log_p_K, Ẑ, Zₖ, ℓ, t_max) = 
    ℓ + gumbel_max_sample(log_p_K .+ Ẑ .- Zₖ[ℓ:ℓ+t_max-1]) - 1


function post_sample_gauss_kernels!(
    X::Matrix, Mu::Matrix, Sig::Array, C::Vector, config::Dict) 
    
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
            μ₀ = Σ₀ * (inv(Sig[:, :, k]) * x_sum) |> vec 
            Mu[:, k] .= MvNormal(μ₀, Σ₀) |> rand

            aₖ = a₀ + n / 2 
            bₖ = b₀ .+ sum((X_tmp .- Mu[:, k]).^2; dims=2) / 2 |> vec
            Sig[:, :, k] .= rand_inv_gamma(aₖ, bₖ, config)
        end  
    end 
end 
 

function post_sample_gauss_kernels_mc(X, ℓ, t_max, Mu, Sig, C, config::Dict; n_mc=20) 
    g₀ = config["g₀"] 
    a₀ = config["a₀"] 
    b₀ = config["b₀"]
    τ = config["τ"]

    dim = size(Mu, 1)  
    K = ℓ + t_max - 1
    Mu_mc = zeros(Float64, dim, K, n_mc)
    Sig_mc = zeros(Float64, dim, dim, K, n_mc)
    
    normal = MvNormal(zeros(dim), τ^2) 
    @inbounds for k in 1:K
        X_tmp = X[:, C.==k] 
        n = size(X_tmp, 2) 

        if n == 0 
            @inbounds Mu_mc[:, k, :] .= rand(normal, n_mc)
            for mc = 1:n_mc
                @inbounds Sig_mc[:, :, k, mc] .= rand_inv_gamma(a₀, b₀, config)
            end
        else
            x_sum = sum(X_tmp; dims=2)  
            Σ₀ = inv(τ^2*I + n * inv(Sig[:, :, k]))
            μ₀ = Σ₀ * (inv(Sig[:, :, k]) * x_sum) |> vec 
            Mu_mc[:, k, :] .= rand(MvNormal(μ₀, Σ₀), n_mc)

            aₖ = a₀ + n / 2 
            bₖ = b₀ .+ sum((X_tmp .- Mu[:, k]).^2; dims=2) / 2 |> vec
            Sig_mc[:, :, k, :] .= rand_inv_gamma(aₖ, bₖ, config; n=n_mc) 
        end  
    end 
    return Mu_mc, Sig_mc
end 


function rand_inv_gamma(a::Float64, b::Float64, config::Dict; n=1)::Diagonal
    dim = config["dim"]
    return rand_inv_gamma(a, fill(b, dim), config; n=n)
end 

 
function rand_inv_gamma(a::Float64, bb::Vector{Float64}, config::Dict; n=1) 
    dim = config["dim"]
    l_σ2 = config["l_σ2"]
    u_σ2 = config["u_σ2"]

    Λ = n == 1 ? Diagonal(zeros(Float64, dim)) : zeros(Float64, dim, dim, n) 
    @inbounds for p in 1:dim
        if n == 1
            # Using gamma to sample inverse gamma R.V. is always more robust in Julia
            σ2 = truncated(Gamma(a, 1/bb[p]), 1/u_σ2, 1/l_σ2) |> rand 
            @assert σ2 > 0
            Λ[p, p] = 1 / σ2
        else 
            # Using gamma to sample inverse gamma R.V. is always more robust in Julia
            σ2 = rand(truncated(Gamma(a, 1/bb[p]), 1/u_σ2, 1/l_σ2), n) 
            @assert all(σ2 .> 0)
            Λ[p, p, :] .= 1 ./ σ2
        end  
    end  
    return Λ
end 


function sample_repulsive_gauss!(
    X::Matrix, Mu::Array, Sig::Array, ℓ::Int, config::Dict)

    g₀ = config["g₀"] 
    a₀ = config["a₀"] 
    b₀ = config["b₀"]
    τ = config["τ"] 

    dim, K = size(Mu)
    normal = MvNormal(zeros(dim), τ^2)

    min_d = 0.     # min wasserstein distance 
    while rand() > min_d   
        @inbounds for k in ℓ+1:K 
            Mu[:, k] .= rand(normal)
            Sig[:, :, k] .= rand_inv_gamma(a₀, b₀, config)
        end 
        min_d = min_wass_distance(Mu, Sig, g₀)
    end 
end 


function post_sample_repulsive_gauss!(
    X::Matrix, Mu::Array, Sig::Array, C::Vector, config::Dict)::Int
    
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
 

function initialize!(
    X::Matrix, Mu::Array, Sig::Array, C::Vector, config::Dict) 
    
    size(Mu, 2) == size(Sig, 3) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    
    dim, K = size(Mu)

    sample_repulsive_gauss!(X, Mu, Sig, 0, config)
    # C[:] .= sample(1:K-1, n, replace=true) 

    normal_k(k) = MvNormal(Mu[:, k], Sig[:, :, k])
    @inbounds for i in eachindex(C)
        # Make sure that the last cluster is not assigned anything
        C[i] = (argmax ∘ map)(
            k -> dlogpdf(normal_k(k), X[:, i]), 1:K-1)
    end
end 
 