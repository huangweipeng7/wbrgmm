function blocked_gibbs(
    X:: jnp.Array, 
    g0: float = 100., 
    beta: float = 1., 
    a0: float = 1., 
    b0: float = 1., 
    l_sig2: float = 0.001, 
    u_sig2: float = 10000., 
    tau: float = 1.,
    K: int = 5, 
    t_max: int = 2,
    burnin: int = 2000, 
    runs: int = 3000, 
    thinning: int = 1
): 
    n, dim = X.shape

    config = {
        "g0": g0, "beta": beta, "a0": a0, "b0": b0,
        "t_max": t_max, "dim": dim, "tau": tau,
        "l_sig2": l_sig2, "u_sig2": u_sig2
    }

    C_mc = Vector{Vector}()
    Mu_mc = Vector{Array}()
    Sig_mc = Vector{Array}()
    llhd_mc = Vector{Float64}()
  
    C = zeros(Int, n) 
    Mu = zeros(Float64, dim, K+1)
    Sig = zeros(Float64, dim, dim, K+1)
    initialize!(X, Mu, Sig, C, config) 

    logV = logV_nt(n, t_max)
    Zₖ = numerical_Zₖ(2K, dim, config)
    
    pbar = ProgressBar(1:(burnin+runs))
    @inbounds for iter in pbar
        Mu, Sig, llhd = post_sample_C!(
            X, Mu, Sig, C, logV, Zₖ, config)
        post_sample_repulsive_gauss!(X, Mu, Sig, C, config)

        set_description(
            pbar, "loglikelihood: $(round(llhd, sigdigits=3))")
        
        if iter > burnin && iter % thinning == 0 
            push!(C_mc, deepcopy(C))
            push!(Mu_mc, deepcopy(Mu))  
            push!(Sig_mc, deepcopy(Sig))
            push!(llhd_mc, llhd)
        end
    end 
    return C_mc, Mu_mc, Sig_mc, llhd_mc
end 


gumbel_max_sample(lp)::Int = lp + rand(GUMBEL, size(lp)) |> argmax


function post_sample_C(X, Mu, Sig, C, logV, Zk, config):  
 
    t_max = config["t_max"]
    beta = config["beta"]  

    n = X.shape[0]
    K = Mu.shape[0]     
 
    lp = Vector{Float64}()
    lc = Vector{Float64}()
    llhd = 0.                       # Log likelihood
    for i in range(n):  
        x = X[i, :] 

        C_, Mu_, Sig_, ell = sample_K_and_swap_indices(
            i, n, C, Mu, Sig, t_max; exclude_i=true)
        @inbounds C = C_ 
        @inbounds C[i] = -1       # pseudo component assignment
        @assert maximum(C) <= ell

        # K plus 1
        Kp1 = size(Mu_, 2)  

        sample_repulsive_gauss!(X, Mu_, Sig_, C, ell, config)
        @inbounds for k = 1:Kp1
            @assert all(diag(Sig_[:, :, k]) .> 0) Sig_[:, :, k]
        end  
        
        # resize the vectors for loglikelihood of data and the corresponding coefficients
        resize!(lp, Kp1) 
        resize!(lc, Kp1) 
        @inbounds for k = 1:Kp1 
            n_k = sum(C == k) - (C[i] == k)  
            lp[k] = dlogpdf(MvNormal(Mu_[:, k], Sig_[:, :, k]), x) 
            lc[k] = (k != Kp1 ? log(n_k+beta) : log(beta) + logV[ell+1] - logV[ell]) 
        end
        Mu, Sig = Mu_, Sig_ 
        C[i] = gumbel_max_sample(lp .+ lc) 
        llhd += lp[C[i]]
   
        C_, Mu_, Sig_, _ = sample_K_and_swap_indices(
            i, n, C, Mu, Sig, t_max; exclude_i=false)
        C = C_

        Mu, Sig = Mu_, Sig_  
    end  
    return Mu, Sig, llhd
end   


function sample_K_and_swap_indices(
    i, n, C, Mu, Sig, t_max, 
    exclude_i=False, approx=True, X=None, Zk=None): 

    dim = Mu.shape[1]
    C_ = jnp.zeros_like(C)
    inds = jnp.arange(n)

    if exclude_i:
        cluster_count = Counter(C[inds!=i]) 
    else: 
        cluster_count = Counter(C)  
    nz_k_inds = list(cluster_count.keys()) 
    ell = length(nz_k_inds) 

    # Sample K  
    log_p_K = log_prob_K(ell, t_max, n)
    if approx: 
        K = sample_K(log_p_K, ell, t_max, n) 
    else:
        assert Zhat is not None
        assert X is not None 
        
        Mu_mc, Sig_mc = post_sample_gauss_kernels_mc(
            X, ell, t_max, Mu, Sig, C, config, n_mc=100)
        Zhat = numerical_Zhat(Mu_mc, Sig_mc, g0, ell, t_max)
        K = post_sample_K(log_p_K, Zhat, Zk, ell, t_max)   

    # Always leave a place for a new cluster. Therefore, we use K+1
    Mu_ = zeros(Float64, dim, K+1)
    Sig_ = zeros(Float64, dim, dim, K+1)

    Mu_[1:ell, :] = Mu[nz_k_inds, :]
    Sig_[1:ell, :, :] = Sig[nz_k_inds, :, :]  
    
    for i_nz, nz_k_ind in enumerate(nz_k_inds):
        C_[C == nz_k_ind] = i_nz

    return C_, Mu_, Sig_, ell   


def sample_K(log_p_K, ell, t_max, n):
    return ell + gumbel_max_sample(log_p_K) - 2


def post_sample_K(log_p_K, Zhat, Zk, ell, t_max):  
    return ell + gumbel_max_sample(log_p_K + Zhat - Zk[ell-1:ell+t_max-2]) - 2


function post_sample_gauss_kernels!(X, Mu, Sig, C, config::Dict) 
    g0 = config["g0"] 
    a0 = config["a0"] 
    b0 = config["b0"]
    tau = config["tau"] 

    dim, K = size(Mu)  
    normal = stats.multivariate_normal(jnp.zeros(dim), tau^2)  
    
    tau_sq = tau ** 2
    for k in range(K):
        X_tmp = X[C==k, :] 
        n = X_tmp.shape[0]
        
        if n == 0: 
            Mu[k, :] = normal.rv()
            Sig[k, :, :] = rand_inv_gamma(a0, b0, config)
        else:  
            x_sum = X_tmp.sum(axis=0)  
            sig0 = jnp.linalg.inv(
                tau_sq * jnp.eye(dim) + n * jnp.linalg.inv(Sig[k, :, :])
            )
            mu0 = sig0 * (jnp.linalg.inv(Sig[k, :, :]) @ x_sum.T) 
            Mu[k, :] = stats.multivariate_normal.rv(mu0, Sig[k, :, :])

            a_k = a0 + n / 2 
            b_k = b0 .+ jnp.sum((X_tmp - Mu[k, :]).pow(2), axis=0) / 2  
            Sig[k, :, :] = rand_inv_gamma(a_k, b_k, config)
 

# function post_sample_gauss_kernels_mc(X, ell, t_max, Mu, Sig, C, config::Dict; n_mc=20) 
#     g0 = config["g0"] 
#     a0 = config["a0"] 
#     b0 = config["b0"]
#     tau = config["tau"]

#     dim = size(Mu, 1)  
#     K = ell + t_max - 1
#     Mu_mc = zeros(Float64, dim, K, n_mc)
#     Sig_mc = zeros(Float64, dim, dim, K, n_mc)
    
#     normal = MvNormal(zeros(dim), tau^2) 
#     @inbounds for k in 1:K
#         X_tmp = X[:, C==k] 
#         n = size(X_tmp, 2) 

#         if n == 0 
#             @inbounds Mu_mc[:, k, :] = rand(normal, n_mc)
#             for mc = 1:n_mc
#                 @inbounds Sig_mc[:, :, k, mc] = rand_inv_gamma(a0, b0, config)
#             end
#         else
#             x_sum = sum(X_tmp; dims=2)  
#             Σ0 = inv(tau^2*I + n * inv(Sig[:, :, k]))
#             μ0 = Σ0 * (inv(Sig[:, :, k]) * x_sum) |> vec 
#             Mu_mc[:, k, :] = rand(MvNormal(μ0, Σ0), n_mc)

#             aₖ = a0 + n / 2 
#             bₖ = b0 .+ sum((X_tmp .- Mu[:, k]).^2; dims=2) / 2 |> vec
#             Sig_mc[:, :, k, :] = rand_inv_gamma(aₖ, bₖ, config; n=n_mc) 
#         end  
#     end 
#     return Mu_mc, Sig_mc
# end 


def rand_inv_gamma(a, b, config, n=1): 
    dim = config["dim"]
    return rand_inv_gamma(a, fill(b, dim), config, n=n)
  
 
def rand_inv_gamma(a, bb, config, n=1): 
    dim = config["dim"]
    l_sig2 = config["l_sig2"]
    u_sig2 = config["u_sig2"]

    if n == 1:
        Gamma = jnp.zeros(dim)
    else: 
        Gamma = jnp.zeros(n, dim)

    for p in range(dim):
        if n == 1 
            sig2 = stats.gamma(a, scale=1/bb[p])  
            # assert sig2 > 0 
            while sig2 < 1/u_sig2 || sig2 > 1/l_sig2:
                sig2 = stats.gamma(a, scale=1/bb[p]) 
            Λ[p] = 1 / sig2
        else:  
            for j in range(n):
                sig2 = stats.gamma(a, scale=1/bb[p])  
                # assert sig2 > 0
                while sig2 < 1/u_sig2 || sig2 > 1/l_sig2:
                    sig2 = stats.gamma(a, scale=1/bb[p])
                Λ[j, p] = 1 ./ sig2

    return Gamma 


def sample_repulsive_gauss(X, Mu, Sig, C, ell, config):
    min_d = 0.     # min wasserstein distance 
    g0 = config["g0"] 
    a0 = config["a0"] 
    b0 = config["b0"]
    tau = config["tau"]
    dim = config["dim"]

    K = size(Mu, 2)
    normal = stats.multivariate_normal(jnp.zeros(dim), tau**2)
    while rand() > min_d:   
        for k in range(ell, K): 
            Mu[k, :] = normal.rv()
            Sig[k, :, :] = rand_inv_gamma(a0, b0, config) 
        min_d = min_wass_distance(Mu, Sig, g0) 


def post_sample_repulsive_gauss(X, Mu, Sig, C, config): 
    min_d = 0.              # min wasserstein distance
    reject_counts = 0 

    g0 = config["g0"]
    while rand() > min_d:
        post_sample_gauss_kernels(X, Mu, Sig, C, config)
        reject_counts += 1 
        min_d = min_wass_distance(Mu, Sig, g0)
    return reject_counts 
 

def initialize(X, Mu, Sig, C, config) 
    min_d = 0.
    n = X.shape[0]

    g0 = config["g0"] 
    a0 = config["a0"] 
    b0 = config["b0"]
    tau = config["tau"]

    K, dim = Mu.shape   
    normal = stats.multivariate_normal(jnp.zeros(dim), tau**2)
    while random.random() > min_d:   
        for k in range(K):
            Mu[:, k] = normal.rv()
            Sig[:, :, k] = rand_inv_gamma(a0, b0, config)
        min_d = min_wass_distance(Mu, Sig, g0) 

    def normal_k(k, x):
        return multivariate_normal.logpdf(x, Mu[:, k], Sig[:, :, k])

    for i in range(n):
        # Make sure that the last cluster is not assigned anything
        C[i] = jnp.argmax(
            list(map(lambda k: normal_k(k, X[i, :]), range(K-1)))
        )
    end
end 
 