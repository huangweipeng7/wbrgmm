import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
import scipy.stats as stats
import math
import numpy as np
import random
import tqdm 
from collections import Counter
from time import time 

from measure.wasserstein import min_wass_distance
from .sampler import rand_inv_gamma
from logV import logV_nt
from numerical_Zk import numerical_Zk
from numerical_Zhat import numerical_Zhat
from log_p_K import log_prob_K


def blocked_gibbs(
    X: jax.Array, 
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

    C_mc = []
    Mu_mc = []
    Sig_mc = []
    llhd_mc = []
  
    C = np.zeros(n, dtype=np.int32) 
    Mu = np.zeros((K+1, dim), dtype=jnp.float32)
    Sig = np.zeros((K+1, dim, dim), dtype=jnp.float32)
    initialize(X, Mu, Sig, C, config)       # in place

    logV = logV_nt(n, t_max)
    Zk = numerical_Zk(2*K, dim, config)
    
    pbar = tqdm.trange(burnin+runs)
    for iter in pbar:
        C, Mu, Sig, llhd = post_sample_C(
            X, Mu, Sig, C, logV, Zₖ, config)

        C, Mu_, Sig_, _ = post_sample_K_and_swap_indices(
            n, C, Mu, Sig, t_max)
        
        Mu, Sig = Mu_, Sig_  
        post_sample_repulsive_gauss(X, Mu, Sig, C, config)

        pbar.set_description(f"loglikelihood: {llhd:.3f}")
        
        if iter > burnin and iter % thinning == 0: 
            C_mc.append(deepcopy(C))
            Mu_mc.append(deepcopy(Mu))  
            Sig_mc.append(deepcopy(Sig))
            llhd_mc.append(llhd)
    return C_mc, Mu_mc, Sig_mc, llhd_mc
    

def post_sample_C(X, Mu, Sig, C, logV, Zk, config):  
    key = jrandom.PRNGKey(random.randint(0, 1000_000))

    t_max = config["t_max"]
    beta = config["beta"]  

    n = X.shape[0]
    K = Mu.shape[0]     
 
    lp = []
    lc = []
    llhd = 0.                       # Log likelihood
    for i in range(n):  
        x = X[i] 

        C_, Mu_, Sig_, ell = sample_K_and_swap_indices(
            i, n, C, Mu, Sig, t_max, exclude_i=True)
        C_[i] = -1       # pseudo component assignment
        assert jnp.max(C_) < ell

        # K plus 1
        Kp1 = Mu_.shape[0]
        sample_repulsive_gauss(X, Mu_, Sig_, C, ell, config)
        # print(Mu_, 'after')

        for k in range(Kp1):
            assert jnp.all(jnp.diagonal(Sig_[k]) > 0), print(Sig_[k])  
        
        # resize the vectors for loglikelihood of data and the corresponding coefficients
        lp = []
        lc = [] 
        for k in range(Kp1): 
            n_k = sum(C == k) - (C[i] == k)  
            lp.append(
                stats.multivariate_normal.logpdf(x, Mu_[k, :], Sig_[k, :, :])
            )
            lc.append(
                math.log(n_k+beta) if k != Kp1 else 
                math.log(beta) + logV[ell+1] - logV[ell]
            ) 
        
        lp = jnp.array(lp)
        lc = jnp.array(lc)
        Mu, Sig = Mu_, Sig_ 
        C_[i] = jrandom.categorical(key, lp + lc) 
        llhd += lp[C[i]] 
    return C_, Mu, Sig, llhd


def sample_K_and_swap_indices(
    i, n, C, Mu, Sig, t_max, 
    exclude_i=False, approx=True, X=None, Zk=None
): 

    dim = Mu.shape[1]
    C_ = np.zeros_like(C)
    inds = jnp.arange(n)

    if exclude_i:
        cluster_count = Counter(C[inds!=i]) 
    else: 
        cluster_count = Counter(C)  
    nz_k_inds = list(map(int, cluster_count.keys())) 
    ell = len(nz_k_inds) 

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
    Mu_ = np.zeros((K+1, dim))
    Sig_ = np.zeros((K+1, dim, dim))
 
    # print(K, nz_k_inds, ell, " exclude_i? ", exclude_i, cluster_count, C)
 
    Mu_[0:ell] = Mu[nz_k_inds]
    Sig_[0:ell] = Sig[nz_k_inds]  

    for i_nz, nz_k_ind in enumerate(nz_k_inds):
        C_[C == nz_k_ind] = i_nz
    # print(Mu_, 'before', jnp.max(C_).item())
    assert jnp.max(C_).item() < ell 

    return C_, Mu_, Sig_, ell   


def post_sample_K_and_swap_indices(n, C, Mu, Sig, t_max):
    return sample_K_and_swap_indices(
        None, n, C, Mu, Sig, t_max, exclude_i=False
    )


def sample_K(log_p_K, ell, t_max, n):
    key = jrandom.PRNGKey(random.randint(0, 1000_000))
    return ell + jrandom.categorical(key, log_p_K) 


def post_sample_K(log_p_K, Zhat, Zk, ell, t_max):  
    return ell + gumbel_max_sample(log_p_K + Zhat - Zk[ell-1:ell+t_max-2]) - 2


def post_sample_gauss_kernels(X, Mu, Sig, C, config): 
    g0 = config["g0"] 
    a0 = config["a0"] 
    b0 = config["b0"]
    tau = config["tau"] 

    K, dim = Mu.shape  
    
    tau_sq = tau ** 2
    # print(Sig)
    for k in range(K):
        key = jrandom.PRNGKey(random.randint(k, 1000_000))

        X_tmp = X[C==k] 
        n = X_tmp.shape[0]
        
        tau_sq = tau ** 2
        if n == 0: 
            Mu[k] = jrandom.multivariate_normal(
                key, jnp.zeros(dim), tau_sq * jnp.eye(dim)
            )  
            np.fill_diagonal(Sig[k], rand_inv_gamma(a0, b0, config))
        else:  
            x_sum = X_tmp.sum(axis=0)  
            sig0 = jnp.linalg.inv(
                tau_sq * jnp.eye(dim) + n * jnp.linalg.inv(Sig[k])
            )
            mu0 = sig0 @ (jnp.linalg.inv(Sig[k, :, :]) @ x_sum.T) 
            Mu[k] = stats.multivariate_normal.rvs(mu0, Sig[k])

            a_k = a0 + n / 2.
            b_k = b0 + jnp.sum(jnp.pow((X_tmp - Mu[k]), 2), axis=0) / 2. 
            # if k == K-1:
            #     print(C)
            #     print(b0, b_k, Mu[k], mu0, Sig[k], n, k, X_tmp)
            np.fill_diagonal(Sig[k], rand_inv_gamma(a_k, b_k, config))

        assert np.all(~np.isnan(Mu[k])) and np.all(~np.isnan(Sig[k]))
            

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


def sample_repulsive_gauss(X, Mu, Sig, C, ell, config):
    min_d = 0.     # min wasserstein distance 
    g0 = config["g0"] 
    a0 = config["a0"] 
    b0 = config["b0"]
    tau = config["tau"]
    dim = config["dim"]

    K = Mu.shape[0]
    while random.random() > min_d:   
        for k in range(ell, K):  
            Mu[k] = stats.multivariate_normal.rvs(
                jnp.zeros(dim), tau**2 * jnp.eye(dim)
            )
            assert np.all(~np.isnan(Mu[k])) and np.all(~np.isnan(Sig[k]))
            np.fill_diagonal(Sig[k], rand_inv_gamma(a0, b0, config)) 
        min_d = min_wass_distance(Mu, Sig, g0)  



def post_sample_repulsive_gauss(X, Mu, Sig, C, config): 
    min_d = 0.              # min wasserstein distance
    reject_counts = 0 

    g0 = config["g0"]
    while random.random() > min_d:
        post_sample_gauss_kernels(X, Mu, Sig, C, config)
        reject_counts += 1 
        min_d = min_wass_distance(Mu, Sig, g0)
    return reject_counts 
 

def initialize(X, Mu, Sig, C, config):
    def normal_k(k, x):
        return jstats.multivariate_normal.logpdf(x, Mu[k], Sig[k])

    min_d = 0.
    n = X.shape[0]

    g0 = config["g0"] 
    a0 = config["a0"] 
    b0 = config["b0"]
    tau = config["tau"] 

    K, dim = Mu.shape    
    while random.random() > min_d:   
        for k in range(K):
            Mu[k] = stats.multivariate_normal.rvs(
                jnp.zeros(dim), jnp.eye(dim)*tau**2
            )
            np.fill_diagonal(Sig[k], rand_inv_gamma(a0, b0, config))

            assert np.all(~np.isnan(Mu[k])) and np.all(~np.isnan(Sig[k]))
        
        min_d = min_wass_distance(Mu, Sig, g0) 

    for i in range(n):
        # Make sure that the last cluster is not assigned anything
        C[i] = jnp.argmax(
            jnp.array([normal_k(k, X[i]) for k in range(K-1)])
        ).item()

 