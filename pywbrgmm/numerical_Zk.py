import jax 
import jax.numpy as jnp
import numpy as np 
import scipy.stats as stats

from mcmc.sampler import rand_inv_gamma
from measure.wasserstein import wass_gauss


def numerical_Zk(K_max, dim, config, n_mc=100):
    ''' Numerical Computation of ZK as in algorithm 1
    '''
    g0 = config["g0"]
    a0 = config["a0"]
    b0 = config["b0"]
    tau = config["tau"]
    
    mu_mc = jnp.zeros((n_mc, K_max, dim))
    sig_mc = jnp.zeros((n_mc, K_max, dim, dim))

    mu_mc = stats.multivariate_normal.rvs(
        jnp.zeros(dim), tau**2 * jnp.eye(dim), (n_mc, K_max)
    ) 
    sig_mc = np.zeros((n_mc, K_max, dim, dim))
    for n in range(n_mc):
        for k in range(K_max):
            np.fill_diagonal(sig_mc[n, k], rand_inv_gamma(a0, b0, config))
    sig_mc = jnp.array(sig_mc) 

    gg = jnp.zeros((n_mc, K_max)) 
    for n in range(n_mc): 
        for k in range(K_max): 
            min_d = 1.
            for i in range(k):
                for j in range(i, k): 
                    d = wass_gauss(
                        mu_mc[n, i], sig_mc[n, i], mu_mc[n, j], sig_mc[n, j]
                    ) 
                    min_d = min(min_d, d/(g0+d))
            gg.at[n, k].set(min_d)  
     
    Z = jnp.log(jnp.mean(gg, axis=0))
    Z.at[0].set(0.)
    return Z
