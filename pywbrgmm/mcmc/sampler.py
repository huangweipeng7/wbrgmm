import jax.numpy as jnp
import numpy as np 
import scipy.stats as stats


def rand_inv_gamma_vec(a, bb, config, n=1): 
    dim = config["dim"]
    l_sig2 = config["l_sig2"]
    u_sig2 = config["u_sig2"]

    if n == 1:
        Gamma = np.zeros(dim)
    else: 
        Gamma = np.zeros(n, dim)

    for p in range(dim):
        if n == 1:
            sig2 = stats.gamma.rvs(a, scale=1/bb[p])  
            # assert sig2 > 0 
            while sig2 < 1/u_sig2 or sig2 > 1/l_sig2:
                sig2 = stats.gamma.rvs(a, scale=1/bb[p]) 
            Gamma[p] = 1 / sig2
        else:  
            for j in range(n):
                sig2 = stats.gamma.rvs(a, scale=1/bb[p])  
                # assert sig2 > 0
                while sig2 < 1/u_sig2 or sig2 > 1/l_sig2:
                    sig2 = stats.gamma.rvs(a, scale=1/bb[p])
                Gamma[j, p] = 1 / sig2

    return Gamma 


def rand_inv_gamma(a, b, config, n=1): 
    dim = config["dim"]
    return rand_inv_gamma_vec(a, jnp.repeat(b, dim), config, n=n)
  
 
