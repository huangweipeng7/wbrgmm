import jax.numpy as jnp
from scipy.linalg import sqrtm


def wass_gauss(mu0, sig0, mu1, sig1):
    d = jnp.sum(jnp.pow(mu0 - mu1, 2)) 
    d += 2 * jnp.sum(jnp.pow(sqrtm(sig0) - sqrtm(sig1), 2)) 
    # print(d, mu0, mu1, sig0, sig1)
    return jnp.sqrt(d)


def min_wass_distance(Mu, Sig, g0):
    K = Mu.shape[0]  
    min_d = 1.
    for i in range(K-1): 
        for j in range(i+1, K):   
            d = wass_gauss(Mu[i], Sig[i], Mu[j], Sig[j])  
            min_d = min(min_d, d/(d+g0))
    return min_d