import jax.numpy as jnp
from scipy.linalg import sqrtm


def wass_gauss(mu0, sig0, mu1, sig1):
	d = jnp.pow(mu0 - mu1, 2).sum() 
	d += 2 * jnp.pow(sqrtm(sig0) - sqrtm(sig1), 2).sum() 
	return sqrt(d)


def min_wass_distance(Mu, Sig, g0):
    K = Mu.shape[0]  
    min_d = 1.
    for i in range(K): 
    	for j in range(i, K-1):   
        	d = wass_gauss(
            	Mu[i, :], Sig[i, :, :], Mu[j, :], Sig[j, :, :]
            )  
        	min_d = min(min_d, d/(d+g₀))
    return min_d