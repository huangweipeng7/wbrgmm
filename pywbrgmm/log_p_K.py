import jax.numpy as jnp 
from jax.lax import lgamma 


def log_prob_wrapper(k, ell, n): 
	return lgamma(k+1.) - lgamma(k-ell+1.) - lgamma(k+n+1.)


def log_prob_K(ell, t_max, n): 
	lp = [log_prob_wrapper(j, ell, n) for j in range(ell, ell+t_max)]
	return jnp.array(lp) 
