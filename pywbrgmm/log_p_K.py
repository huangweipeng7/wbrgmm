import jax.numpy as jnp 
import jax.lax.lgamma as lgamma 

 
def log_prob_K(ell, t_max, n): 
	def log_prob_(k, ell, n): 
		return lgamma(k+1.) - lgamma(k-ell+1.) - lgamma(k+n+1.)
	lp = [log_prob_(j, ell, n) for j in range(ell, ell+t_max)]
	return jnp.array(lp) 
