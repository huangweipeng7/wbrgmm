import jax.numpy as jnp 
import math 

from jax.lax import lgamma 


def logV_nt(n, t_max):
    logV = jnp.zeros(t_max)
    tol = 1e-6

    log_exp_m_1 = math.log(math.e - 1)
    for t in range(t_max):
        logV.at[t].set(-math.inf)
        if t <= n - 1: 
            a, c, k, p = 0, -math.inf, 1, 0
            while math.fabs(a - c) > tol or p < 1. - tol:
                # Note: The first condition is false when a = c = -Inf
                if k >= t + 1:
                    a = c 
                    b = - lgamma(k-t+1.) - lgamma(float(k+n)) + lgamma(float(k)) - log_exp_m_1  
                    m = max(a, b)
                    if m == -math.inf:
                        c = -math.inf  
                    else: 
                        c = m + math.log(math.exp(a - m) + math.exp(b - m))

                p += math.exp(-log_exp_m_1 - lgamma(k+1.))
                k += 1 

            logV.at[t].set(c)

    return logV
