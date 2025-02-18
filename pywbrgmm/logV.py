import jax.lax.lgamma as lgamma 
import jax.numpy as jnp 
import math 


def logV_nt(n, t_max):
    logV = jnp.zeros(t_max)
    tol = 1e-12 

    log_exp_m_1 = math.log(math.e - 1)
    for t in range(t_max):
        logV[t] = -math.inf
        if t <= n - 1: 
            a, c, k, p = 0, -math.inf, 1, 0
            while math.abs(a - c) > tol or p < 1. - tol:
                # Note: The first condition is false when a = c = -Inf
                if k >= t
                    a = c 
                    b = - lgamma(k-t+1.) - lgamma(k+n) + lgamma(float(k)) - log_exp_m_1  
                    m = max(a, b);
                    if m == -math.inf:
                        c = -math.inf  
                    else: 
                        m + math.log(math.exp(a - m) + math.exp(b - m))

                p += exp(-log_exp_m_1 - lgamma(k+1.))
                k += 1 

            logV[t] = c

    return logV
