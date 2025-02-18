import jax.numpy as jnp 


def numerical_Zhat(Mu_mc, Sig_mc, g0, ell, t_max): 
    n_mc = Mu_mc.shape[0] 
    gg = jnp.zeros(n_mc, t_max)
 
    for n in range(n_mc): 
        for k in range(ell, (ell+t_max)):  
            min_d = 1.
            for i in range(k):
                for j in range(i:k-1):
                # println(Sig_mc[:, :, i, n],  Sig_mc[:, :, j, n])
                d = wass_gauss(
                    Mu_mc[:, i, n], Sig_mc[:, :, i, n], 
                    Mu_mc[:, j, n], Sig_mc[:, :, j, n]) 
                min_d = min(min_d, d/(g₀+d)) 
            gg[n, k-ell] = min_d  

    Zhat = jnp.mean(gg, axis=1).log()
    return Zhat 