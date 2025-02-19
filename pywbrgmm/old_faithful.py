import jax
import numpy as np
import pandas as pd 

from mcmc.blocked_collapse_gibbs import blocked_gibbs
 
np.random.seed(100)


def main():

	df = pd.read_csv('faithful_data.csv')
	X = df.values[:, 1:3]
	
	g0 = 10.
	beta = 1.
	tau = 0.1
	a0 = 1.
	b0 = 1.
	l_sig2 = 0.001
	u_sig2 = 1000.
	K = 5
 
	C_mc, Mu_mc, Sigma_mc, llhd_mc = blocked_gibbs(
		X, g0=g0, K=K, beta=beta, tau=tau, a0=a0, b0=b0, 
		l_sig2=l_sig2, u_sig2=u_sig2,
		burnin=5000, runs=7500, thinning=10) 

	println(C_mc[end])
	println(countmap(C_mc[end]))


if __name__ == '__main__':
	main()