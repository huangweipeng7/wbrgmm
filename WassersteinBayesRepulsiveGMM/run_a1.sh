#!/bin/bash
set -e

data="a1"
n_burnin=10000
n_iter=5000
tau=1e5
g0=100
thinning=1


julia -t4 ./test/run.jl \
	--dataname $data \
	--method mean \
	--n_burnin $n_burnin \
	--n_iter $n_iter \
	--thinning $thinning \
	--tau $tau \
	--g0 $g0 \
	--nu0 4 

julia -t8 ./test/plot.jl --dataname $data --method mean  
#########################################


julia -t4 ./test/run.jl \
	--dataname $data \
	--method wasserstein \
	--n_burnin $n_burnin \
	--n_iter $n_iter \
	--thinning $thinning \
	--tau $tau \
	--g0 $g0 \
	--nu0 4 

julia -t8 ./test/plot.jl --dataname $data --method wasserstein  
########################################
 

julia -t8 ./test/plot.jl --dataname $data --method all 
