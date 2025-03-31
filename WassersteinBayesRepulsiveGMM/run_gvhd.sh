#!/bin/bash
set -e

data="GvHD" 
n_burnin=10000
n_iter=5000 
tau=1e4
g0=20
nu0=4
thinning=1 


julia -t4 ./test/run.jl \
	--dataname $data \
	--method brgm \
	--n_burnin $n_burnin \
	--n_iter $n_iter \
	--thinning $thinning \
	--tau $tau \
	--g0 $g0 \
	--nu0 $nu0
##########################################

julia -t4 ./test/run.jl \
	--dataname $data \
	--method mean \
	--n_burnin $n_burnin \
	--n_iter $n_iter \
	--thinning $thinning \
	--tau $tau \
	--g0 $g0 \
	--nu0 $nu0 
#########################################

julia -t4 ./test/run.jl \
	--dataname $data \
	--method wasserstein \
	--n_burnin $n_burnin \
	--n_iter $n_iter \
	--thinning $thinning \
	--tau $tau \
	--g0 $g0 \
	--nu0 $nu0  
#######################################


julia -t4 ./test/run.jl \
	--dataname $data \
	--method no \
	--n_burnin $n_burnin \
	--n_iter $n_iter \
	--thinning $thinning \
	--tau $tau \
	--g0 $g0 \
	--nu0 $nu0  
#######################################


julia -t8 ./test/plot.jl --dataname $data --method brgm  

julia -t8 ./test/plot.jl --dataname $data --method mean  

julia -t8 ./test/plot.jl --dataname $data --method wasserstein

julia -t8 ./test/plot.jl --dataname $data --method no

julia -t8 ./test/plot.jl --dataname $data --method all 
