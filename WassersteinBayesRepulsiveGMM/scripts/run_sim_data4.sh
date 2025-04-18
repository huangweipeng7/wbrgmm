#!/bin/bash
set -e


data="sim_data_new_6"
n_burnin=2000
n_iter=2000
tau=100
g0=5
nu0=4
thinning=2

for method in "rgm-full" "wrgm-full" "dpgm-full"
do 
julia -t4 ./test/run.jl \
	--dataname $data \
	--method $method \
	--n_burnin $n_burnin \
	--n_iter $n_iter \
	--thinning $thinning \
	--tau $tau \
	--g0 $g0 \
	--nu0 $nu0  
done 
##################################################################################

for method in "rgm-full" "wrgm-full" "dpgm-full"
do 
	julia -t16 ./test/plot.jl --dataname $data --method $method  
done
##################################################################################

# julia -t16 ./test/plot.jl --dataname $data --method all --dist_type Mean
# julia -t16 ./test/plot.jl --dataname $data --method all --dist_type Wasserstein
