#!/bin/bash
set -e

data="sim_data3"

julia -t4 ./test/run.jl \
	--dataname $data \
	--method mean \
	--n_burnin 8000 \
	--n_iter 2000 \
	--thinning 5 \
	--tau 0.1 \
	--g0 5 \
	--nu0 5 

julia -t8 ./test/plot.jl --dataname $data --method mean  
#########################################


julia -t4 ./test/run.jl \
	--dataname $data \
	--method wasserstein \
	--n_burnin 8000 \
	--n_iter 2000 \
	--thinning 5 \
	--tau 0.1 \
	--g0 5 \
	--nu0 5 

julia -t8 ./test/plot.jl --dataname $data --method wasserstein  
########################################


julia -t4 ./test/run.jl \
	--dataname $data \
	--method no \
	--n_burnin 8000 \
	--n_iter 2000 \
	--thinning 5 \
	--tau 0.1 \
	--g0 0.1 \
	--nu0 5 

julia -t8 ./test/plot.jl --dataname $data --method no  
##########################################
