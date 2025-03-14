#!/bin/bash
set -e

julia -t4 ./test/run.jl \
	--dataname sim_data1 \
	--method mean \
	--n_burnin 4000 \
	--n_iter 8000 \
	--thinning 5 \
	--tau 0.1 \
	--g0 0.1 \
	--nu0 5 

julia -t8 ./test/plot.jl --dataname sim_data1 --method mean  
#########################################


# julia -t4 ./test/run.jl \
# 	--dataname sim_data1 \
# 	--method wasserstein \
# 	--n_burnin 4000 \
# 	--n_iter 8000 \
# 	--thinning 5 \
# 	--tau 0.001 \
# 	--g0 0.1 \
# 	--nu0 5 

# julia -t8 ./test/plot.jl --dataname sim_data1 --method wasserstein  
# ########################################


# julia -t4 ./test/run.jl \
# 	--dataname sim_data1 \
# 	--method no \
# 	--n_burnin 4000 \
# 	--n_iter 8000 \
# 	--thinning 5 \
# 	--tau 0.001 \
# 	--g0 0.1 \
# 	--nu0 5 

# julia -t8 ./test/plot.jl --dataname sim_data1 --method no  
# ##########################################
