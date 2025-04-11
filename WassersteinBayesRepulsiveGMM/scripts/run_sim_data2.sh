#!/bin/bash
set -e


data="sim_data_new_4"
n_burnin=10000
n_iter=5000
tau=100
g0=5
nu0=4


julia -t8 ./test/run.jl \
	--dataname $data \
	--method brgm \
	--n_burnin $n_burnin \
	--n_iter $n_iter \
	--thinning 1 \
	--tau $tau \
	--g0 $g0 \
	--nu0 $nu0
julia -t8 ./test/plot.jl --dataname $data --method brgm  


julia -t4 ./test/run.jl \
	--dataname $data \
	--method mean \
	--n_burnin $n_burnin \
	--n_iter $n_iter \
	--thinning 1 \
	--tau $tau \
	--g0 $g0 \
	--nu0 $nu0

# julia -t8 ./test/plot.jl --dataname $data --method mean  
# # # #########################################


julia -t4 ./test/run.jl \
	--dataname $data \
	--method wasserstein \
	--n_burnin $n_burnin \
	--n_iter $n_iter \
	--thinning 1 \
	--tau $tau \
	--g0 $g0 \
	--nu0 $nu0

julia -t8 ./test/plot.jl --dataname $data --method mean  
julia -t8 ./test/plot.jl --dataname $data --method wasserstein  
julia -t8 ./test/plot.jl --dataname $data --method all
########################################	


# julia -t4 ./test/run.jl \
# 	--dataname $data \
# 	--method no \
# 	--n_burnin 8000 \
# 	--n_iter 2000 \
# 	--thinning 5 \
# 	--tau 10 \
# 	--g0 1 \
# 	--nu0 5 

# julia -t8 ./test/plot.jl --dataname $data --method no  
# # ##########################################
