#!/bin/bash
set -e

data="a1" 
n_burnin=10000
n_iter=5000 
tau=1e5
g0=100
nu0=4
thinning=1



# julia -t4 ./test/run.jl \
# 	--dataname $data \
# 	--method wrgm-diag \
# 	--n_burnin $n_burnin \
# 	--n_iter $n_iter \
# 	--thinning $thinning \
# 	--tau $tau \
# 	--g0 $g0 \
# 	--nu0 $nu0 
# #########################################


# julia -t4 ./test/run.jl \
# 	--dataname $data \
# 	--method brgm \
# 	--n_burnin $n_burnin \
# 	--n_iter $n_iter \
# 	--thinning $thinning \
# 	--tau $tau \
# 	--g0 $g0 \
# 	--nu0 $nu0
# ##########################################

julia -t4 ./test/run.jl \
	--dataname $data \
	--method mrgm \
	--n_burnin $n_burnin \
	--n_iter $n_iter \
	--thinning $thinning \
	--tau $tau \
	--g0 $g0 \
	--nu0 $nu0 
#########################################

julia -t4 ./test/run.jl \
	--dataname $data \
	--method wrgm \
	--n_burnin $n_burnin \
	--n_iter $n_iter \
	--thinning $thinning \
	--tau $tau \
	--g0 $g0 \
	--nu0 $nu0  
######################################


julia -t4 ./test/run.jl \
	--dataname $data \
	--method dpgm \
	--n_burnin $n_burnin \
	--n_iter $n_iter \
	--thinning $thinning \
	--tau $tau \
	--g0 $g0 \
	--nu0 $nu0  
#######################################


julia -t8 ./test/plot.jl --dataname $data --method wrgm-diag 

julia -t8 ./test/plot.jl --dataname $data --method brgm  

julia -t8 ./test/plot.jl --dataname $data --method mrgm  

julia -t8 ./test/plot.jl --dataname $data --method wrgm

julia -t8 ./test/plot.jl --dataname $data --method dpgm

julia -t8 ./test/plot.jl --dataname $data --method all 
