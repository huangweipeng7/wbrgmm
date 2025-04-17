for data in "a1" "GvHD" "sim_data_new_1" "sim_data_new_2" "sim_data_new_3" "sim_data_new_4"
do 
	for method in "rgm-full" "rgm-diag" "wrgm-full" "wrgm-diag" "dpgm-full" "dpgm-diag" 
	do 
		julia -t8 ./test/plot.jl --dataname $data --method $method  
	done
	##################################################################################

	julia -t8 ./test/plot.jl --dataname $data --method all --dist_type Mean
	julia -t8 ./test/plot.jl --dataname $data --method all --dist_type Wasserstein
done 