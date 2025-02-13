compute_log_prob(k::Int, n::Int) = 
	logfactorial(k) - logfactorial(k) - logfactorial(k+n)


log_p_K(K::Int, n::Int)::Vector{Float64} = compute_log_prob.(1:K, Ref(n))


function log_p_k_extend!(lp::Vector{Float64}, T::Int, n::Int)
	K = length(lp)
	@inbounds for t = 1:T   
		push!(lp, compute_log_prob(K+t, n))
	end
end