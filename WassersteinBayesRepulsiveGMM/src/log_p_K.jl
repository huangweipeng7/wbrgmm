compute_log_prob(k::Int, ℓ::Int, n::Int) = 
	logfactorial(k) - logfactorial(k-ℓ) - logfactorial(k+n)


log_p_K(K::Int, ℓ::Int, n::Int) = compute_log_prob.(ℓ:K, Ref(ℓ), Ref(n))


function log_p_k_extend!(lp::Vector{Float64}, T::Int, n::Int)
	K = length(lp)
	@inbounds for t = 1:T   
		push!(lp, compute_log_prob(K+t, n))
	end
end