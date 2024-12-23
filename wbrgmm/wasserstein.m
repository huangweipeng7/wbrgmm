function w = numerical_ZK(mu1, Sig1, mu2, Sig2) 
    Sig2_sqrt = sqrtm(Sig2)
    d = sum(
        (mu1 - mu2).^2  
        + trace(Sig1 + Sig2 - Sig2_sqrt Sig1 Sig2_sqrt) 
    );
end