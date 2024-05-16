function posterior_data_generation(object::ZhangUnobservedCountModel)
   "foo" 
end


# MAP or Mean
function posterior_data_generation(object::BayesianUnobservedCountModel; est = "MAP", rep = 500)
   par = object.coefs[est]
   Q = length.(object.:data)[1]
   
   M  = par[1:Q]
   γ₁ = par[end - 1]
   γ₂ = par[end]
   u  = 1 
   if length(par) != Q + 2
     u  = par[(Q + 1):(end - 2)]
   end # end if
   μ  = (object.data[3] .^ γ₁) .* ((object.data[2] ./ object.data[3]) .^ γ₂)
   μ  = 1 ./ (1 .+ 1 ./ μ)

   rand.(Binomial.(M, u .* μ), rep)
end