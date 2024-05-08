function binomial_model(m, N, n; start = "glm", iter = 2000, 
                        warm_up = floor(Int, iter / 2), grid,
                        save_simulation = true,
                        k_prior = :auto, θ_prior = :auto, 
                        Σ_prior = :auto, rand_eff = false)
    # TODO:: add X, Z arguments and then methods for type X/Z nothing or formula
    df = DataFrame(
        y = m,
        x1 = log.(N),
        x2 = log.(n ./ N)
    )

    X = ones(length(n))
    X = X[:, :]
    
    Z = ones(length(n))
    Z = Z[:, :]

    start = Vector{Any}([m])
    mm = nothing

    if start == "glm"
        mm = glm(@formula(y ~ x1 + x2 + 0), df, Poisson(), LogLink())
        append!(start, coef(mm))
    else
        mm = lm(@formula(log(y) ~ x1 + x2 + 0), df)
        append!(start, coef(mm))
    end # end

    # Asigning priors
    if Σ_prior == :auto
        Σ_prior = vcov(mm)
    end
    
    if θ_prior == :auto
        θ_prior = diag(Σ_prior) ./ start[(end-1):end]
    end

    if k_prior == :auto
        k_prior = start[(end-1):end] ./ θ_prior
    end

    # TODO warnings if bad prior
    res = nothing
    if !rand_eff
        res = gibbs_sampler_binomial_model(start, grid, iter, n, N, m, k_prior, θ_prior, Σ_prior)
    else
        push!(start, ones(Float64, length(m)))
        res = gibbs_sampler_binomial_model_random_eff(start, grid, iter, n, N, m, k_prior, θ_prior, Σ_prior)
    end # end if 


    # return object with summary statistics and
    res
end # end function