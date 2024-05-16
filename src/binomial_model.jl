function binomial_model(m, N, n; start = "glm", iter = 2000, 
                        warm_up = 1, grid,
                        save_simulation = true,
                        k_prior = :auto, theta_prior = :auto, 
                        sigma_prior = :auto, rand_eff = false,
                        u_method = "exact")
    # TODO:: add X, Z arguments and then methods for type X/Z nothing or formula
    df = DataFrame(
        y = m,
        x1 = log.(N),
        x2 = log.(n ./ N)
    )
    # TODO:: check length of n, m, N
    Q = length(n)

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
    if sigma_prior == :auto
        sigma_prior = vcov(mm)
    end
    
    if theta_prior == :auto
        theta_prior = diag(sigma_prior) ./ start[(end-1):end]
    end

    if k_prior == :auto
        k_prior = start[(end-1):end] ./ θ_prior
    end

    # TODO warnings if bad prior
    res = nothing
    if !rand_eff
        res = gibbs_sampler_binomial_model(start, grid, iter, n, N, m, k_prior, theta_prior, sigma_prior)
    else
        push!(start, ones(Float64, length(m)))
        res = gibbs_sampler_binomial_model_random_eff(start, grid, iter, n, N, m, k_prior, theta_prior, sigma_prior, u_method)
    end # end if 

    #M  = start[1]
    #γ₁ = start[2]
    #γ₂ = start[3]
    #u  = start[4]


     # TODO:: Add standard errors and add code to compute standard errors of medians via:
     # Var(median) = (4 * n * dP/dλ(median) ^ 2)
     # and standard errors of means
    coefs = Dict(
        # push conditiona; expected value estimates
        "Mean" => reduce(vcat,
            [[mean(res[k][warm_up:end]) for k in 1:Q], mean(res[Q+1][warm_up:end]), mean(res[Q+2][warm_up:end])]
        ),
        # push maximum a posteriori estimates
        "MAP" => reduce(vcat,
           [[median(res[k][warm_up:end]) for k in 1:Q], median(res[Q+1][warm_up:end]), median(res[Q+2][warm_up:end])]
        )
    )

    if rand_eff
        # append estimates for u
        append!(coefs["Mean"], [  mean(res[Q+2+k][warm_up:end]) for k in 1:Q])
        append!(coefs[ "MAP"], [median(res[Q+2+k][warm_up:end]) for k in 1:Q])
    end # end if
    # return object with summary statistics and

    BayesianUnobservedCountModel(
        res, # sim_res
        [m, n, N], # data
        coefs, # coefs
        [sigma_prior, k_prior, theta_prior], #prior
        iter, # iter
        nrow(df) # Q
    )
end # end function