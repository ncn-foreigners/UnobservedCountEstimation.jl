function zhang_model(m, N, n; start = "glm")
    # TODO:: add X, Z arguments and then methods for type X/Z nothing or formula
    df = DataFrame(
        y = m,
        x1 = log.(N),
        x2 = log.(n ./ N)
    )

    # TODO :: dependent α, β
    if start == "glm"
        start = coef(glm(@formula(y ~ x1 + x2 + 0), df, Poisson(), LogLink()))
    else
        start = coef(lm(@formula(log(y) ~ x1 + x2 + 0), df))
    end # end if
    
    #= mm  = glm(@formula(y ~ x1 + x2 + 0), df, Poisson(), LogLink())
    ols =  lm(@formula(log(y) ~ x1 + x2 + 0), df) =#
    X = ones(length(n))
    X = X[:, :]
    
    Z = ones(length(n))
    Z = Z[:, :]

    log_l_f  = x -> log_lik_original_model(x[1], x[2], x[3], m, N, n, X, Z) * (-1)
    grad_l_f = x -> grad_log_lik_original_model(x[1], x[2], x[3], m, N, n, X, Z) * (-1)
    hes_l_f  = x -> hess_log_lik_original_model(x[1], x[2], x[3], m, N, n, X, Z) * (-1)

    #= result   = optimize(log_l_f, [start[1], start[2], 1], Newton(); inplace = false)
    result_1 = optimize(log_l_f, grad_l_f, [start[1], start[2], 1], Newton(); inplace = false) =#
    # TODO :: dependent α, β
    optim_problem = optimize(log_l_f, grad_l_f, hes_l_f, [start[1], start[2], 0], NewtonTrustRegion(); inplace = false)
    α̂ = optim_problem.minimizer[1]
    β̂ = optim_problem.minimizer[2]
    ϕ̂ = optim_problem.minimizer[3]
    ξ̂ = sum(N .^ α̂)
    #[coef(ols), coef(mm)]
    #[start, log_l, grad_l, hes_l]
    [[α̂, β̂, ϕ̂, ξ̂], optim_problem]
end # end function