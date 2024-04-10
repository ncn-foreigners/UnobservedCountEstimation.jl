function binomial_model(m, N, n; start = "glm")
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

    log_l_f  = x -> log_lik_binomial_model(x[1:(end - 2)], x[end - 1], x[end], m, N, n, X, Z) * (-1.0)
    grad_l_f = x -> grad_log_lik_binomial_model(x[1:(end - 2)], x[end - 1], x[end], m, N, n, X, Z) * (-1.0)
    hes_l_f  = x -> hess_log_lik_binomial_model(x[1:(end - 2)], x[end - 1], x[end], m, N, n, X, Z) * (-1.0)

    #= result   = optimize(log_l_f, [start[1], start[2], 1], Newton(); inplace = false)
    result_1 = optimize(log_l_f, grad_l_f, [start[1], start[2], 1], Newton(); inplace = false) =#
    # TODO :: dependent α, β

    start = Float64[]
    append!(start, zeros(length(N)))

    if start == "glm"
        append!(start, coef(glm(@formula(y ~ x1 + x2 + 0), df, Poisson(), LogLink())))
    else
        append!(start, coef(lm(@formula(log(y) ~ x1 + x2 + 0), df)))
    end # end if

    optim_problem = optimize(log_l_f, grad_l_f, hes_l_f, start, NewtonTrustRegion(); inplace = false)
    α̂ = optim_problem.minimizer[end - 1]
    β̂ = optim_problem.minimizer[end]
    M̂ = optim_problem.minimizer[1:length(N)]
    ξ̂ = N .^ α̂
    #[coef(ols), coef(mm)]
    #[start, log_l, grad_l, hes_l]
    [[α̂, β̂, ξ̂, M̂, sum(ξ̂), sum(M̂)], optim_problem]
end # end function