function zhang_model(m, N, n; start = "glm")
    df = DataFrame(
        y = m,
        x1 = log.(N),
        x2 = log.(n ./ N)
    )

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

    log_l = log_lik_original_model(start[1], start[2], 1, m, N, n, X, Z)

    grad_l = grad_log_lik_original_model(start[1], start[2], 1, m, N, n, X, Z)

    hes_l = hess_log_lik_original_model(start[1], start[2], 1, m, N, n, X, Z)

    #[coef(ols), coef(mm)]
    [start, log_l, grad_l, hes_l]
end # end function