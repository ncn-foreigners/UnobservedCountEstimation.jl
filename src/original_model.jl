function zhang_model(m, N, n; start = "glm")
    df = DataFrame(
        y = m,
        x1 = log.(N),
        x2 = log.(n ./ N)
    )

    if start == "glm"
        start = coef(glm(@formula(y ~ x1 + x2 + 0), df, Poisson(), LogLink()))
    else
        start = coef(m(@formula(log(y) ~ x1 + x2 + 0), df))
    end # end if
    
    #= mm  = glm(@formula(y ~ x1 + x2 + 0), df, Poisson(), LogLink())
    ols =  lm(@formula(log(y) ~ x1 + x2 + 0), df) =#

    #[coef(ols), coef(mm)]
    start
end # end function