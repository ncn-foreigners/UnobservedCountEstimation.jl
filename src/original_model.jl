placeholder(x) = x

function zhang_model(m, N, n)
    df = DataFrame(
        y = m,
        x1 = log.(N),
        x2 = log.(n ./ N)
    )

    mm  = glm(@formula(y ~ x1 + x2 + 0), df, Poisson(), LogLink())
    ols =  lm(@formula(log(y) ~ x1 + x2 + 0), df)

    [coef(ols), coef(mm)]
end # end function