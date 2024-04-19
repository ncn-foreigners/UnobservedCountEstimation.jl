function binomial_model(m, N, n; start = "glm", iter = 2000, 
                        warm_up = floor(Int, iter / 2), grid,
                        save_simulation = true)
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

    start = [M]

    if start == "glm"
        append!(start, coef(glm(@formula(y ~ x1 + x2 + 0), df, Poisson(), LogLink())))
    else
        append!(start, coef(lm(@formula(log(y) ~ x1 + x2 + 0), df)))
    end # end
    
    res = gibbs_sampler_binomial_model(start, grid, iter, n, N, m)

    # return object with summary statistics and 
end # end function