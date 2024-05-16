# logical matrix for if country has strata
function sample_gamma_1_cond_random_eff(grid, n, N, γ₂, M, m, u, k_prior, θ_prior, Σ_prior)
    distr_prior_marginal = Gamma(k_prior[1], θ_prior[1])
    copula = GaussianCopula(Σ_prior)

    # get unscaled density function and evaluate it on a grid
    function density_function(x)
        μ = (N .^ x) .* ((n ./ N) .^ γ₂)
        μ = 1 ./ (1 .+ 1 ./ μ)
        lξ = x .* log.(N)
        c = gamma_inc.(k_prior, [x, γ₂] ./ θ_prior)
        c = [c[ii][1] for ii in eachindex(c)]
        
        #res = (μ .^ m) .* ((1 .- μ) .^ (M .- m))
        res = BigFloat.(m .* log.(μ) + (M - m) .* log.(1 .- u .* μ) - exp.(lξ) + M .* lξ)
        exp(sum(res)) * pdf(copula, c) * pdf(distr_prior_marginal, x)
    end # end funciton

    evaluated_denisty = density_function.(grid)    
    # temporary solution
    evaluated_denisty[isnan.(evaluated_denisty)] .= 0
    evaluated_denisty ./= sum(evaluated_denisty)

    # sample acording to evaluation
    grid[rand(Categorical(evaluated_denisty))]
end # end funciton

function sample_gamma_2_cond_random_eff(grid, n, N, γ₁, M, m, u, k_prior, θ_prior, Σ_prior)
    distr_prior_marginal = Gamma(k_prior[2], θ_prior[2])
    copula = GaussianCopula(Σ_prior)
    # get unscaled density function and evaluate it on a grid
    function density_function(x)
        μ = (N .^ γ₁) .* ((n ./ N) .^ x)
        μ = 1 ./ (1 .+ 1 ./ μ)
        #ξ = N .^ γ₁
        c = gamma_inc.(k_prior, [γ₁, x] ./ θ_prior)
        c = [c[ii][1] for ii in eachindex(c)]

        #res = (μ .^ m) .* ((1 .- μ) .^ (M .- m))
        res = BigFloat.(m .* log.(μ) + (M - m) .* log.(1 .- u .* μ))
        exp(sum(res)) * pdf(copula, c) * pdf(distr_prior_marginal, x)
    end # end funciton

    evaluated_denisty = density_function.(grid)
    # temporary solution
    evaluated_denisty[isnan.(evaluated_denisty)] .= 0
    evaluated_denisty ./= sum(evaluated_denisty)
    # sample acording to evaluation
    grid[rand(Categorical(evaluated_denisty))]
end # end funciton

function sample_M_cond_random_eff(n, N, m, γ₁, γ₂, u)
    # compute ξ, μ
    μ = (N .^ γ₁) .* ((n ./ N) .^ γ₂)
    μ = 1 ./ (1 .+ 1 ./ μ)
    ξ = N .^ γ₁
    # draw M-m vector from poisson intependently
    M_minus_m = reduce(vcat, rand.(Poisson.(ξ .* (1 .- u .* μ)), 1))
    # return M = (M-m) + increment
    m + M_minus_m
end # end funciton

function sample_u_cond_random_eff(n, N, m, M, γ₁, γ₂; prec = 40, method)
    # compute normalization factor
    μ = (N .^ γ₁) .* ((n ./ N) .^ γ₂)
    μ = 1 ./ (1 .+ 1 ./ μ)
    beta_distr = Beta.(n, N - n)
    U = rand(Uniform(), length(N))

    res = Vector{Float64}()
    
    if method == "exact"
        for k in eachindex(M)
            setprecision(prec) do
                # TODO:: this could be much faster if broadcasted but GaussLegendre() doesn't work in R^d where d=>2
                f(x, p) = exp(BigFloat(m[k] * log(x) + (M - m)[k] * log(1 - x * μ[k]) + logpdf(beta_distr[k], x)))
                # this is deprecated
                # prob(x) = IntegralProblem(f, (0, x))
                # Pkg.status("Integrals")
                # for older versions:
                prob(x) = IntegralProblem(f, 0, x)
                ff(x)   = solve(prob(x), GaussLegendre(), reltol = 1e-10, abstol = 1e-10)
                R = ff(1)
                a = optimize(x -> (ff(x)[1] / R[1] - U[k]) ^ 2, 0, 1, Brent(), rel_tol = 1e-10)
                push!(res, a.minimizer)
            end # end set precision
        end # end for
    else 
        grid = 0.001:0.001:(1-0.001)
        f(x, k) = exp(BigFloat(m[k] * log(x) + (M - m)[k] * log(1 - x * μ[k]) + logpdf(beta_distr[k], x)))
        for k in eachindex(M)
            evaluated_denisty = f.(grid, k)
            evaluated_denisty[isnan.(evaluated_denisty)] .= 0
            evaluated_denisty ./= sum(evaluated_denisty)

            push!(res, grid[rand(Categorical(evaluated_denisty))])
        end # ned for
    end # end if
    res
end # end funciton

function gibbs_sampler_binomial_model_random_eff(start, grid, iter, n, N, m, k_prior, θ_prior, Σ_prior, u_method)
    # create storage vectors
    M  = start[1]
    γ₁ = start[2]
    γ₂ = start[3]
    u  = start[4]

    storage = reduce(vcat, [[[M[k]] for k in eachindex(M)], [[u[k]] for k in eachindex(M)], [[γ₁]], [[γ₂]]])
    prog = Progress(iter; desc = "Sampling progress ...")

    for _ in 1:iter
        # sample M  conditional on γ₁ and γ₂
        M = sample_M_cond_random_eff(n, N, m, γ₁, γ₂, u)
        #println(M)
        u = sample_u_cond_random_eff(
            n, N, m, M, γ₁, γ₂; prec = 40, method = u_method
        )
        # sample γ₁ conditional on M  and γ₂
        γ₁ = sample_gamma_1_cond_random_eff(
            grid, n, N, γ₂, M, m, u, 
            k_prior, θ_prior, Σ_prior
        )
        #println(γ₁)
        # sample γ₂ conditional on γ₁ and M
        γ₂ = sample_gamma_2_cond_random_eff(
            grid, n, N, γ₁, M, m, u, 
            k_prior, θ_prior, Σ_prior
        )
        #println(γ₂)
        # store them
        for ii in eachindex(M)
            append!(storage[ii], M[ii])
        end # end for
        for ii in eachindex(M)
            append!(storage[ii + eachindex(M)[end]], u[ii])
        end # end for
        append!(storage[end - 1], γ₁)
        append!(storage[end], γ₂)
        next!(prog)
    end # end for
    # return stored values
    storage
end # end funciton