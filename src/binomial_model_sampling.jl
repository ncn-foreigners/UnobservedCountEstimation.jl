function sample_gamma_1_cond(grid, n, N, γ₂, M, m, k_prior, θ_prior, Σ_prior)
    distr_prior_marginal = Gamma(k_prior[1], θ_prior[1])
    copula = GaussianCopula(Σ_prior)
    # compute R_i's
    # is this needed?
    #f(x, p) = exp.(x .* M .* log.(N) - N .^ x) .* pdf(distr, x)
    #prob = IntegralProblem(f, [-Inf, Inf])
    #R = solve(prob, HCubatureJL(), reltol = ε, abstol = ε)
    #println(R)
    #R .*= exp.(-logfactorial.(M)) # <--- if this fails this is probably why
    #println(R)
    #error("bcd")

    # shift grid towards the mean since it is most probable
    grid1 = grid #.* (k_prior[1] * θ_prior[1])
    # get unscaled density function and evaluate it on a grid
    function density_function(x)
        μ = (N .^ x) .* ((n ./ N) .^ γ₂)
        μ = 1 ./ (1 .+ 1 ./ μ)
        lξ = x .* log.(N)
        c = gamma_inc.(k_prior, [x, γ₂] ./ θ_prior)
        c = [c[ii][1] for ii in eachindex(c)]

        #res = (μ .^ m) .* ((1 .- μ) .^ (M .- m))
        res = zeros(BigFloat, length(M))
        res .+= m .* log.(μ) + (M - m) .* log.(1 .- μ) - exp.(lξ) + M .* lξ
        exp(sum(res)) * pdf(copula, c) * pdf(distr_prior_marginal, x)
    end # end funciton
    evaluated_denisty = density_function.(grid1)
    # temporary solution
    evaluated_denisty[isnan.(evaluated_denisty)] .= 0
    #println(evaluated_denisty)
    evaluated_denisty ./= sum(evaluated_denisty)
    # sample acording to evaluation
    grid1[rand(Categorical(evaluated_denisty))]
end # end funciton

function sample_gamma_2_cond(grid, n, N, γ₁, M, m, k_prior, θ_prior, Σ_prior)
    distr_prior_marginal = Gamma(k_prior[2], θ_prior[2])
    copula = GaussianCopula(Σ_prior)
    # shift grid towards the mean since it is most probable
    grid1 = grid #.* (k_prior[2] * θ_prior[2])
    # get unscaled density function and evaluate it on a grid
    function density_function(x)
        μ = (N .^ γ₁) .* ((n ./ N) .^ x)
        μ = 1 ./ (1 .+ 1 ./ μ)
        #ξ = N .^ γ₁
        c = gamma_inc.(k_prior, [γ₁, x] ./ θ_prior)
        c = [c[ii][1] for ii in eachindex(c)]

        #res = (μ .^ m) .* ((1 .- μ) .^ (M .- m))
        res = zeros(BigFloat, length(M))
        res .+= m .* log.(μ) + (M - m) .* log.(1 .- μ)
        exp(sum(res)) * pdf(copula, c) * pdf(distr_prior_marginal, x)
    end # end funciton
    evaluated_denisty = density_function.(grid1)
    # temporary solution
    evaluated_denisty[isnan.(evaluated_denisty)] .= 0
    #println(evaluated_denisty)
    evaluated_denisty ./= sum(evaluated_denisty)
    # sample acording to evaluation
    grid1[rand(Categorical(evaluated_denisty))]
end # end funciton

function sample_M_cond(n, N, m, γ₁, γ₂)
    # compute ξ, μ
    μ = (N .^ γ₁) .* ((n ./ N) .^ γ₂)
    μ = 1 ./ (1 .+ 1 ./ μ)
    ξ = N .^ γ₁
    # draw M-m vector from poisson intependently
    M_minus_m = reduce(vcat, rand.(Poisson.(ξ .* (1 .- μ)), 1))
    # return M = (M-m) + increment
    m + M_minus_m
end # end funciton

function gibbs_sampler_binomial_model(start, grid, iter, n, N, m, k_prior, θ_prior, Σ_prior)
    # create storage vectors
    M  = start[1]
    γ₁ = start[2]
    γ₂ = start[3]

    storage = reduce(vcat, [[[M[k]] for k in eachindex(M)], [[γ₁]], [[γ₂]]])

    for k in 1:iter
        # sample M  conditional on γ₁ and γ₂
        M = sample_M_cond(n, N, m, γ₁, γ₂)
        #println(M)
        # sample γ₁ conditional on M  and γ₂
        γ₁ = sample_gamma_1_cond(
            grid, n, N, γ₂, M, m, 
            k_prior, θ_prior, Σ_prior
        )
        #println(γ₁)
        # sample γ₂ conditional on γ₁ and M
        γ₂ = sample_gamma_2_cond(
            grid, n, N, γ₁, M, m, 
            k_prior, θ_prior, Σ_prior
        )
        #println(γ₂)
        # store them
        for ii in eachindex(M)
            append!(storage[ii], M[ii])
        end # end for
        append!(storage[end - 1], γ₁)
        append!(storage[end], γ₂)
    end # end for
    # return stored values
    storage
end # end funciton