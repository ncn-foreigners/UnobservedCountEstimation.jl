function sample_gamma_1_cond(grid, n, N, γ₂, M, m, μ_γ₁, μ_γ₂, ρ, σ_γ₁, σ_γ₂, ε = 1e-6)
    # get posteriori normal parameters
    μ_γ₁_post  = μ_γ₁ + ρ * σ_γ₂ / σ_γ₁ * (γ₂ - μ_γ₂)
    σ_γ₁_post = sqrt(1 - ρ ^ 2) * σ_γ₁
    distr = Normal(μ_γ₁, σ_γ₁)
    # compute R_i's
    #log_N_sq = log.(N) .^ 2
    #= R = zeros(BigFloat, length(M))
    next_iter = true
    t = 0
    while next_iter
        println(t)
        # Rprev = copy(R)
        # iterate sum
        R_add = cgf.(distr, (M .- t) .* log.(N)) .- logfactorial(t)
        R .+= exp.(R_add) .* (-1) ^ t
        println(R_add)
        # check convergence
        next_iter = any(exp.(R_add) .< ε) | t > 1
        t += 1
    end # end while
    println(R) =#
    f(x, p) = exp.(x .* M .* log.(N) - N .^ x) .* pdf(distr, x)
    prob = IntegralProblem(f, [-Inf, Inf])
    R = solve(prob, HCubatureJL(), reltol = ε, abstol = ε)
    #println(R)
    R .*= exp.(-logfactorial.(M)) # <--- if this fails this is probably why
    #println(R)
    #error("bcd")

    # shift grid towards the mean since it is most probable
    grid1 = grid .+ μ_γ₁_post
    # get unscaled density function and evaluate it on a grid
    function density_function(x)
        μ = (N .^ x) .* ((n ./ N) .^ γ₂)
        μ = 1 ./ (1 .+ exp.(-μ))
        lξ = x .* log.(N)

        #res = (μ .^ m) .* ((1 .- μ) .^ (M .- m))
        res = zeros(BigFloat, length(M))
        res .+= m .* log.(μ) .+ (M .- m) .* log.(1 .- μ) .- log.(R) .- logfactorial.(M .- m) .- exp.(lξ) .+ M .* lξ
        #println(exp(sum(res)))
        exp(sum(res)) .* pdf(Normal(μ_γ₁_post, σ_γ₁_post), x)
    end # end funciton
    evaluated_denisty = density_function.(grid1)
    #println(evaluated_denisty)
    evaluated_denisty ./= sum(evaluated_denisty)
    # sample acording to evaluation
    grid1[rand(Categorical(evaluated_denisty))]
end # end funciton

function sample_gamma_2_cond(grid, n, N, γ₁, M, m, μ_γ₁, μ_γ₂, ρ, σ_γ₁, σ_γ₂)
    # get posteriori normal parameters
    μ_γ₂_post  = μ_γ₂ + ρ * σ_γ₁ / σ_γ₂ * (γ₁ - μ_γ₁)
    σ_γ₂_post = sqrt(1 - ρ ^ 2) * σ_γ₂
    # shift grid towards the mean since it is most probable
    grid1 = grid .- μ_γ₂_post
    # get unscaled density function and evaluate it on a grid
    function density_function(x)
        μ = (N .^ γ₁) .* ((n ./ N) .^ x)
        μ = 1 ./ (1 .+ exp.(-μ))
        #ξ = N .^ γ₁

        res = logfactorial.(M) .- logfactorial.(M .- m) .- logfactorial.(m) .+ m .* log.(μ).+ (M .- m) .* log.(1 .- μ)
        exp(sum(res)) * pdf(Normal(μ_γ₂_post, σ_γ₂_post), x)
    end # end funciton
    evaluated_denisty = density_function.(grid1)
    evaluated_denisty ./= sum(evaluated_denisty)
    # sample acording to evaluation
    grid1[rand(Categorical(evaluated_denisty))]
end # end funciton

function sample_M_cond(n, N, m, γ₁, γ₂)
    # compute ξ, μ
    μ = (N .^ γ₁) .* ((n ./ N) .^ γ₂)
    μ = 1 ./ (1 .+ exp.(-μ))
    ξ = N .^ γ₁
    # draw M-m vector from poisson intependently
    M_minus_m = reduce(vcat, rand.(Poisson.(ξ .* (1 .- μ)), 1))
    # return M = (M-m) + increment
    m + M_minus_m
end # end funciton

function gibbs_sampler_binomial_model(start, grid, iter, n, N, m, μ_γ₁, μ_γ₂, σ_γ₁, σ_γ₂, ρ, ε = 1e-6)
    # create storage vectors
    M  = start[1]
    γ₁ = start[2]
    γ₂ = start[3]

    storage = [[M], [γ₁], [γ₂]]

    for k in iter
        # sample M  conditional on γ₁ and γ₂
        M = sample_M_cond(n, N, m, γ₁, γ₂)
        #println(M)
        # sample γ₁ conditional on M  and γ₂
        γ₁ = sample_gamma_1_cond(
            grid, n, N, γ₂, 
            M, m, μ_γ₁, μ_γ₂, 
            ρ, σ_γ₁, σ_γ₂, ε
        )
        #println(γ₁)
        # sample γ₂ conditional on γ₁ and M
        γ₂ = sample_gamma_2_cond(
            grid, n, N, γ₁, 
            M, m, μ_γ₁, μ_γ₂, 
            ρ, σ_γ₁, σ_γ₂
        )
        #println(γ₂)
        # store them
        append!(storage[1], M)
        append!(storage[2], γ₁)
        append!(storage[3], γ₂)
    end # end for
    # return stored values
    storage
end # end funciton