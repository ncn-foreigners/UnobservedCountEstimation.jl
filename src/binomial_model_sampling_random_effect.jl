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

function sample_u_cond_random_eff(n, N, m, M, γ₁, γ₂; prec = 80, method)
    # compute normalization factor
    μ = (N .^ γ₁) .* ((n ./ N) .^ γ₂)
    μ = 1 ./ (1 .+ 1 ./ μ)
    beta_distr = Beta.(n, N - n)
    U = rand(Uniform(0, 1), length(N))
    res = Vector{Float64}()
    #num_nodes = 1_000_000_000
    
    if method == "diffeq"
        # TODO:: this could be much faster if broadcasted but GaussLegendre() doesn't work in R^d where d=>2
        #=for k in eachindex(M)
            ff(x, p, t) = exp(-BigFloat(m[k] * log(x) + (M - m)[k] * log(1 - x * μ[k]) + logpdf(beta_distr[k], x)))
            setprecision(prec) do
                prob = ODEProblem(ff, 0, (0, U[k]))
                sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)
                println(sol)
                error("abc")
            end # end setprecision
        end # end for
        =#
        error("Not yet done")
    elseif method == "exact"
        for k in eachindex(M)
            setprecision(prec) do
                # TODO:: this could be much faster if broadcasted but GaussLegendre() doesn't work in R^d where d=>2
                f(x, p) = exp(BigFloat(m[k] * log(x) + (M - m)[k] * log(1 - x * μ[k]) + logpdf(beta_distr[k], x)))
                # this is deprecated
                prob(x) = IntegralProblem(f, (0, x))
                # this is sooo slow, maybe one could approximate it?      
                #ff(x) = solve(prob(x), Integrals.GaussLegendre(), reltol = 1e-10, abstol = 1e-10)
                ff(x) = solve(prob(x), Integrals.HCubatureJL(), reltol = 1e-10, abstol = 1e-10)
                R = ff(1)
                #= 
                function gr(x)
                    x = x[1]
                    x = (1 + exp(-x[1])) ^ -1
                    2 * (ff(x)[1] / R[1] - U[k]) * f(x, 1) * x * (1 - x) / R[1]
                end # end function
                function hs(x)
                    x   = x[1]
                    p1  = 2 * (f(x, 1) / R[1]) ^ 2

                    p2  = (1 - x) ^ (N[k] - n[k]) * x ^ (n[k] + m[k] - 2) * (1 - μ[k] * x) ^ (M[k] - m[k])
                    p2 *= (N[k] + M[k] - 2) * μ[k] * x ^ 2 + (-μ[k] * n[k] + (1 - M[k]) * μ[k] - m[k] - N[k] + 2) * x + n[k] + m[k] - 1
                    p2 /= (1 - x) ^ 2 * (1 - μ[k] * x)
                    p2 *= 2 * (ff(x)[1] / R[1] - U[k]) / R[1]
                    p1 + p2 / R[1]
                end # end funciton =#
                #a = optimize(x -> (ff((1 + exp(-x[1])) ^ -1)[1] / R[1] - U[k]) ^ 2, gr, [.5], Optim.BFGS(); inplace = false)
                #Optim.NelderMead():
                a = optimize(x -> abs(ff(x)[1] - R[1] * U[k]), 0, 1, Optim.Brent(), abs_tol = 1e-6)
                push!(res, a.minimizer[1])
            end # end set precision
        end # end for
    elseif method == "grid"
        grid = 0.0001:0.0001:(1-0.0001)
        for k in eachindex(M)
            evaluated_denisty = nothing
            evaluated_denisty = exp.(BigFloat.(m[k] * log.(grid) + (M - m)[k] * log.(1 .- grid .* μ[k]) + logpdf.(beta_distr[k], grid)))
            evaluated_denisty[isnan.(evaluated_denisty)] .= 0
            evaluated_denisty ./= sum(evaluated_denisty)

            push!(res, grid[rand(Categorical(evaluated_denisty))])
        end # end for
    end # end if

    #=
    println(all(0 .< res .< 1))
    println(res[.!(0 .< res .< 1)])
    println(M[.!(0 .< res .< 1)])
    println(μ[.!(0 .< res .< 1)])
    println(m[.!(0 .< res .< 1)])
    println(μ)
    error("abc") =#

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