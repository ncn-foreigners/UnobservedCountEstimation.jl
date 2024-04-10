# likelihood and derivatives go here

function log_lik_binomial_model(M, α, β, m, N, n, X, Z)
    μ = (N .^ (X * α)) .* ((n ./ N) .^ (Z * β))
    M = exp.(M) .+ m
    μ = exp.(μ) ./ (1 .+ exp.(μ))
    p = μ ./ M

    sum(loggamma.(M .+ 1) .- loggamma.(m .+ 1.0) .- loggamma.(M .- m .+ 1.0) .+ m .* log.(p) .+ (M .- m) .* log.(1 .- p))
end # end function

function grad_log_lik_binomial_model(M, α, β, m, N, n, X, Z)
    μ = (N .^ (X * α)) .* ((n ./ N) .^ (Z * β))
    Mprev = copy(M)
    M = exp.(M) .+ m
    μ = exp.(μ) ./ (1 .+ exp.(μ))
    p = μ ./ M

    dp = m ./ p .- (M .- m) ./ (1 .- p)

    dα = dp .* (1 .- μ) .* μ .* (N .^ (X * α)) .* ((n ./ N) .^ (Z * β)) .* log.(N) ./ M
    dβ = dp .* (1 .- μ) .* μ .* (N .^ (X * α)) .* ((n ./ N) .^ (Z * β)) .* log.(n ./ N) ./ M

    dM = digamma.(M .+ 1) .- digamma.(M .- m .+ 1) .- (m ./ M) .+ (M .- m) .* ((1 .- p) .^ -1) .* (p ./ M) .+ log.(1 .- p)
    # TODO:: derivative correction M, exp link

    vcat(dM .* exp.(Mprev), X' * dα, Z' * dβ)
end # end function

function hess_log_lik_binomial_model(M, α, β, m, N, n, X, Z)
    μ = (N .^ (X * α)) .* ((n ./ N) .^ (Z * β))
    μ = exp.(μ) ./ (1 .+ exp.(μ))
    Mprev = copy(M)
    M = exp.(M) .+ m
    p = μ ./ M
    
    dp   = m ./ p .- (M .- m) ./ (1 .- p)
    dp_2 = -m ./ p .^ 2 .- (M .- m) ./ (1 .- p) .^ 2

    dα_2   = -2 .* μ .^ 2 .* (1 .- μ) .* (log.(N) .^ 2) .* (N .^ (2 * X * α)) .* ((n ./ N) .^ (2 * Z * β))
    dα_2  += (log.(N) .^ 2) .* (N .^ (2 * X * α)) .* ((n ./ N) .^ (2 * Z * β)) .* μ .* (1 .- μ)
    dα_2  += (N .^ (X * α)) .* ((n ./ N) .^ (Z * β)) .* (log.(N) .^ 2) .* μ .* (1 .- μ)
    dα_2 .*= dp ./ M
    dα_2  += dp_2 .* ((1 .- μ) .* μ .* (N .^ (X * α)) .* ((n ./ N) .^ (Z * β)) .* log.(N) ./ M) .^ 2

    dβ_2   = -2 .* μ .^ 2 .* (1 .- μ) .* (log.(n ./ N) .^ 2) .* (N .^ (2 * X * α)) .* ((n ./ N) .^ (2 * Z * β))
    dβ_2  += (log.(n ./ N) .^ 2) .* (N .^ (2 * X * α)) .* ((n ./ N) .^ (2 * Z * β)) .* μ .* (1 .- μ)
    dβ_2  += (N .^ (X * α)) .* ((n ./ N) .^ (Z * β)) .* (log.(n ./ N) .^ 2) .* μ .* (1 .- μ)
    dβ_2 .*= dp ./ M
    dβ_2  += dp_2 .* ((1 .- μ) .* μ .* (N .^ (X * α)) .* ((n ./ N) .^ (Z * β)) .* log.(N) ./ M) .^ 2

    dαdβ   = dp .* (N .^ (X * α)) .* ((n ./ N) .^ (Z * β)) .* log.(N) .* log.(n ./ N) .* μ .* (1 .- μ) ./ M
    dαdβ .*= ((N .^ (X * α)) .* ((n ./ N) .^ (Z * β)) .* (exp.((N .^ (X * α)) .* ((n ./ N) .^ (Z * β))) .- 1) - exp.((N .^ (X * α)) .* ((n ./ N) .^ (Z * β))) .- 1)
    dαdβ  += dp_2 .* ((1 .- μ) .* μ) .^ 2 .* (N .^ (2 * X * α)) .* ((n ./ N) .^ (2 * Z * β)) .* log.(N) .* log.(n ./ N) ./ (M .^ 2)

    dpdM   = (1 .- m ./ M) .* ((1 .- p) .^ -2) .- (1 .- p) .^ -1
    dpdM .*=  exp.(Mprev)

    dαdM = dpdM .* (1 .- μ) .* μ .* (N .^ (X * α)) .* ((n ./ N) .^ (Z * β)) .* log.(N) ./ M
    dβdM = dpdM .* (1 .- μ) .* μ .* (N .^ (X * α)) .* ((n ./ N) .^ (Z * β)) .* log.(n ./ N) ./ M

    dM_2  = trigamma.(M .+ 1) .- trigamma.(M .- m .+ 1) .+ m ./ M .^ 2 
    dM_2 += (m ./ M .^ 2) .* p ./ (1 .- p) .+ μ ./ ((1 .- p) .* M .^ 2)
    dM_2 -= (1 .- m ./ M) .* (p ./ M) ./ (1 .- p) .^ 2
    dM    = digamma.(M .+ 1) .- digamma.(M .- m .+ 1) .- (m ./ M) .+ (M .- m) .* ((1 .- p) .^ -1) .* (p ./ M) .+ log.(1 .- p)
    dM_2  = dM_2 .* exp.(2 .* Mprev) .+ dM .* exp.(Mprev)
    # TODO:: derivative correction M, exp link

    ## TODO if M is a vector then dM_2 is a diagonal matrix and dαdM
    vcat(
        hcat(Diagonal(dM_2[:, 1]), Diagonal(dαdM[:, 1]) * X, Diagonal(dβdM[:, 1]) * Z),
        hcat(X' * Diagonal(dαdM[:, 1]), X' * (dα_2 .* X), (X' * (dαdβ .* Z))'),
        hcat(Z' * Diagonal(dβdM[:, 1]), (X' * (dαdβ .* Z))', Z' * (dβ_2 .* Z))
    )
end # end function