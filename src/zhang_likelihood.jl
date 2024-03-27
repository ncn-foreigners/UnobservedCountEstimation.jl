# likelihood and derivatives go here

function log_lik_original_model(α, β, ϕ, m, N, n, X, Z)
    ϕ = exp(ϕ)
    μ = (N .^ (X * α)) .* ((n ./ N) .^ (Z * β))
    sum(m .* log.(μ) .- (m .+ ϕ) .* log.(μ .+ ϕ) .+ ϕ .* log.(ϕ) .- loggamma.(ϕ) .- loggamma.(m .+ 1) .+ loggamma.(m .+ ϕ))
end # end function

function grad_log_lik_original_model(α, β, ϕ, m, N, n, X, Z)
    ϕ = exp(ϕ)
    # SpecialFunctions.jl has digamma and trigamma function
    μ = (N .^ (X * α)) .* ((n ./ N) .^ (Z * β))
    dμ = m ./ μ .- (ϕ .+ m) ./ (μ .+ ϕ)

    dα = dμ .* μ .* log.(N)
    dβ = dμ .* μ .* log.(n ./ N)
    dϕ = sum(SpecialFunctions.digamma.(ϕ .+ m) .- log.(μ .+ ϕ) .+ log.(ϕ) .- SpecialFunctions.digamma.(ϕ) .+ 1 .- (ϕ .+ m) ./ (ϕ .+ μ))
    dϕ *= ϕ # log link

    vcat(X' * dα, Z' * dβ, dϕ)
end # end function

function hess_log_lik_original_model(α, β, ϕ, m, N, n, X, Z)
    ϕ = exp(ϕ)
    μ = (N .^ (X * α)) .* ((n ./ N) .^ (Z * β))
    dμ = m ./ μ .- (ϕ .+ m) ./ (μ .+ ϕ)
    dμ_2 = (ϕ .+ m) ./ (μ .+ ϕ) .^ 2 .- m ./ μ .^ 2
    dμdϕ = - (μ .- m) ./ (μ .+ ϕ) .^ 2
    # log link
    dμdϕ *= ϕ

    #dϕ = sum(SpecialFunctions.digamma.(ϕ .+ m) .- log.(μ .+ ϕ) .+ log.(ϕ) .- SpecialFunctions.digamma.(ϕ) .+ 1 .- (ϕ .+ m) ./ (ϕ .+ μ))
    dϕ_2  = sum(SpecialFunctions.trigamma.(ϕ .+ m) .- SpecialFunctions.trigamma.(ϕ) .- 2 ./ (ϕ .+ μ) .+ (ϕ .+ m) ./ (ϕ .+ μ) .^ 2 .+ ϕ ^ -1)
    # log link
    dϕ_2 *= ϕ ^ 2
    dϕ_2 += sum(SpecialFunctions.digamma.(ϕ .+ m) .- log.(μ .+ ϕ) .+ log.(ϕ) .- SpecialFunctions.digamma.(ϕ) .+ 1 .- (ϕ .+ m) ./ (ϕ .+ μ)) * ϕ

    dα_2 = dμ_2 .* (μ .* log.(N)) .^ 2 .+ dμ .* μ .* (log.(N) .^ 2)
    dβ_2 = dμ_2 .* (μ .* log.(n ./ N)) .^ 2 .+ dμ .* μ .* (log.(n ./ N) .^ 2)
    dαdβ = dμ_2 .* μ .* log.(N) .* μ .* log.(n ./ N) .+ dμ .* μ .* log.(N) .* log.(n ./ N)
    dαdϕ = dμdϕ .* μ .* log.(N)
    dβdϕ = dμdϕ .* μ .* log.(n ./ N)

    vcat(
        hcat(X' * (dα_2 .* X), X' * (dαdβ .* Z), X' * dαdϕ),
        hcat((X' * (dαdβ .* Z))', Z' * (dβ_2 .* Z), Z' * dβdϕ),
        hcat((X' * dαdϕ)', (Z' * dβdϕ)', dϕ_2)
    )
end # end function