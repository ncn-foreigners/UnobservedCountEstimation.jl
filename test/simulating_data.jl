using UnobservedCountEstimation
using Distributions, Copulas, Random, CSV, DataFrames
using LinearAlgebra
Random.seed!(1234567)

Σ  = [1 .5; .5 1]
k  = [24.0, 13.5]
θ  = [0.04, 0.09523809523809523]
Q  = 20

##### No random effect #####
# This works exactly as expected
my_distr = SklarDist(GaussianCopula(Σ), Gamma.(k, θ))
γ₁, γ₂ = rand(my_distr)

N = rand(Poisson(100), Q)
n = rand.(Binomial.(N, rand(Beta(), Q)))
M = rand.(Poisson.(N .^ γ₁))
u = 1
μ = (N .^ γ₁) .* ((n ./ N) .^ γ₂)
μ = μ ./ (1 .+ μ)
m = rand.(Binomial.(M, u .* μ))

df = DataFrame(
    n = n, N = N, m = m, M = M,
    ξ = N .^ γ₁, μ = μ, u = 1,
    γ₁ = γ₁, γ₂ = γ₂
)

CSV.write("test/test_csv_binomial_no_random_effect.csv", df)

a = binomial_model(
    df[:, :m], df[:, :N], df[:, :n]; 
    start =  "lm", grid = .2:.01:3.5, 
    k_prior = k, theta_prior = θ, 
    sigma_prior = Σ, iter = 1_000, 
    rand_eff = false
)

mean((a.coefs["Mean"][(Q+1):(Q+2)] - [γ₁, γ₂]) ./ [γ₁, γ₂])
mean((a.coefs["Mean"][1:Q] - M) ./ M)
mean((a.coefs["Mean"][1:Q] - N .^ γ₁) ./ (N .^ γ₁))

##### Random Effect #####
my_distr = SklarDist(GaussianCopula(Σ), Gamma.(k, θ))
γ₁, γ₂ = rand(my_distr)

N = rand(Poisson(100), Q)
n = rand.(Binomial.(N, rand(Uniform(.25, .75), Q)))
M = rand.(Poisson.(N .^ γ₁))
u = rand.(Beta.(n, N - n))
μ = (N .^ γ₁) .* ((n ./ N) .^ γ₂)
μ = μ ./ (1 .+ μ)
m = rand.(Binomial.(M, u .* μ))

df = DataFrame(
    n = n, N = N, m = m, M = M,
    ξ = N .^ γ₁, μ = μ, u = u,
    γ₁ = γ₁, γ₂ = γ₂
)

CSV.write("test/test_csv_binomial_with_random_effect.csv", df)

b = binomial_model(
    df[:, :m], df[:, :N], df[:, :n]; 
    start =  "lm", grid = .2:.001:3.5, 
    k_prior = k, theta_prior = θ, 
    sigma_prior = Σ, iter = 1_000, 
    rand_eff = true, u_method = "grid"
)

c = binomial_model(
    df[:, :m], df[:, :N], df[:, :n]; 
    start =  "lm", grid = .2:.001:3.5, 
    k_prior = k, theta_prior = θ, 
    sigma_prior = Σ, iter = 1_000, 
    rand_eff = true, u_method = "exact"
)

mean((b.coefs["Mean"][(Q+1):(Q+2)] - [γ₁, γ₂]) ./ [γ₁, γ₂])
mean((b.coefs["Mean"][1:Q] - M) ./ M)
mean((b.coefs["Mean"][1:Q] - N .^ γ₁) ./ (N .^ γ₁))

mean((c.coefs["Mean"][(Q+1):(Q+2)] - [γ₁, γ₂]) ./ [γ₁, γ₂])
mean((c.coefs["Mean"][1:Q] - M) ./ M)
mean((c.coefs["Mean"][1:Q] - N .^ γ₁) ./ (N .^ γ₁))

#### With Covariates ####
# Number of stratas
#stratas = rand([1, 2, 3, 4, 5], Q)
# for now static equal for all countrie, may be generalized later
stratas = [3 for k in 1:Q]
Σ = rand(Wishart(2 * stratas[1] + 2, Matrix{Float64}(I, 2 * stratas[1] + 1, 2 * stratas[1] + 1)))

#k  = [rand(Normal(22, 3), 2 * stratas[k] + 1) for k in 1:Q]
#θ  = [rand(Exponential(1 / 5), 2 * stratas[1] + 1) for k in 1:Q]
k  = rand(Uniform(17, 22), 2 * stratas[1] + 1)
θ  = rand(Uniform(1 / 22, 1 / 17), 2 * stratas[1] + 1)

my_distr = SklarDist(GaussianCopula(Σ), Gamma.(k, θ))
γ = rand(my_distr)
γ₀ = γ[1]
γ₁ = γ[2:4]
γ₂ = γ[5:7]

N = [rand(Poisson(100 ./ stratas[k]), stratas[k]) for k in 1:Q]
n = [rand.(Binomial.(N[k], rand(Uniform(.10, .6), stratas[k]))) for k in 1:Q]
ξ = [N[k] .^ γ₁ for k in 1:Q]
ξ = reduce(hcat, ξ)'
M = rand.(Poisson.(ξ)) .+ rand.(Poisson.(sum.(N) .^ γ₀))
# country wide random effects
u = rand.(Beta.(sum.(n), sum.(N) - sum.(n)))
#μ = [(N[k] .^ γ₁) .* ((n[k] ./ N[k]) .^ γ₂) for k in 1:Q]
#μ = [1 ./ (1 .+ 1 ./ μ[k]) for k in 1:Q]
n = reduce(hcat, n)'
N = reduce(hcat, N)'
μ  = reduce(vcat, [(k .^ γ₁)' for k in eachrow(N)]) .* reduce(vcat, [(k .^ γ₂)' for k in eachrow(n ./ N)])
μ  = 1 ./ (1 .+ 1 ./ μ)
AAA = [u[k] * μ[k, :] for k in 1:Q]
m = [rand.([Binomial.(M[j, :], AAA[j]) for j in 1:Q][k]) for k in 1:Q]
m = reduce(hcat, m)'

df = DataFrame(
    n₁ = n[:, 1], n₂ = n[:, 2], n₃ = n[:, 3], 
    N₁ = N[:, 1], N₂ = N[:, 2], N₃ = N[:, 3], 
    m₁ = m[:, 1], m₂ = m[:, 2], m₃ = m[:, 3], 
    M₁ = M[:, 1], M₂ = M[:, 2], M₃ = M[:, 3],
    ξ₁ = ξ[:, 1], ξ₂ = ξ[:, 2], ξ₃ = ξ[:, 3], 
    μ₁ = μ[:, 1], μ₂ = μ[:, 2], μ₃ = μ[:, 3], 
    u = u, γ₀ = γ₀, γ₁₁ = γ₁[1], γ₁₂ = γ₁[2], 
    γ₁₃ = γ₁[3], γ₂₁ = γ₂[1], γ₂₂ = γ₂[2], γ₂₃ = γ₂[3]
)

CSV.write("test/test_csv_binomial_with_covariates.csv", df)

## Testing

using SpecialFunctions, HypergeometricFunctions, BenchmarkTools, Plots
# Absolite path is needed
include("/Users/pich3772/Desktop/Julia_package/src/binomial_model_sampling_covariates.jl")

b = @benchmarkable sample_M_matrix_variate_cond_random_eff(n, N, m, γ₀, γ₁, γ₂, u, M) samples = 100 seconds = 600
run(b)

res = zeros(20, 3)
res_1 = zeros(20, 3)

@time (for _ in 1:1000
    res   .+= sample_M_matrix_variate_cond_random_eff(n, N, m, γ₀, γ₁, γ₂, u, M) - M
    res_1 .+= (rand.(Poisson.(ξ)) .+ rand.(Poisson.(sum(N, dims = 2) .^ γ₀))) - M
end)

res   /= 1000
res_1 /= 1000

heatmap(abs.(res - res_1) ./ M)

plot(
    heatmap(res),
    heatmap(res_1)
)