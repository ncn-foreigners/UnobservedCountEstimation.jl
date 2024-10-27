using UnobservedCountEstimation
#using Test
# TODO:: CSV is only used for testing, maybe specify that in Project.toml
using CSV, DataFrames, Random, Statistics

df = CSV.read(pwd() * "/test/test_csv_binomial_with_random_effect.csv", DataFrame)
γ₁ = 0.8983650801874796
γ₂ = 1.578563831198963
Σ  = [1 .5; .5 1]
k  = [24.0, 13.5]
θ  = [0.04, 0.09523809523809523]
Q  = 20

Random.seed!(1234567)
c = binomial_model(
    df[:, :m], df[:, :N], df[:, :n]; 
    start =  "lm", grid = .2:.01:3.5, 
    k_prior = k, theta_prior = θ, 
    sigma_prior = Σ, iter = 1_000, 
    rand_eff = true, u_method = "exact"
)
res_c = reduce(vcat, [c.coefs["Mean"][1:Q], [c.coefs["Mean"][end - 1]], [c.coefs["Mean"][end]]])
#res_c = reduce(vcat, [c.coefs["MAP"][1:Q], [c.coefs["MAP"][end - 1]], [c.coefs["MAP"][end]]])
#res_c = reduce(vcat, [[mean(c.sim_res[k][501:end]) for k in 1:Q], mean(c.sim_res[end - 1][501:end]), mean(c.sim_res[end][501:end])])
#res_c = reduce(vcat, [[mean(c.sim_res[k][151:end]) for k in 1:Q], mean(c.sim_res[end - 1][151:end]), mean(c.sim_res[end][151:end])])
mean(res_c[1:Q] - df[:, :M])
mean((res_c[1:Q] - df[:, :M]) ./ df[:, :M])
sum(res_c[1:Q] - df[:, :M]) ./ sum(df[:, :M])