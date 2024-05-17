using UnobservedCountEstimation
using Test
# TODO:: CSV is only used for testing, maybe specify that in Project.toml
using CSV, DataFrames, Random, Statistics

@testset "zhang_model.jl" begin
    df = CSV.read(pwd() * "/test_csv_zhang.csv", DataFrame)
    #df = CSV.read(pwd() * "/test/test_csv_zhang.csv", DataFrame)

    a = zhang_model(df[:, :m], df[:, :N], df[:, :n]; start = "glm")
    b = zhang_model(df[:, :m], df[:, :N], df[:, :n]; start =  "lm")

    @test a.coefs[4] ≈ sum(df[:, :ξ]) rtol=.075
    @test a.coefs[4] ≈ sum(df[:, :M]) rtol=.075

    @test b.coefs[4] ≈ sum(df[:, :ξ]) rtol=.075
    @test b.coefs[4] ≈ sum(df[:, :M]) rtol=.075
end # end test "zhang_model.jl"

@testset "binomial_model.jl" begin
    df = CSV.read(pwd() * "/test_csv_binomial_no_random_effect.csv", DataFrame)
    #df = CSV.read(pwd() * "/test/test_csv_binomial_no_random_effect.csv", DataFrame)
    γ₁ = 0.8983650801874796
    γ₂ = 1.578563831198963
    Σ  = [1 .5; .5 1]
    k  = [24.0, 13.5]
    θ  = [0.04, 0.09523809523809523]
    Q  = 20

    Random.seed!(1234)
    a = binomial_model(
        df[:, :m], df[:, :N], df[:, :n]; 
        start = "glm", grid = .3:.001:3, 
        k_prior = k, theta_prior = θ, 
        sigma_prior = Σ, iter = 2_000
    )
    
    res_a = a.coefs["Mean"]
    parm  = reduce(vcat, [df[:, :M], [γ₁], [γ₂]])

    @test res_a ≈ parm rtol = .12
    @test all(quantile.(a.sim_res[1:Q], 2.5 / 100) <= df[:, :M] <= quantile.(a.sim_res[1:Q], 97.5 / 100))
    
    @test res_a[(end-1):end] ≈ [γ₁, γ₂] rtol = .06
    @test quantile(a.sim_res[end - 1], 2.5 / 100) <= γ₁ <= quantile(a.sim_res[end - 1], 97.5 / 100)
    @test quantile(a.sim_res[end],     2.5 / 100) <= γ₂ <= quantile(a.sim_res[end],     97.5 / 100)

    # new data
    df = CSV.read(pwd() * "/test_csv_binomial_with_random_effect.csv", DataFrame)
    #df = CSV.read(pwd() * "/test/test_csv_binomial_with_random_effect.csv", DataFrame)
    # γ parameters are the same as in previous case
    Random.seed!(1234)
    b = binomial_model(
        df[:, :m], df[:, :N], df[:, :n]; 
        start =  "lm", grid = .3:.01:2.5, 
        k_prior = k, theta_prior = θ, 
        sigma_prior = Σ, iter = 1_000, 
        rand_eff = true, u_method = "grid"
    )

    res_b = reduce(vcat, [b.coefs["Mean"][1:Q], [b.coefs["Mean"][end - 1]], [b.coefs["Mean"][end]]])
    parm  = reduce(vcat, [df[:, :M], [γ₁], [γ₂]])

    @test res_b[1:Q] ≈ df[:, :M] rtol = .15
    @test all(quantile.(b.sim_res[1:Q], 2.5 / 100) .<= df[:, :M] .<= quantile.(b.sim_res[1:Q], 97.5 / 100))
    
    @test res_b[Q + 1] ≈ γ₁ rtol = .05
    @test quantile(b.sim_res[end - 1], 2.5 / 100) <= γ₁ <= quantile(b.sim_res[end - 1], 97.5 / 100)
    @test quantile(b.sim_res[end ],    2.5 / 100) <= γ₂ <= quantile(b.sim_res[end],     97.5 / 100)

    Random.seed!(1234)
    c = binomial_model(
        df[:, :m], df[:, :N], df[:, :n]; 
        start =  "lm", grid = .3:.01:2.5, 
        k_prior = k, theta_prior = θ, 
        sigma_prior = Σ, iter = 300, 
        rand_eff = true, u_method = "exact"
    )

    res_c = reduce(vcat, [c.coefs["Mean"][1:Q], [c.coefs["Mean"][end - 1]], [c.coefs["Mean"][end]]])
    @test res_c[1:Q] ≈ df[:, :M] rtol = .5

    @test res_c[Q + 1] ≈ γ₁ rtol = .15

    @test quantile(c.sim_res[end - 1],  5 / 100) <= γ₁ <= quantile(c.sim_res[end - 1],    1)
    @test quantile(c.sim_res[end ],   2.5 / 100) <= γ₂ <= quantile(c.sim_res[end],     97.5 / 100)
end # end test "binomial_model.jl"

