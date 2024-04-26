using UnobservedCountEstimation
using Test
# TODO:: CSV is only used for testing, maybe specify that in Project.toml
using CSV, DataFrames, Random, Statistics

@testset "zhang_model.jl" begin
    df = CSV.read(pwd() * "/test_csv_zhang.csv", DataFrame)

    a = zhang_model(df[:, :m], df[:, :N], df[:, :n]; start = "glm")
    b = zhang_model(df[:, :m], df[:, :N], df[:, :n]; start =  "lm")

    @test a[1][4] ≈ sum(df[:, :ξ]) rtol=.075
    @test a[1][4] ≈ sum(df[:, :M]) rtol=.075

    @test b[1][4] ≈ sum(df[:, :ξ]) rtol=.075
    @test b[1][4] ≈ sum(df[:, :M]) rtol=.075
end

@testset "binomial_model.jl" begin
    Random.seed!(1234)
    df = CSV.read(pwd() * "/test_csv_binomial_no_random_effect.csv", DataFrame)
    γ₁ = 0.8983650801874796
    γ₂ = 1.578563831198963
    Σ  = [1 .5; .5 1]
    k  = [24.0, 13.5]
    θ  = [0.04, 0.09523809523809523]
    Q  = 20

    a = binomial_model(df[:, :m], df[:, :N], df[:, :n]; start = "glm", grid = .3:.01:3, k_prior = k, θ_prior = θ, Σ_prior = Σ)
    b = binomial_model(df[:, :m], df[:, :N], df[:, :n]; start =  "lm", grid = .3:.01:3, k_prior = k, θ_prior = θ, Σ_prior = Σ)
    
    res_a = reduce(vcat, [[mean(a[k]) for k in 1:Q], [mean(a[end-1])], [mean(a[end])]])
    res_b = reduce(vcat, [[mean(a[k]) for k in 1:Q], [mean(a[end-1])], [mean(a[end])]])
    parm  = reduce(vcat, [df[:, :M], [γ₁], [γ₂]])

    @test res_a ≈ parm rtol = .25
    @test res_b ≈ parm rtol = .25

    @test res_a[(end-1):end] ≈ [γ₁, γ₂] rtol = .06
    @test res_b[(end-1):end] ≈ [γ₁, γ₂] rtol = .06
end

