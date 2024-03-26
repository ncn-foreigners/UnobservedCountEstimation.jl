using UnobservedCountEstimation
using Test

@test UnobservedCountEstimation.placeholder(65) == 65

@testset "UnobservedCountEstimation.jl" begin
    @test UnobservedCountEstimation.placeholder(65) == 65
end
