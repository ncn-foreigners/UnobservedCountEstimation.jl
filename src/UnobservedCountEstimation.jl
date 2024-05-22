module UnobservedCountEstimation

# Imports
# TODO:: explicit function Imports instead of using everything
using Copulas
using DataFrames
using Distributions
using GLM
using Integrals
using FastGaussQuadrature
using LinearAlgebra
using Optim
using ProgressMeter
using SpecialFunctions
using Statistics
using HypergeometricFunctions
# TODO:: StatsBase.jl compatibility
# TODO:: Turing.jl    compatibility

include("zhang_likelihood.jl")
include("original_model.jl")

include("binomial_model_sampling_no_random_effect.jl")
include("binomial_model_sampling_random_effect.jl")
include("binomial_model.jl")
include("struct.jl")

include("interface.jl")

export 
    # functions
    zhang_model,
    binomial_model,
    placeholder,
    posterior_data_generation

    # Concrete types
    ZhangUnobservedCountModel,
    BayesianUnobservedCountModel,

    # Abstract supertype
    AbstractUnobservedCountModel

end # end module