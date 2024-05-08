module UnobservedCountEstimation

# Write your package code here.

# Imports
using Optim 
using Statistics
using GLM
using Distributions
using DataFrames
using Integrals, FastGaussQuadrature
using SpecialFunctions
using LinearAlgebra
using Copulas
using ProgressMeter

include("zhang_likelihood.jl")
include("original_model.jl")

include("binomial_model_sampling_no_random_effect.jl")
include("binomial_model_sampling_random_effect.jl")
include("binomial_model.jl")

include("interface.jl")

export placeholder
export zhang_model
export binomial_model

end
