# TODO:: rename after deciding on type name

abstract type
    AbstractUnobservedCountModel
end


mutable struct ZhangUnobservedCountModel <: AbstractUnobservedCountModel
    coefs
    optim_result
end


mutable struct BayesianUnobservedCountModel <: AbstractUnobservedCountModel
    sim_res
    data
    coefs
    prior
    iter::Int
    Q::Int
end
