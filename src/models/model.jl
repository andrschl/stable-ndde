abstract type Model end

abstract type GroundTruthModel <: Model end

mutable struct ODEGroundTruthModel <: GroundTruthModel


end

mutable struct DDEGroundTruthModel <: GroundTruthModel


end

abstract type DDEModel <: Model end

abstract type

predict

dense_predict
