module SNNUtils

using Requires
using SpikingNeuralNetworks
SNN.@load_units
using DrWatson
using Parameters
using Random
using Distributions
using Printf
using JLD
using Serialization
using BSON
using JSON
using ThreadTools
using RollingFunctions
using StatsBase
using Statistics
using LinearAlgebra
using StatisticalMeasures

# include("structs.jl")
include("stimuli/base.jl")
include("models/models.jl")
include("analysis/performance.jl")
include("analysis/EI_balance.jl")
include("analysis/weights.jl")
include("analysis/classifiers.jl")


end

