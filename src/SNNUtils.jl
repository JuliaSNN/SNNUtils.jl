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
using Dates
using BSON
using JSON
using ThreadTools
using RollingFunctions
using StatsBase
using Statistics


# include("structs.jl")
include("stimuli/base.jl")
include("models/models.jl")
include("analysis/performance.jl")

function __init__()
    # @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plot.jl")
    @require MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7" include("analysis/classifiers.jl")
end

end


