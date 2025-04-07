# function evaluate(population, intervals::Vector{Vector{Float32}}, target::Symbol, neurons::Dict{Symbol,Any} = Dict())
#     count = 0
#     for time_interval in intervals
#         interval_range = range(first(time_interval), stop=last(time_interval), length=length(time_interval))
#         firing_rates = Dict(w => mean(SNN.average_firing_rate(population; interval=interval_range, pop=neurons[w])) for w in keys(neurons))

#         if all(firing_rates[target] > firing_rates[w] for w in keys(neurons) if w != target)
#             count += 1
#         end
#     end
#     return count / length(intervals)
# end

function average_weight(pre_pop_neurons::Vector{Int}, post_pop_neurons::Vector{Int}, synapse::SNN.SpikingSynapse)
    @unpack W = synapse
    rowptr = synapse.rowptr
    J = synapse.J  # Presynaptic neuron indices
    index = synapse.index 
    all_weights = Float32[]  # Store weights for all filtered connections
    for neuron in post_pop_neurons
        # Get the range in W for this postsynaptic neuron's incoming connections
        for st = rowptr[neuron]:(rowptr[neuron + 1] - 1)
            st = index[st]
            if (J[st] in pre_pop_neurons)
                push!(all_weights, W[st])
            end
        end
    end
    return mean(all_weights)
end

function weights_indices(pre_pop_neurons::Vector{Int}, post_pop_neurons::Vector{Int}, synapse::SNN.SpikingSynapse)
    rowptr = synapse.rowptr
    J = synapse.J  # Presynaptic neuron indices
    index = synapse.index 
    indices = Int64[]  # Store weights for all filtered connections
    for neuron in post_pop_neurons
        # Get the range in W for this postsynaptic neuron's incoming connections
        for st = rowptr[neuron]:(rowptr[neuron + 1] - 1)
            st = index[st]
            if (J[st] in pre_pop_neurons)
                push!(indices, st)
            end
        end
    end
    return indices
end

function update_weight!(pre_pop_neurons::Vector{Int}, post_pop_neurons::Vector{Int}, synapse::SNN.SpikingSynapse)
    @unpack W = synapse
    rowptr = synapse.rowptr
    J = synapse.J  # Presynaptic neuron indices
    index = synapse.index 
    for neuron in post_pop_neurons
        # Get the range in W for this postsynaptic neuron's incoming connections
        for st = rowptr[neuron]:(rowptr[neuron + 1] - 1)
            st = index[st]
            if (J[st] in pre_pop_neurons)
                W[st] *= 1.2
            end
        end
    end
end

# export evaluate, 
export average_weight, update_weight!, weights_indices
