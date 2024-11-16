function evaluate(population, intervals::Vector{Vector{Float32}}, target::Symbol, cells::Dict{Symbol,Any} = Dict())
    count = 0
    for time_interval in intervals
        interval_range = range(first(time_interval), stop=last(time_interval), length=length(time_interval))
        firing_rates = Dict(w => mean(SNN.average_firing_rate(population; interval=interval_range, pop=cells[w])) for w in keys(cells))

        if all(firing_rates[target] > firing_rates[w] for w in keys(cells) if w != target)
            count += 1
        end
    end
    return count / length(intervals)
end


function compute_weight(pre_pop_cells, post_pop_cells, synapse)
    Win = synapse.W
    rowptr = synapse.rowptr
    J = synapse.J  # Presynaptic neuron indices
    index = synapse.index 
    all_weights = Float64[]  # Store weights for all filtered connections

    for neuron in post_pop_cells
        # Get the range in W for this postsynaptic neuron's incoming connections
        for st = rowptr[neuron]:(rowptr[neuron + 1] - 1)
            st = index[st]
            if (J[st] in pre_pop_cells)
                push!(all_weights, Win[st])
            end
        end
    end

    return mean(all_weights)
end

export evaluate, compute_weight