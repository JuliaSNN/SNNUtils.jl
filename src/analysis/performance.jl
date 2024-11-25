using StatsBase

function evaluate_avg_firing_rate(population, intervals::Vector{Vector{Float32}}, target::Symbol, cells::Dict{Symbol,Any} = Dict())
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

function compute_firing_rates_moving_window(spike_times::Vector{Float32}, 
    stimulus_intervals::Vector{Vector{Vector{Float32}}}, 
    window_shift::Float64, 
    targets::Vector{Int64})
    # Number of shifts based on the window shift
    n_shifts = Int(100ms / window_shift)  

    target_labels = []
    for (target, intervals) in zip(targets, stimulus_intervals) 
        append!(target_labels, fill(target, length(intervals)))
    end
     
    # Initialize arrays to store features and targets
    features = []  # To store firing rates

    for intervals in stimulus_intervals 
        for interval in intervals
            firing_rates = Float32[]
            # Calculate firing rates for each shifted interval
            for shift_idx in 0:n_shifts-1 
                shifted_interval = interval[1] + shift_idx * window_shift, interval[2] + shift_idx * window_shift
                start_time, end_time = shifted_interval
                count = StatsBase.count(spike -> spike >= start_time && spike <= end_time, spike_times)
                interval_length = end_time - start_time
                append!(firing_rates, count / interval_length)
            end
            push!(features, firing_rates)
        end
    end

    dataset = (
        features = features,    # Array of firing rates (features)
        targets = target_labels # Vector of targets
    )

    return dataset
end


export evaluate_avg_firing_rate, compute_weight, evaluate_logistic_regression, compute_firing_rates_moving_window