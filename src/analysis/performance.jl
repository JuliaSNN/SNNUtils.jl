# using LinearAlgebra
# using ScikitLearn
# @sk_import linear_model: LogisticRegression
# @sk_import metrics: accuracy_score, confusion_matrix

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

# function evaluate_logistic_regression(X, y)
#     # Split 80% training, 20% test
#     train_ratio = 0.8
#     n_train = floor(Int, train_ratio * n_trials)

#     X_train = X[1:n_train, :]
#     y_train = y[1:n_train]

#     X_test = X[n_train+1:end, :]
#     y_test = y[n_train+1:end]


#     # Initialize and train logistic regression
#     clf = LogisticRegression()
#     fit!(clf, X_train, y_train)

#     # Predict on test set
#     y_pred = predict(clf, X_test)

#     # Calculate accuracy
#     accuracy = accuracy_score(y_test, y_pred)

#     return accuracy
# end

export evaluate_avg_firing_rate, compute_weight, evaluate_logistic_regression