using LIBSVM
using CategoricalArrays
using StatsBase
using MultivariateStats
using LinearAlgebra
using StatisticalMeasures
import SNNModels: AbstractPopulation

"""
    SVCtrain(Xs, ys; seed=123, p=0.6)
    
    Train a Support Vector Classifier with a linear kernel on the data Xs and labels ys.

    # Arguments
    - `Xs::Matrix{Float32}`: The data matrix with shape `(n_features, n_samples)`.
    - `ys::Vector{Int64}`: The labels.

    # Returns
    The accuracy of the classifier on the test set.


"""
function SVCtrain(Xs, ys; seed = 123, p = 0.5, labels = false)
    X = Xs .+ 1e-2
    y = string.(ys)
    y = CategoricalVector(string.(ys))
    @assert length(y) == size(Xs, 2)
    if p < 1
        train, test = partition(eachindex(y), p, rng = seed, stratify = y)
        # n = 1
        if length(train)<2 || length(test)<2
            @error "Not enough samples in train or test set"
            Xtrain = X
            Xtest = X
            ytrain = y
            ytest = y
        else
            ZScore = StatsBase.fit(StatsBase.ZScoreTransform, X[:, train], dims = 2)
            Xtrain = StatsBase.transform(ZScore, X[:, train])
            Xtest = StatsBase.transform(ZScore, X[:, test])
            ytrain = y[train]
            ytest = y[test]
        end
    else
        Xtrain = X
        Xtest = X
        ytrain = y
        ytest = y
    end

    @assert size(Xtrain, 2) == length(ytrain)
    mach = svmtrain(Xtrain, ytrain, kernel = Kernel.Linear)
    ŷ, decision_values = svmpredict(mach, Xtest);
    confusion_matrix = confmat(ŷ, ytest)
    score = kappa(confusion_matrix)
    return score, confusion_matrix
    # if labels
    #     return ŷ, ytest, mean(ŷ .== ytest)
    # else
    #     return mean(ŷ .== ytest)
    # end
end

"""
    trial_average(array::Array, sequence::Vector, dim::Int = -1)

    Compute the trial-averaged data for each unique label in the sequence.

    # Arguments
    - `array::Array`: The input data array with dimensions (..., n_trials).
    - `sequence::Vector`: A vector of labels corresponding to each trial.
    - `dim::Int = -1`: The dimension along which to average (default is the last dimension).

    # Returns
    - `spatial_code::Array{Float32}`: An array containing the trial-averaged data for each label.
    - `labels::Vector`: The sorted unique labels from the sequence.

    # Notes
    - The function averages the data along the specified dimension for each unique label.
    - The output array has dimensions (..., n_labels) where n_labels is the number of unique labels.
    - The labels are sorted in ascending order.
"""
function trial_average(array::Array, sequence::Vector, dim::Int = -1)
    trial_dim = dim < 0 ? ndims(array) : dim
    my_dims = collect(size(array))
    popat!(my_dims, trial_dim)
    labels = unique(sequence) |> sort
    spatial_code = zeros(Float32, my_dims..., length(labels))
    ave_dim = ndims(spatial_code)

    for i in eachindex(labels)
        sound = labels[i]
        sound_ids = findall(==(sound), sequence)
        selectdim(spatial_code, ave_dim, i) .=
            mean(selectdim(array, trial_dim, sound_ids), dims = ave_dim)
    end
    return spatial_code, labels
end

function trial_sort(array::Array, sequence::Vector, dim::Int = -1)
    trial_dim = dim < 0 ? ndims(array) : dim
    labels = unique(sequence) |> sort

    data = Dict{Symbol, Vector{Array}}()
    for i in eachindex(labels)
        sound = labels[i]
        sound_ids = findall(==(sound), sequence)
        data[sound] = Vector{Array}()
        for id in sound_ids
            push!(data[sound], copy(selectdim(array, trial_dim, id)))
        end
    end
    return data, labels
end

export trial_average, trial_sort


"""
    spikecount_features(pop::T, offsets::Vector) where {T<:AbstractPopulation}

    Extract spike count features from a population of neurons over specified time intervals.

    # Arguments
    - `pop::T`: The population object containing the recorded data.
    - `offsets::Vector`: A vector of time intervals (each as a range) for which to compute spike counts.

    # Returns
    Matrix::Float32: The feature matrix (n_features x n_samples) containing the spike counts of each neuron over each time interval.

    # Notes
    - The function counts spikes for each neuron within each specified time interval.
    - Uses parallel processing with Threads.@threads for efficiency.
    - The output matrix has dimensions (n_neurons, n_offsets) where n_neurons is the number of neurons in the population.
"""
function spikecount_features(pop::T, offsets::Vector) where {T<:AbstractPopulation}
    N = pop.N
    X = zeros(N, length(offsets))
    Threads.@threads for i in eachindex(offsets)
        offset = offsets[i]
        X[:, i] = length.(spiketimes(pop, interval = offset))
    end
    return X
end



"""
    sym_features(sym::Symbol, pop::T, offsets::Vector) where T <: AbstractPopulation

    Return a matrix with the mean of the interpolated record of the symbol `sym` in the population `pop` for each offset in `offsets`.

    # Arguments
    - `sym::Symbol`: The symbol representing the variable to extract from the population.
    - `pop::T`: The population object containing the recorded data.
    - `offsets::Vector`: A vector of time intervals (each as a range) for which to compute the mean.

    # Returns
    Matrix::Float32: The feature matrix (n_features x n_samples) containing the mean values of the specified variable over each time interval.

    # Notes
    - The function uses interpolation to compute values at 1ms intervals within each offset range.
    - If an offset range exceeds the available recorded data, that offset is skipped.
    - The function is thread-safe and uses parallel processing for efficiency.
"""
function sym_features(sym::Symbol, pop::T, offsets::Vector) where {T<:AbstractPopulation}
    N = pop.N
    X = zeros(N, length(offsets))
    var, r_v = interpolated_record(pop, sym)
    Threads.@threads for i in eachindex(offsets)
        offset = offsets[i]
        offset[end] > r_v[end] && continue
        range = offset[1]:1ms:offset[2]
        X[:, i] = mean(var[:, range], dims = 2)[:, 1]
    end
    return X
end

"""
    score_spikes(model, seq, target_interval = :offset, delay = nothing, pop = :E)

    Evaluate the performance of a neural population in recognizing word assemblies based on spike activity.

    # Arguments
    - `model`: The model containing the neural population and stimulus information.
    - `seq`: The sequence object containing the experimental protocol information. `seq` must contain: 
        - `seq.sequence[seq.line_id.type, :]`: The type of target intervals (e.g., :offset).
        - `seq.sequence[seq.line_id.words, :]`: The words associated with each target interval.
        - `seq.sequence[seq.line_id.offset, :]`: The offset times for each target interval.
    - `target_interval = :offset`: The target interval type to analyze (default: :offset).
    - `delay = nothing`: Optional delay to apply to the target intervals. If nothing, the function will search for the optimal delay.
    - `pop = :E`: The population to analyze (default: :E).

    # Returns
    - If `delay` is provided:
        - `score`: The Cohen's kappa score for the classification performance.
        - `confusion_matrix`: The confusion matrix of the classification.
        - `activity_matrix`: The matrix showing the average activity of each word assembly during each target word.
    - If `delay` is not provided:
        - `scores`: A vector of Cohen's kappa scores for each tested delay.
        - `best_delay`: The delay that yielded the highest score.
        - `cms`: A named tuple containing all confusion matrices and the corresponding delays.

    # Notes
    - The function identifies word assemblies from the stimulus information and evaluates how well the neural population can recognize these assemblies based on spike activity.
    - If no delay is specified, the function tests delays from -100ms to 100ms in 10ms increments to find the optimal delay.
    - The function uses parallel processing with a spin lock for thread safety.
    - Spike activity is binned at 10ms intervals for analysis.
"""
function score_spikes(model, seq, target_interval = :offset, delay = nothing, pop = :E)
    ## Get word intervals 
    offsets_ids = findall(seq.sequence[seq.line_id.type, :] .== target_interval)
    words = seq.sequence[seq.line_id.words, offsets_ids]
    offset_times = seq.sequence[seq.line_id.offset, offsets_ids]

    if isempty(offsets_ids)
        throw("No target intervals found in sequence")
    end
    ## Get word assemblies
    labels, neurons = subpopulations(filter(x->!occursin("noise", x.name), model.stim))
    word_assemblies =
        Dict(
            Symbol(labels[n][3:end]) => neurons[n] for
            n in eachindex(labels) if startswith(labels[n], "w_")
        ) |> dict2ntuple
    word_list = keys(word_assemblies) |> collect |> sort
    word_count = [count(x->x==word, words) for word in word_list]
    assemblies = [word_assemblies[word] for word in word_list]

    my_lock = Threads.SpinLock()
    confusion_matrix = zeros(Float32, length(word_assemblies), length(word_assemblies))
    activity_matrix = zeros(Float32, length(word_assemblies), length(word_assemblies))
    _spikes = spiketimes(model.pop[pop])
    spike_count, r = bin_spiketimes(_spikes, interval = 0:10ms:(offset_times[end]+100ms))

    function _score(delay, test_interval = 0:100)
        predicted = Symbol[]
        target = Symbol[]
        @inbounds @fastmath for i in eachindex(offsets_ids)
            target_word = findfirst(word_list .== words[i])
            target_interval = offset_times[i] .+ test_interval .+ delay
            r_idx = findall(
                x->(r[x]>target_interval[1] && r[x]<target_interval[end]),
                eachindex(r),
            )
            _spikes = sum(spike_count[:, r_idx], dims = 2)[:, 1]

            lock(my_lock)
            for word in eachindex(word_list)
                activity_matrix[word, target_word] +=
                    mean(_spikes[assemblies[word]]) / word_count[target_word]
            end
            push!(
                predicted,
                word_list[argmax(mean.([_spikes[assembly] for assembly in assemblies]))],
            )
            push!(target, word_list[target_word])
            unlock(my_lock)
        end
        confusion_matrix = confmat(target, predicted)
        score = kappa(confusion_matrix)
        return score, confusion_matrix, activity_matrix
    end

    if !isnothing(delay)
        return _score(delay)
    else
        delays = -100:10:100
        scores = Vector{Float32}(undef, length(delays))
        cms = Vector{Any}(undef, length(delays))
        for i in eachindex(delays)
            score, cm, _ = _score(delays[i])
            scores[i] = score
            cms[i] = cm
        end
        best_score = argmax(scores)
        best_delay = delays[best_score]
        return scores, best_delay, (; cms, delays)
    end
end


"""
    MultinomialLogisticRegression(X::Matrix{Float64}, labels::Array{Int64}; λ::Float64 = 0.5, test_ratio::Float64 = 0.5)

    Train and evaluate a multinomial logistic regression model on the given data.

    # Arguments
    - `X::Matrix{Float64}`: The feature matrix with shape `(n_features, n_samples)`.
    - `labels::Array{Int64}`: The integer labels for each sample.
    - `λ::Float64 = 0.5`: The regularization strength parameter.
    - `test_ratio::Float64 = 0.5`: The ratio of data to use for testing (the rest is used for training).

    # Returns
    - `scores::Float64`: The accuracy of the model on the test set.
    - `params::Matrix{Float64}`: The learned model parameters with shape `(n_features + 1, n_classes)`.

    # Notes
    - The function first converts the labels to a one-hot encoded matrix.
    - Data is standardized using Z-score normalization.
    - NaN values in the data are replaced with 0.
    - The model is trained using multinomial logistic regression with the specified regularization strength.
    - The model is evaluated on the test set and returns both the accuracy score and the learned parameters.
"""
function MultinomialLogisticRegression(
    X::Matrix{Float64},
    labels::Array{Int64};
    λ = 0.5::Float64,
    test_ratio = 0.5,
)
    n_classes = length(Set(labels))
    y = labels_to_y(labels)
    n_features = size(X, 1)

    train, test = make_set_index(length(y), test_ratio)
    @show length(test) + length(train)
    @show length(train)

    train_std = StatsBase.fit(ZScoreTransform, X[:, train], dims = 2)
    StatsBase.transform!(train_std, X)
    intercept = false
    X[isnan.(X)] .= 0

    # deploy MultinomialRegression from MLJLinearModels, λ being the strenght of the reguliser
    mnr = MultinomialRegression(λ; fit_intercept = intercept)
    # Fit the model
    θ = MLJLinearModels.fit(mnr, X[:, train]', y[train])
    # # The model parameters are organized such we can apply X⋅θ, the following is only to clarify
    # Get the predictions X⋅θ and map each vector to its maximal element
    # return θ, X
    preds = MLJLinearModels.softmax(MLJLinearModels.apply_X(X[:, test]', θ, n_classes))
    targets = map(x -> argmax(x), eachrow(preds))
    #and evaluate the model over the labels
    scores = mean(targets .== y[test])
    params = reshape(θ, n_features + Int(intercept), n_classes)
    return scores, params
end

"""
    symbols_to_int(symbols)

    Convert a vector of symbols to a vector of integers and create a mapping dictionary.

    # Arguments
    - `symbols::Vector{Symbol}`: A vector of symbols to be converted to integers.

    # Returns
    - `symbols_int::Vector{Int}`: A vector of integers corresponding to the input symbols.
    - `mapping::Dict{Symbol,Int}`: A dictionary mapping each unique symbol to its corresponding integer.

    # Notes
    - The symbols are first converted to a sorted vector of unique symbols.
    - Each unique symbol is assigned an integer from 1 to n, where n is the number of unique symbols.
    - The mapping dictionary allows for easy lookup of the integer corresponding to any symbol.
"""
function symbols_to_int(symbols)
    v = unique(symbols) |> collect |> sort
    n = length(v)
    mapping = Dict{Symbol,Int}(v[i] => i for i = 1:n)
    symbols_int = zeros(Int, length(symbols))
    for i in eachindex(symbols)
        symbols_int[i] = mapping[symbols[i]]
    end
    return symbols_int, mapping
end

"""
    standardize(data::Matrix, dim = 1)

    Standardize a matrix by subtracting the mean and dividing by the standard deviation.

    # Arguments
    - `data::Matrix`: The input matrix to be standardized.
    - `dim = 1`: The dimension along which to compute the mean and standard deviation (default is 1).

    # Returns
    - `Matrix`: The standardized matrix with zero mean and unit variance along the specified dimension.

    # Notes
    - Uses StatsBase.ZScoreTransform to perform the standardization.
    - The function preserves the original data structure and only returns the transformed data.
"""
function standardize(data::Matrix, dim = 1)
    dt = StatsBase.fit(StatsBase.ZScoreTransform, data, dims = dim)
    return StatsBase.transform(dt, data)
end

"""
    do_pca(data::Matrix)

    Perform Principal Component Analysis (PCA) on the input data.

    # Arguments
    - `data::Matrix`: The input matrix to perform PCA on.

    # Returns
    - `PCA`: A PCA object containing the principal components and other PCA-related information.

    # Notes
    - The input data is first standardized using the `standardize` function.
    - Uses MultivariateStats.fit(PCA, ...) to perform the PCA.
    - The function returns the PCA result object which can be used to transform new data or extract components.
"""
function do_pca(data::Matrix)
    data = standardize(data)
    pca_result = fit(PCA, data;)
    return pca_result
end

export SVCtrain, spikecount_features, sym_features, score_spikes, pca, MultinomialLogisticRegression, symbols_to_int, standardize, do_pca, trial_average

