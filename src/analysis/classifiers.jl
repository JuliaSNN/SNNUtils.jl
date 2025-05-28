using LIBSVM
using MLJ
using CategoricalArrays
using StatsBase
using MultivariateStats
using LinearAlgebra
using StatisticalMeasures
    
"""
    SVCtrain(Xs, ys; seed=123, p=0.6)
    
    Train a Support Vector Classifier with a linear kernel on the data Xs and labels ys.

    # Arguments
    - `Xs::Matrix{Float32}`: The data matrix with shape `(n_features, n_samples)`.
    - `ys::Vector{Int64}`: The labels.

    # Returns
    The accuracy of the classifier on the test set.


"""
function SVCtrain(Xs, ys; seed=123, p=0.5, labels=false)
    X = Xs .+ 1e-2
    y = string.(ys)
    y = CategoricalVector(string.(ys))
    @assert length(y) == size(Xs, 2)
    if p <1 
        train, test = partition(eachindex(y), p, rng=seed, stratify=y)
        # n = 1
        if length(train)<2 || length(test)<2
            @error "Not enough samples in train or test set"
            Xtrain = X
            Xtest = X
            ytrain = y
            ytest = y
        else
            ZScore = StatsBase.fit(StatsBase.ZScoreTransform, X[:,train], dims=2)
            Xtrain = StatsBase.transform(ZScore, X[:,train])
            Xtest = StatsBase.transform(ZScore, X[:,test])
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
    mach = svmtrain(Xtrain, ytrain, kernel=Kernel.Linear)
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

function trial_average(array::Array, sequence::Vector, dim::Int=-1)
    trial_dim = dim < 0 ? ndims(array) : dim
    my_dims = collect(size(array))
    popat!(my_dims, trial_dim)
    labels = unique(sequence) |> sort
    spatial_code = zeros(Float32, my_dims..., length(labels))
    ave_dim = ndims(spatial_code)

    for i in eachindex(labels)
        sound = labels[i]
        sound_ids = findall(==(sound), sequence)
        selectdim(spatial_code, ave_dim, i) .= mean(selectdim(array, trial_dim, sound_ids), dims=ave_dim)
    end
    return spatial_code, labels
end

export trial_average

    # try
            # machine_loaded = false
            # mach = nothing
    #     # SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
    #     # svm = LIBSVM.SVC() # Use the scikit-like interface
    #     # svm = SVMClassifier(kernel=LIBSVM.Kernel.Linear)
    #     # MLJ.fit!(mach, verbosity=0)
    #     # mach = machine(svm, Xtrain', ytrain, scitype_check_level=0) 
    # catch
    #     @warn "SVC not loaded -  wait 5s"
    #     sleep(5)
    #     return
    # end

    # ŷ, classes = svmpredict(classifier, Xtest);
    
    # @info "Accuracy: $(mean(ŷ .== ytest) * 100)"

"""
    spikecount_features(pop::T, offsets::Vector)  where T <: AbstractPopulation

    Return a matrix with the spike count of each neuron in the population `pop` for each offset in `offsets`.
    
    # Arguments
    - `pop::T`: The population.
    - `offsets::Vector`: The time offsets.

    # Returns
    Matrix::Float32: The spike count matrix (n_features x n_samples).
    
"""
function spikecount_features(pop::T, offsets::Vector)  where T <: SNN.AbstractPopulation
    N = pop.N
    X = zeros(N, length(offsets))
    Threads.@threads for i in eachindex(offsets)
        offset = offsets[i]
        X[:,i] = length.(spiketimes(pop, interval = offset))
    end
    return X
end



"""
    sym_features(sym::Symbol, pop::T, offsets::Vector) where T <: AbstractPopulation

    Return a matrix with the mean of the interpolated record of the symbol `sym` in the population `pop` for each offset in `offsets`.

    # Arguments
    - `sym::Symbol`: The symbol.
    - `pop::T`: The population.
    - `offsets::Vector`: The time offsets.

    # Returns
    Matrix::Float32: The feature matrix (n_features x n_samples).
"""
function sym_features(sym::Symbol, pop::T, offsets::Vector) where T <: SNN.AbstractPopulation
    N = pop.N
    X = zeros(N, length(offsets))
    var, r_v = SNN.interpolated_record(pop, sym)
    Threads.@threads for i in eachindex(offsets)
        offset = offsets[i]
        offset[end] > r_v[end] && continue
        range = offset[1]:1ms:offset[2]
        X[:,i] = mean(var[:, range], dims=2)[:,1]
    end
    return X
end

"""
    score_spikes(model, seq, interval=[0ms, 100ms])

Compute the most active population in a given interval with respect to the offset time of the input presented, then compute the confusion matrix.

The function computes the activity of the spiking neural network model for each symbol in the sequence and get the symbol with maximum activity. It then updates the confusion matrix accordingly.

The function computes the activity of the spiking neural network model for each symbol in the sequence and get the symbol with maximum activity. It then updates the confusion matrix accordingly.

## Arguments
- `model`: The spiking neural network model, containg the target.
- `seq`::NamedTuple the sequence object with the order of presentations. It must contains:
    - `line_id`: The line id of the seq.sequence.
    - `sequence`: An array with: [words, offset, interval_type, onset_time, offset_time]. 
- `target_interval`::Symbol The type of interval to be used for the target. Default is `:offset`.
- `pop`: The population whose spike will be computed. Default is `:E`.

## Returns
- `score`: The score of the model, which is the difference between the activity of the most active population and a random score.
- `confusion_matrix`: The confusion matrix, normalized by the number of occurrences of each symbol in the sequence. The matrix has (predicted x true) dimensions.
"""
function score_spikes(model, seq, target_interval=:offset, delay=nothing, pop=:E)
    ## Get word intervals 
    offsets_ids = findall(seq.sequence[seq.line_id.type,:].==target_interval)
    words = seq.sequence[seq.line_id.words, offsets_ids]
    offset_times = seq.sequence[seq.line_id.offset, offsets_ids]

    if isempty(offsets_ids) 
        throw("No target intervals found in sequence")
    end
    ## Get word assemblies
    labels, neurons = subpopulations(filter(x->!occursin("noise",x.name), model.stim))
    word_assemblies = Dict(Symbol(labels[n][3:end])=> neurons[n] for n in eachindex(labels) if startswith(labels[n], "w_")) |> dict2ntuple
    word_list = keys(word_assemblies) |> collect |> sort
    word_count = [count(x->x==word, words) for word in word_list]
    assemblies = [word_assemblies[word] for word in word_list]

    my_lock = Threads.SpinLock()
    confusion_matrix = zeros(Float32, length(word_assemblies), length(word_assemblies))
    activity_matrix  = zeros(Float32, length(word_assemblies), length(word_assemblies))
    _spikes = spiketimes(model.pop[pop])
    spike_count, r = bin_spiketimes(_spikes, interval=0:10ms:offset_times[end]+100ms)

    function _score(delay, test_interval = 0:100)
        predicted = Symbol[]
        target = Symbol[]
        @inbounds @fastmath for i in eachindex(offsets_ids)
            target_word = findfirst(word_list.==words[i])
            target_interval= offset_times[i] .+ test_interval .+ delay
            r_idx = findall(x->(r[x]>target_interval[1] && r[x]<target_interval[end]), eachindex(r))
            _spikes = sum(spike_count[:, r_idx], dims=2)[:,1]

            lock(my_lock)
            for word in eachindex(word_list)
                activity_matrix[word, target_word] += mean(_spikes[assemblies[word]]) / word_count[target_word]
            end
            push!(predicted, word_list[argmax(mean.([_spikes[assembly] for assembly in assemblies]))])
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
        delays= -100:10:100
        scores = Vector{Float32}(undef, length(delays))
        cms = Vector{Any}(undef, length(delays))
        for i in eachindex(delays)
            score, cm, _ = _score(delays[i])
            scores[i] = score
            cms[i] = cm
        end
        best_score = argmax(scores)
        best_delay = delays[best_score]
        return scores, best_delay, (;cms, delays)
    end
end




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

function symbols_to_int(symbols)
    v = unique(symbols) |> collect |> sort
    n = length(v)
    mapping = Dict{Symbol, Int}(v[i] => i for i in 1:n)
    symbols_int = zeros(Int, length(symbols))
    for i in eachindex(symbols)
        symbols_int[i] = mapping[symbols[i]]
    end
    return symbols_int, mapping
end

function standardize(data::Matrix, dim=1)
    dt = StatsBase.fit(StatsBase.ZScoreTransform, data, dims=dim)
    return StatsBase.transform(dt, data)
end

# Function to perform PCA and return the PCA matrix and principal components
function do_pca(data::Matrix)
    data = standardize(data, )
    pca_result = fit(PCA, data;)
    return pca_result
end


export SVCtrain, spikecount_features, sym_features, score_spikes, pca