using LIBSVM
using MLJ
using CategoricalArrays
using StatsBase
    
"""
    SVCtrain(Xs, ys; seed=123, p=0.6)
    
    Train a Support Vector Classifier with a linear kernel on the data Xs and labels ys.

    # Arguments
    - `Xs::Matrix{Float32}`: The data matrix with shape `(n_features, n_samples)`.
    - `ys::Vector{Int64}`: The labels.

    # Returns
    The accuracy of the classifier on the test set.


"""
function SVCtrain(Xs, ys; seed=123, p=0.6)
    X = Xs .+ 1e-1
    y = string.(ys)
    y = CategoricalVector(string.(ys))
    @assert length(y) == size(Xs, 2)
    train, test = partition(eachindex(y), p, rng=seed)

    ZScore = fit(StatsBase.ZScoreTransform, X[:,train], dims=2)
    Xtrain = StatsBase.transform(ZScore, X[:,train])
    Xtest = StatsBase.transform(ZScore, X[:,test])
    ytrain = y[train]
    ytest = y[test]

    @assert size(Xtrain, 2) == length(ytrain)
    # classifier = svmtrain(Xtrain, ytrain)
    SVMClassifier = MLJ.@load SVC pkg=LIBSVM
    svm = SVMClassifier(;kernel=LIBSVM.Kernel.Linear)
    mach = machine(svm, Xtrain', ytrain, scitype_check_level=0) |> MLJ.fit!

    # Test model on the other half of the data.
    ŷ = MLJ.predict(mach, Xtest');
    # ŷ, classes = svmpredict(classifier, Xtest);
    
    # @info "Accuracy: $(mean(ŷ .== ytest) * 100)"
    return mean(ŷ .== ytest)
end

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
    score_activity(model, seq, interval=[0ms, 100ms])

Compute the most active population in a given interval with respect to the offset time of the input presented, then compute the confusion matrix.

The function computes the activity of the spiking neural network model for each symbol in the sequence and get the symbol with maximum activity. It then updates the confusion matrix accordingly.

The function computes the activity of the spiking neural network model for each symbol in the sequence and get the symbol with maximum activity. It then updates the confusion matrix accordingly.

## Arguments
- `model`: The spiking neural network model, containg the target.
- `seq`: The sequence of symbols to be recognized.
- `interval`: The time interval during which the activity is measured. Default is `[0ms, 100ms]`.
- `pop`: The population whose spike will be computed. Default is `:E`.

## Returns
- `confusion_matrix`: The confusion matrix, normalized by the number of occurrences of each symbol in the sequence. The matrix has (predicted x true) dimensions.
"""
function score_activity(model, seq, interval=[0ms, 100ms]; pop=:E)
    offsets, ys = all_intervals(:words, seq, interval=interval)
    S = spikecount_features(getfield(model.pop,pop), offsets)
    confusion_matrix= zeros(length(seq.symbols.words), length(seq.symbols.words))
    activity = zeros(length(seq.symbols.words))
    occurences = zeros(length(seq.symbols.words))
    for y in eachindex(ys)
        word = ys[y]
        word_id = findfirst(==(word), seq.symbols.words)
        occurences[word_id] += 1
        for w in eachindex(seq.symbols.words)
            word_test = seq.symbols.words[w]
            cells = getstim(model.stim, word_test, :d).cells
            activity[w] = mean(S[cells, y])
        end
        activated = argmax(activity)
        confusion_matrix[activated, word_id] += 1
    end
    return confusion_matrix./occurences'
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



export SVCtrain, spikecount_features, sym_features, score_activity