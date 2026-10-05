using LIBSVM
using CategoricalArrays
using StatsBase
using MultivariateStats
using LinearAlgebra
using StatisticalMeasures
import SNNModels: AbstractPopulation
using ScientificTypes
using MLJ
using Tables
using MLJLinearModels
using DataFrames
using MLJTransforms
using Logging
MultinomialClassifier = MLJ.@load MultinomialClassifier pkg=MLJLinearModels

"""
    SVCtrain(Xs, ys; seed = 123, p = 0.5, labels = false)

Train a linear support-vector classifier (LIBSVM, `cost = 0.01`) on the features `Xs` and
labels `ys` and evaluate it.

A small Gaussian jitter (standard deviation `1e-5`) is added to `Xs`. If `p < 1`, the samples are
split with a stratified partition (fraction `p` for training, RNG seed `seed`), the features are
z-scored with the statistics of the training set, and the classifier is evaluated on the test
set; if either set has fewer than 2 samples, or if `p >= 1`, all samples are used for both
training and testing. `labels` is unused.

# Arguments
- `Xs`: feature matrix of size `(n_features, n_samples)`.
- `ys`: vector of labels (any type; converted to strings), one per sample.

# Returns
`(score, confusion_matrix)`: `score` is Cohen's kappa of the test predictions (not raw accuracy)
and `confusion_matrix` the corresponding `StatisticalMeasures` confusion matrix.

# Example
```julia
using SNNUtils
Xs = hcat(randn(3, 40), randn(3, 40) .+ 3)
ys = vcat(fill(1, 40), fill(2, 40))
kappa, cm = SVCtrain(Xs, ys)
```
"""
function SVCtrain(Xs, ys; seed = 123, p = 0.5, labels = false)
    X = Xs .+ randn(size(Xs)) .* 1e-5
    y = string.(ys)
    y = CategoricalVector(string.(ys))
    @assert length(y) == size(Xs, 2)
    if p < 1
        train, test = partition(eachindex(y), p, rng = seed, stratify = y)
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
    mach = svmtrain(Xtrain, ytrain, kernel = Kernel.Linear, cost=0.01)
    ŷ, decision_values = svmpredict(mach, Xtest);
    confusion_matrix = confmat(ŷ, ytest)
    score = kappa(confusion_matrix)
    return score, confusion_matrix
end

"""
    LogRegtrain(Xs, ys; seed = 123, p = 0.5, gamma = 0, lambda = 0.5)

Multinomial logistic-regression counterpart of [`SVCtrain`](@ref) (MLJLinearModels
`MultinomialClassifier(lambda = lambda, fit_intercept = false)`, inputs standardised with the
training-set statistics). `gamma` is unused. Returns `(score, confusion_matrix)` with Cohen's
kappa. Not exported (`SNNUtils.LogRegtrain`).
"""
function LogRegtrain(Xs, ys; seed = 123, p = 0.5, gamma=0, lambda=0.5)
    X = Xs .+ randn(size(Xs)) .* 1e-5
    X = DataFrame(X', :auto)
    y = CategoricalVector(string.(ys))
    @assert length(y) == size(Xs, 2)

    if p < 1 
        (X_train, X_test), (y_train, y_test) = partition((X,y), p, stratify=y, rng=seed, multi=true)
    else
        X_train = X
        X_test = X
        y_train = y
        y_test = y
    end

    with_logger(ConsoleLogger(stderr, Logging.Warn)) do
         mdl = MultinomialClassifier(lambda=lambda, fit_intercept=false)
         zscore = machine(MLJ.Standardizer(), X_train)
         MLJ.fit!(zscore)
         X_train_std = MLJ.transform(zscore, X_train)
         X_test_std = MLJ.transform(zscore, X_test)
         mach = MLJ.fit!(MLJ.machine(mdl, X_train_std, y_train))
        ypred = MLJ.predict_mode(mach, X_test_std)
        confusion_matrix = confmat(ypred, y_test)
        score = kappa(confusion_matrix)

        return score, confusion_matrix
    end
    # mdl = MultinomialClassifier(;gamma, lambda)
    # zscore = machine(MLJ.Standardizer(), X_train)
    # MLJ.fit!(zscore)
    # X_train_std = MLJ.transform(zscore, X_train)
    # X_test_std = MLJ.transform(zscore, X_test)
    # mach = MLJ.fit!(MLJ.machine(mdl, X_train_std, y_train))

end

"""
    trial_average(array::Array, sequence::Vector, dim::Int = -1)

Average `array` over the trials that share the same label.

`sequence` gives the label of each slice of `array` along dimension `dim` (default: the last
dimension). Returns `(spatial_code, labels)`: `labels = sort(unique(sequence))` and
`spatial_code`, a `Float32` array with the trial dimension removed and a new last dimension of
length `length(labels)` holding the mean over the trials of each label.

Any `dim` works. (Up to SNNUtils 0.2.9 the mean was taken along the last dimension of the
selected slices, which is wrong when `dim` is not the last dimension.)

# Example
```julia
using SNNUtils
code, labels = trial_average(rand(4, 6), [1, 2, 1, 2, 1, 2])   # size(code) == (4, 2)
```
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
        selectdim(spatial_code, ave_dim, i) .= dropdims(
            mean(selectdim(array, trial_dim, sound_ids), dims = trial_dim),
            dims = trial_dim,
        )
    end
    return spatial_code, labels
end

"""
    trial_sort(array::Array, sequence::Vector, dim::Int = -1)

Group the slices of `array` along `dim` (default: last dimension) by label. Returns
`(data, labels)`: `data[label]` is a vector of copies of the slices with that label, and
`labels = sort(unique(sequence))`. The dictionary is typed `Dict{Symbol, Vector{Array}}`, so
labels must be convertible to `Symbol`.
"""
function trial_sort(array::Array, sequence::Vector, dim::Int = -1)
    trial_dim = dim < 0 ? ndims(array) : dim
    labels = unique(sequence) |> sort

    data = Dict{Symbol,Vector{Array}}()
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

Spike counts of every neuron of `pop` in each time window of `offsets`.

`offsets` is a vector of windows accepted by `spiketimes(pop; interval)` (e.g. `[t0, t1]` in ms
or a range). Requires a `:fire` record. Windows are processed in parallel with
`Threads.@threads`.

# Returns
`Matrix{Float64}` of size `(pop.N, length(offsets))`.

# Example
```julia
using SpikingNeuralNetworks, SNNUtils
SNN.@load_units
E = SNN.Poisson(N = 10, param = SNN.PoissonParameter(10Hz))
SNN.monitor!(E, [:fire])
SNN.sim!([E]; duration = 1s)
X = spikecount_features(E, [[0ms, 500ms], [500ms, 1000ms]])
```
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
    sym_features(sym::Symbol, pop::T, offsets::Vector) where {T<:AbstractPopulation}

Mean of the recorded variable `sym` of every neuron of `pop` in each window
`offset = [t0, t1]` of `offsets` (sampled every 1 ms from the interpolated record). Windows that
end after the last recorded time are skipped (their column stays zero). Returns a
`Matrix{Float64}` of size `(pop.N, length(offsets))`.

(Up to SNNUtils 0.2.9 it called `SNN.record`, undefined inside SNNUtils, and always threw.)
"""
function sym_features(sym::Symbol, pop::T, offsets::Vector) where {T<:AbstractPopulation}
    N = pop.N
    X = zeros(N, length(offsets))
    var, r_v = SNNModels.record(pop, sym, range = true)
    Threads.@threads for i in eachindex(offsets)
        offset = offsets[i]
        offset[end] > r_v[end] && continue
        range = offset[1]:1ms:offset[2]
        X[:, i] = mean(var(axes(var, 1), range), dims = 2)[:, 1]
    end
    return X
end

"""
    score_spikes(model, seq, target_interval = :offset; delay = nothing, pop = :Exc)

Decode which word was presented from the spike counts of the word assemblies and score the
decoding.

The word assemblies are the neurons targeted by the stimuli of `model.stim` whose names start with
`w_` (stimuli whose name contains `"noise"` are excluded), obtained with `subpopulations`. For each
element of `seq` of type `target_interval` (row `seq.line_id.type`, default `:offset`), the spikes
of `model.pop[pop]` are counted in 10 ms bins over the window `offset_time .+ (0:100) .+ delay`
(ms), and the predicted word is the assembly with the largest mean count.

# Arguments
- `model`: model with `pop` and `stim`; `seq`: sequence from [`generate_sequence`](@ref).
- `target_interval = :offset`: element type used as decoding time.
- `delay = nothing`: delay (ms) of the decoding window; if `nothing`, the delays `-100:10:100`
  are all tested.
- `pop = :Exc`: population to decode from.

# Returns
- With `delay`: `(score, confusion_matrix, activity_matrix)`, where `score` is Cohen's kappa and
  `activity_matrix[i, j]` the mean count of assembly `i` during word `j`, averaged over the
  presentations of `j`.
- Without `delay`: `(scores, best_delay, (; cms, delays))`.

The computation is serial. `activity_matrix` is computed for the given `delay`. (Up to SNNUtils
0.2.9 it accumulated over all the delays tested.)
"""
function score_spikes(model, seq, target_interval = :offset; delay = nothing, pop = :Exc)
    ## Get word intervals 
    offsets_ids = findall(seq.sequence[seq.line_id.type, :] .== target_interval)
    words = seq.sequence[seq.line_id.words, offsets_ids]
    offset_times = seq.sequence[seq.line_id.offset, offsets_ids]

    if isempty(offsets_ids)
        throw("No target intervals found in sequence")
    end
    ## Get word assemblies
    all_populations = subpopulations(filter(x->!occursin("noise", x.name), model.stim))
    word_assemblies = Dict{Symbol, Vector{Int}}()
    for label in keys(all_populations)
        word = string(label)
        if startswith(word, "w_")
            word_assemblies[Symbol(word[3:end])] = all_populations[label]
        end
    end
    word_assemblies = word_assemblies |> dict2ntuple

    word_list = keys(word_assemblies) |> collect |> sort
    word_count = [count(x->x==word, words) for word in word_list]
    assemblies = [word_assemblies[word] for word in word_list]

    confusion_matrix = zeros(Float32, length(word_assemblies), length(word_assemblies))
    activity_matrix = zeros(Float32, length(word_assemblies), length(word_assemblies))
    _spikes = spiketimes(model.pop[pop])
    spike_count, r = bin_spiketimes(_spikes, interval = 0:10ms:(offset_times[end]+100ms))

    function _score(delay, test_interval = 0:100)
        predicted = Symbol[]
        target = Symbol[]
        fill!(activity_matrix, 0.0f0)
        @inbounds @fastmath for i in eachindex(offsets_ids)
            target_word = findfirst(word_list .== words[i])
            target_interval = offset_times[i] .+ test_interval .+ delay
            r_idx = findall(
                x->(r[x]>target_interval[1] && r[x]<target_interval[end]),
                eachindex(r),
            )
            _spikes = sum(spike_count[:, r_idx], dims = 2)[:, 1]

            for word in eachindex(word_list)
                activity_matrix[word, target_word] +=
                    mean(_spikes[assemblies[word]]) / word_count[target_word]
            end
            push!(
                predicted,
                word_list[argmax(mean.([_spikes[assembly] for assembly in assemblies]))],
            )
            push!(target, word_list[target_word])
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
    MultinomialLogisticRegression(X::Matrix{Float64}, labels::Array{Int64}; λ = 0.5, test_ratio = 0.5)

Multinomial logistic regression (MLJLinearModels `MultinomialRegression(λ)`, no intercept) of
`labels` on the features `X` (`(n_features, n_samples)`), with z-scoring based on a random
training set (a fraction `1 - test_ratio` of the samples, drawn with the global RNG) and `NaN`
replaced by 0. Returns `(accuracy, params)`: the accuracy on the test samples and `params` of
size `(n_features, n_classes)`. `X` is not modified.

(Up to SNNUtils 0.2.9 it called the undefined `make_set_index` and always threw; it also
z-scored `X` in place and printed with `@show`.)
"""
function MultinomialLogisticRegression(
    X::Matrix{Float64},
    labels::Array{Int64};
    λ = 0.5::Float64,
    test_ratio = 0.5,
)
    n_classes = length(Set(labels))
    y, mapping = symbols_to_int(Symbol.(labels))
    n_features = size(X, 1)

    train, test = _make_set_index(length(y), test_ratio)

    X = copy(X)
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

# Random split of `1:n` into training and test indices (`test_ratio` of the samples in the test set).
function _make_set_index(n::Int, test_ratio)
    perm = randperm(n)
    n_test = clamp(round(Int, test_ratio * n), 1, n - 1)
    return sort(perm[(n_test+1):end]), sort(perm[1:n_test])
end

"""
    symbols_to_int(symbols)

Map symbols to integers `1:n` following the sorted order of the unique symbols.

# Returns
`(symbols_int, mapping)`: the vector of integers and the `Dict{Symbol,Int}` used.

# Example
```julia
using SNNUtils
symbols_to_int([:b, :a, :b])   # ([2, 1, 2], Dict(:a => 1, :b => 2))
```
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

Z-score `data` with `StatsBase.ZScoreTransform` fitted with `dims = dim` and return the
transformed copy (the input is not modified).
"""
function standardize(data::Matrix, dim = 1)
    dt = StatsBase.fit(StatsBase.ZScoreTransform, data, dims = dim)
    return StatsBase.transform(dt, data)
end

"""
    do_pca(data::Matrix)

Standardise `data` with [`standardize`](@ref) (dimension 1), fit a PCA with
`MultivariateStats.fit(PCA, data)` (observations are the columns) and return the projected data
`MultivariateStats.transform(pca, data)`, a matrix of size `(n_components, n_samples)`. The PCA
object itself is not returned.
"""
function do_pca(data::Matrix)
    data = standardize(data)
    pca_result = MultivariateStats.fit(PCA, data;)
    return MultivariateStats.transform(pca_result, data)
end

export SVCtrain,
    spikecount_features,
    sym_features,
    score_spikes,
    MultinomialLogisticRegression,
    symbols_to_int,
    standardize,
    do_pca,
    trial_average
