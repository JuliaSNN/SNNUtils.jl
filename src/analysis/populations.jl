using ProgressBars

# struct Population
#     name::String
#     neurons::Vector{Int64}
#     voltage::Vector{Matrix{Float64}}
#     spikes::Vector{Vector{Float64}}
#     firing_times::Vector{Float64}
#     activity::Vector{Vector{Float64}}
#     mean_activity::Float64
#     std_activity::Float64
#     mean_rate::Float64
#     std_rate::Float64
# end
"""
    PopActivity
    Type used to store the spikes of a population for each word or phoneme in the sequence.
    Fields:
    ------
        spiketimes::Matrix{Vector{Spiketimes}}
            The spikes of the population for each word or phoneme
        pops::Vector{Int}
            The indices of the neurons belonging to each populations
        intervals::Vectro{Float32}
            The average duration of the intervals, for each target word
        targets::Vector{Union{String,Char}}
            The target words for each word_spikes row
"""
# PopActivity = NamedTuple{
#     (:spiketimes, :membrane, :pops, :intervals, :targets, :n_intervals),
#     Tuple{
#         Matrix{Vector{Spiketimes}},
#         Matrix{Vector{Vector{Float32}}},
#         Vector{Vector{Int}},
#         Vector{Vector{Float32}},
#         Vector{Union{String,Char}},
#         Vector{Int},
#     },
# }
# @kw_ struct Population 

"""
    population_spikes(sign::String, spikes::Vector{NNSpikes}, seq::Encoding; delay=0, unique=false, kwargs...)

Return the spikes of the population for each word or phoneme in the sequence.
    Parameters
    ----------
    sign::String
        "words" or "phonemes"
    spikes::Vector{NNSpikes} 
        The spikes of the network  
    seq::Encoding
        The sequence of the experiment
    delay::Int
        The delay between the target and the testing population
    unique::Bool
        If true, the population is unique for each word or phoneme
    kwargs
        Keyword arguments for the merge_spiketimes function
    Returns 
    -------
    NamedTuple with fields:
        spiketimes::Matrix{Vector{Spiketimes}}
            The spikes of the population for each word or phoneme
            [the first index is the target population,
            the second index is the measured population]
        pops::Vector{Int}
            The indices of the neurons belonging to each populations
        intervals::
            The average duration of the intervals, for each target word
        targets::Vector{Union{String,Char}}
            The target words for each word_spikes row

"""
function population_spikes(
    sign::String,
    spikes::Vector{NNSpikes},
    seq::Encoding;
    delay = 0,
    unique = false,
    kwargs...,
)#::PopActivity
    isa(delay, Vector) || (delay = [delay, delay])

    if sign == "crossed"
        target_indices = get_sign_indices("phonemes", seq)
        interval_indices = get_sign_indices("words", seq)
    elseif (sign == "words") || (sign == "phonemes")
        target_indices = get_sign_indices(sign, seq)
        interval_indices = get_sign_indices(sign, seq)
    elseif sign == "all"
        target_indices = get_indices(seq)
        interval_indices = get_indices(seq)
    end
    target_spikes =
        Matrix{Vector{Spiketimes}}(undef, length(interval_indices), length(target_indices))
    pops = populations(target_indices, seq, unique)
    spiketimes = merge_spiketimes(spikes; pop = :ALL, cache = true)
    intervals = Vector{Vector{Vector{Float32}}}(undef, length(interval_indices))

    for (n_i, n) in enumerate(interval_indices)
        target_intervals, _ = get_intervals(n, seq)
        intervals[n_i] = target_intervals
        @debug "Target: $(seq.mapping[n]) intervals: $(length(target_intervals))"
        for (m_i, m) in enumerate(target_indices)
            all_pop_spikes = Vector{Spiketimes}()
            for interval in target_intervals
                selected = get_spikes_in_interval(
                    spiketimes[pops[m_i]],
                    [interval[1] + delay[1], interval[2] + delay[2]],
                )
                # @show selected
                push!(all_pop_spikes, relative_time!(selected, interval[1]))
            end
            target_spikes[n_i, m_i] = deepcopy(all_pop_spikes)
        end
    end
    targets = [seq.mapping[target_index] for target_index in target_indices]
    interval_ids = [seq.mapping[target_index] for target_index in interval_indices]
    interval_durations = [[0.0f0, mean(diff.(interval))[1]] for interval in intervals]

    return (
        spiketimes = target_spikes,
        pops = pops,
        intervals_ids = interval_ids,
        intervals = interval_durations,
        targets = targets,
        n_intervals = length.(intervals),
    )
end

function population_membrane(
    sign::String,
    trackers::Vector{NNTracker},
    seq::Encoding;
    delay = 0,
    unique = false,
    targets = nothing,
    kwargs...,
)#::PopActivity

    isa(delay, Vector) || (delay = [delay, delay])


    if sign == "crossed"
        target_indices = get_sign_indices("phonemes", seq)
        interval_indices = get_sign_indices("words", seq)
    elseif (sign == "words") || (sign == "phonemes")
        target_indices = get_sign_indices(sign, seq)
        interval_indices = get_sign_indices(sign, seq)
    elseif sign == "all"
        target_indices = get_indices(seq)
        interval_indices = get_indices(seq)
    elseif sign == "targets"
        @assert !isnothing(targets)
        target_indices = targets.target_indices
        interval_indices = targets.interval_indices
    end
    # # target_spikes = Matrix{Vector{Spiketimes}}(undef, length(interval_indices), length(target_indices))


    pops = populations(target_indices, seq, unique)
    track_pops = [indexin(pop, read(trackers[1], :track_neurons)) for pop in pops]
    intervals = Vector{Vector{Vector{Float32}}}(undef, length(interval_indices))

    # mem = TNN.get_membrane_traces(trackers, target_intervals[1].+[0,150], neurons).+TNN.get_membrane_traces(trackers, target_intervals[30].+[0,150], neurons)

    ## Preliminaries
    n_pops = maximum(length.(track_pops))
    n_ints = minimum([
        length(get_intervals(target_index, seq)[1]) for target_index in word_indices(seq)
    ])
    _int_length = round(
        Int,
        maximum(
            vcat(
                [
                    diff.(get_intervals(target_index, seq)[1]) for
                    target_index in interval_indices
                ]...,
            ),
        )[1] .+ delay[2],
    )
    all_membrane = Array{Float32,6}(
        undef,
        n_pops,
        4,
        _int_length,
        n_ints,
        length(target_indices),
        length(interval_indices),
    )
    all_intervals = vcat([TNN.get_intervals(x, seq)[1] for x in interval_indices]...)
    sorted_indices = sort(1:length(all_intervals), by = x -> all_intervals[x][1])
    last_tt = trackers[end].tt
    all_intervals = all_intervals[sorted_indices]
    all_intervals = [l for l in all_intervals if (l[2] + delay[2]) < last_tt]

    all_interval_targets = vcat(
        [
            repeat(
                [findfirst(x .== interval_indices)],
                length(TNN.get_intervals(x, seq)[1]),
            ) for x in interval_indices
        ]...,
    )[sorted_indices]
    @info size(all_interval_targets)

    int_counter = zeros(Int, length(interval_indices))
    # @info "Interval: $(_int_length[1]), \nneurons: $(n_pops), \ninterval presented: $(n_ints) \nlast interval: $(all_intervals[end])"
    for k in ProgressBar(eachindex(all_intervals))
        n = all_interval_targets[k]
        int_counter[n] += 1
        int_counter[n] > n_ints && continue
        target_interval = all_intervals[k] .+ delay
        # @info "Target $n: $([seq.mapping[target_index] for target_index in target_indices]) 
        # interval: $target_interval; id: $(int_counter[n])"
        all_pops = vcat(track_pops...)
        # @show target_interval, all_pops
        selected = get_membrane_traces(trackers, target_interval, all_pops)[:, :, 2:end]
        Threads.@threads for m in eachindex(target_indices)
            ii = indexin(track_pops[m], all_pops)
            @views all_membrane[
                1:size(track_pops[m], 1),
                :,
                1:size(selected, 3),
                int_counter[n],
                m,
                n,
            ] .= selected[ii, :, :]
        end
    end
    targets = [seq.mapping[target_index] for target_index in target_indices]
    interval_durations = [diff(interval) for interval in all_intervals]
    @info "return population"
    GC.gc()
    @strdict membranes = all_membrane pops = pops intervals = interval_durations targets =
        targets n_intervals = int_counter
end



"""
    score_population_activity(score_function::Function,sign::String, spikes::Vector{NNSpikes}, seq::Encoding; delay=0, unique=false, kwargs...)

Return the confusion matrix of the population activity for each word or phoneme in the sequence.
    Parameters
    ----------
    score_function::Function
        The function to score the population activity (e.g. get_fr)
        The function must take as input the spiketimes of the population (::Spiketimes) and the interval (::Vector{Float32})
    sign::String
        "words" or "phonemes"
    spikes::Vector{NNSpikes} 
        The spikes of the network  
    seq::Encoding
        The sequence of the experiment
    delay::Int
        The delay between the target and the testing population
    unique::Bool
        If true, select the neurons that are unique for each word or phoneme
    kwargs
        Keyword arguments for the score_function
    Returns 
    -------
    confusion_matrix::Matrix{Float64}
        The confusion matrix of the population activity for each word or phoneme in the sequence
    indices::Vector{Int}
        The indices of the words or phonemes

"""
function score_population_activity(
    score_function::Function,
    sign::String,
    spikes::Union{Vector{NNSpikes},Spiketimes},
    seq::Encoding;
    delay = 0,
    unique = false,
    intervaltype = :rp,
    kwargs...,
)
    @unpack duration = seq
    @assert intervaltype in [:full, :rp, :last, :first, :mid, :second, :third]

    recognition_points = get_recognition_points(seq, sign, sorted = true)

    indices =
        sign == "words" ? words(seq, sorted = true, indices = true) :
        phonemes(seq, sorted = true, indices = true)
    interval_indices = copy(indices)
    target_indices = copy(indices)
    target_populations = Vector{Spiketimes}()
    if isa(spikes, Vector{NNSpikes})
        spiketimes = merge_spiketimes(spikes; pop = :ALL, cache = true)
    else
        spiketimes = spikes
    end
    for target_index in interval_indices
        if isa(unique, Bool)
            if unique
                pop = unique_population(target_index, seq)
            else
                pop = seq.populations[target_index]
            end
        elseif unique == :inverse
            un_pop = Set(unique_population(target_index, seq))
            all_pop = Set(seq.populations[target_index])
            pop = collect(setdiff(all_pop, un_pop))
        else
            throw(ErrorException("Set 'unique' to true, false or :inverse"))
        end
        if !isempty(pop)
            push!(target_populations, spiketimes[pop])
        else
            popat!(target_indices, findfirst(target_indices .== target_index))
        end
    end
    # @assert length(target_populations) == length(indices)

    confusion_matrix = zeros(length(target_indices), length(interval_indices))

    for n in eachindex(indices)
        target_index = indices[n]
        intervals, _ = get_intervals(target_index, seq)
        scores = zeros(length(target_indices))
        @inbounds @simd for my_interval in intervals
            if intervaltype == :rp
                target_rp = recognition_points[n]
                interval =
                    duration .* [target_rp - 1, (target_rp)] .+ my_interval[1] .+ delay
            elseif intervaltype == :first
                interval = duration .* [0, 1] .+ my_interval[1] .+ delay
            elseif intervaltype == :second
                interval = duration .* [1, 2] .+ my_interval[1] .+ delay
            elseif intervaltype == :third
                interval = duration .* [2, 3] .+ my_interval[1] .+ delay
            elseif intervaltype == :mid
                interval =
                    my_interval[1] .+ diff(my_interval) / 2 .+ duration * [-0.5, +0.5] .+
                    delay
            elseif intervaltype == :last
                interval = duration .* [-1, 0] .+ my_interval[2] .+ delay
            elseif intervaltype == :full
                interval = my_interval .+ delay
            end
            for m in eachindex(target_indices)
                selected =
                    get_spikes_in_interval(target_populations[m], interval, collapse = true)
                scores[m] = score_function(selected, interval; kwargs...)
            end
            # @show scores
            confusion_matrix[argmax(scores), n] += 1
        end
        confusion_matrix[:, n] ./= length(intervals)
    end
    return confusion_matrix, indices
end

# Define a function to compute precision, recall, and F1 score from a normalized confusion matrix
function compute_f1_score(confusion_matrix::Matrix{Float64})
    # Get the number of classes (assumed square matrix)
    num_classes = size(confusion_matrix, 1)
    
    # Initialize arrays to store precision, recall, and F1 score for each class
    precision = zeros(Float64, num_classes)
    recall = zeros(Float64, num_classes)
    f1_score = zeros(Float64, num_classes)
    
    # Compute precision, recall, and F1 score for each class
    for i in 1:num_classes
        TP = confusion_matrix[i, i]  # True Positive for class i
        FP = sum(confusion_matrix[:, i]) - TP  # False Positive for class i
        FN = sum(confusion_matrix[i, :]) - TP  # False Negative for class i
        
        # Calculate precision and recall
        precision[i] = TP / (TP + FP)
        recall[i] = TP / (TP + FN)
        
        # Calculate F1 score (handling case where precision + recall = 0)
        if precision[i] + recall[i] > 0
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else
            f1_score[i] = 0.0  # If both precision and recall are 0, set F1 score to 0
        end
    end
    
    (precision=precision, recall=recall, f1=f1_score)
end

function compute_precision(confusion_matrix::Matrix{Float64})
    # Get the number of classes (assumed square matrix)
    num_classes = size(confusion_matrix, 1)
    
    # Initialize an array to store precision for each class
    precision = zeros(Float64, num_classes)
    
    # Compute precision for each class
    for i in 1:num_classes
        TP = confusion_matrix[i, i]  # True Positive for class i
        FP = sum(confusion_matrix[i,:]) - TP  # False Positive for class i
        precision[i] = TP / (TP + FP)
    end
    
    return precision
end

score_population_activity(network; unique = true, sign = "words", kwargs...) =
    score_population_activity(
        get_mean_fr,
        sign,
        load_spikes(network.store.data),
        network.seq;
        delay = 35,
        intervaltype = :rp,
        unique = unique,
        kwargs...,
    )

## Compute the Cohen's kappa coefficient, which is a measure of inter-rater agreement
""""
    Cohen_kappa(M::Matrix)

Return the Cohen's kappa coefficient, which is an unbiased score estimation.
    Parameters
    ----------
    M::Matrix
        The confusion matrix
    Returns 
    -------
    kappa::Float64
        The Cohen's kappa coefficient

"""
function Cohen_kappa(M::Matrix)
    n = sum(M)
    p0 = sum(diag(M)) / n
    # predicted
    A = sum(M, dims = 2)
    # presented 
    B = sum(M, dims = 1)
    pe = sum(A' .* B) / n^2
    kappa = (p0 - pe) / (1 - pe)
    return round(kappa, digits = 3)
end

"""
    k_fold(vector, k, do_shuffle=false)

    Return the indices of the k-fold repartition of a Vector.
    Used for distributing lists over threads.
    Parameters
    ----------
    vector::Vector
        The vector to be distributed
    k::Int 
        The number of folds
    do_shuffle::Bool
        If true, shuffle the vector before distributing
    Returns
    -------
    indices::Vector{Vector{Int}}
        The indices of the k-fold repartition of a Vector
"""
function k_fold(vector, k, do_shuffle = false)
    if do_shuffle
        ns = shuffle(1:length(vector))
    else
        ns = 1:length(vector)
    end
    b = length(ns) ÷ k
    indices = Vector{Vector{Int}}()
    for i = 1:k-1
        push!(indices, ns[(i-1)*b+1:b*i])
    end
    push!(indices, ns[1+b*(k-1):end])
    return indices
end

"""
    reactivation_statistics(word_spikes,words )

    Return the reactivation statistics for each target-test couple .
    The reactivation statistics consists of:
    - the mean reactivation time
    - the mean reactivation load
    Parameters
    ----------
    word_spikes::Matrix{IntervalsSpiketimes}
        The spiketimes for each target-test couple. The spiketimes are organized in intervals.
        The spiketimes must have the relative time to the start of the interval.
    words::Vector{String}
        List of the words
    Returns
    -------
    reactivation_time::Matrix{Vector{Float64}}
        The mean reactivation time for each interval of each target-test couple
    reactivation_load::Matrix{Vector{Float64}}
        The mean reactivation load for each interval of each target-test couple
"""
function reactivation_data(word_spikes)
    @unpack spiketimes, pops, intervals = word_spikes
    reactivation_time = Matrix{Vector{}}(undef, length(pops), length(pops))
    reactivation_load = Matrix{Vector{}}(undef, length(pops), length(pops))

    for m in eachindex(pops)
        for n in eachindex(pops)
            fts = Vector{Vector{Float32}}([[] for ii in eachindex(pops[m])])
            loadings = Vector{Vector{Float32}}([[] for ii in eachindex(pops[m])])
            for ii in eachindex(spiketimes[n, m]) # loop over intervals
                st = spiketimes[n, m][ii]
                spike_rates = get_fr(st, intervals[n])
                for nn in eachindex(st)
                    append!(fts[nn], deepcopy(st[nn]))
                    append!(loadings[nn], spike_rates[nn])
                end
            end
            reactivation_time[n, m] = sort.(deepcopy(fts))
            reactivation_load[n, m] = sort.(deepcopy(loadings))
        end
    end
    return reactivation_time, reactivation_load
end

function reactivation_per_interval(word_spikes, words)
    reactivation_time = Matrix{Vector{}}(undef, length(words), length(words))
    reactivation_load = Matrix{Vector{}}(undef, length(words), length(words))
    function my_mean(x)
        if length(x) > 0
            return float(mean(x))
        else
            return -1.0
        end
    end
    # reactivated population
    for m = 1:length(words)
        # target word
        for n = 1:length(words)
            neurons = length(word_spikes[n, m][1])
            @info "target_word $n, reactivated_word $m, neurons $neurons"
            spike_times_per_interval =
                [my_mean.(word_spikes[n, m][i]) for i = 1:length(word_spikes[n, m])]
            spike_counts_per_interval =
                mean.([length.(word_spikes[n, m][i]) for i = 1:length(word_spikes[n, m])])
            all_intervals = Vector{Vector{Float32}}(undef, length(spike_times_per_interval))
            all_loadings = zeros(length(spike_times_per_interval))
            # [all_intervals[n] = Vector{Float32}() for n in 1:length(all_intervals)]
            for i in eachindex(spike_times_per_interval)
                interval = spike_times_per_interval[i]
                all_loadings[i] += sum(spike_counts_per_interval[i]) / neurons
                # for nn in eachindex(all_neurons)
                #     neuron_ft = interval[nn]
                #     if neuron_ft != -1
                #         push!(all_neurons[nn], neuron_ft)
                #     end
                # end
            end
            reactivation_time[n, m] = spike_times_per_interval
            reactivation_load[n, m] = all_loadings
        end
    end
    return reactivation_time, reactivation_load
end

export Cohen_kappa

##

"""
    get_spikes_per_pop(spiketimes::Spiketimes, intervals::Vector{T}, populations::Vector{Vector{Int}}) where T <: AbstractVector{<:Real}

    Return the spiketimes for each population in each of the intervals selected. 
    Make no assumptions on the type of population, or the intervals.
    Parameters
    ----------
    spiketimes::Spiketimes
        The spiketimes for each neuron.
    intervals::Vector{T}
        The intervals to consider.
    populations::Vector{Vector{Int}}
        The populations to consider.
    Returns
    -------
    pop_spikes::Vector{Spiketimes}
        The spiketimes for each population in each of the intervals selected.
"""
function get_spikes_per_pop(
    spiketimes::Spiketimes,
    intervals::Vector{T},
    populations::Vector{Vector{Int}},
    target_interval = [0, 0],
) where {T<:AbstractVector{<:Real}}
    pop_spikes = [Spiketimes() for i = 1:length(populations)]
    for i in eachindex(populations)
        for ii in eachindex(intervals)
            pop = populations[i]
            zz = TNN.interval_standard_spikes(spiketimes[pop], intervals[ii])
            target = sum(target_interval) == 0 ? intervals[ii] : target_interval
            zz = Spiketimes(TNN.get_spikes_in_interval(zz, target))
            push!(pop_spikes[i], vcat(zz...))
        end
    end
    [sort!.(x) for x in pop_spikes]
    return pop_spikes
end

function rate_per_sample(;
    sign::String,
    sample,
    pop_unique,
    τ = 50,
    delay = nothing,
    duration = nothing,
)
    @info "Rate measure\nId: $(sample.id) \nSign: $sign"
    @assert (!isnothing(duration) || !isnothing(delay))
    @unpack stim, seq, net, dends, learn, store, W = sample.network
    spikes = sample.spikes

    word_activity =
        population_spikes(sign, spikes, seq, unique = pop_unique, delay = [0, duration])
    @unpack spiketimes, intervals_ids, intervals, n_intervals, targets = word_activity

    rates = Matrix{Any}(undef, length(intervals_ids), length(targets))

    for x in eachindex(intervals)
        _duration = round(Int, intervals[x][2]) + duration
        for y in eachindex(targets)
            # l_i = length(word_activity.n_intervals[x])
            _rr = zeros(_duration, length(spiketimes[x, y][1]))
            ga = grand_average(spiketimes[x, y])
            for neuron in eachindex(ga)
                _rr[:, neuron] =
                    convolve(ga[neuron], interval = 1.0:_duration, τ = τ) / n_intervals[x]
            end
            r = (mean = mean(_rr, dims = 2)[:, 1], std = std(_rr, dims = 2)[:, 1])
            rates[x, y] = r
        end
    end
    return rates, word_activity
end

function membrane_per_sample(;
    sample,
    pop_unique,
    delay = nothing,
    word::String,
    duration = nothing,
)
    @assert (!isnothing(duration) || !isnothing(delay))
    @unpack stim, seq, net, dends, learn, store, W = sample.network
    @unpack trackers = load_data(df.path[1])
    @unpack seq = oad_network(df.path[1])
    word_membrane = population_membrane(
        "words",
        trackers,
        seq,
        unique = pop_unique,
        delay = [0, duration],
    )
    return word_membrane
end

function grand_average(spiketimes::Vector{Spiketimes})
    n_intervals = length(spiketimes)
    n_neurons = length(spiketimes[1])
    all_spikes = [Vector{Float32}() for i = 1:n_neurons]
    for n = 1:n_intervals
        append!.(all_spikes, spiketimes[n])
    end
    return all_spikes
end