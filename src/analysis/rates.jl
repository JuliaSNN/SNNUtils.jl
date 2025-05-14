
function population_activity(; path::String, tt0 = 0)
    # @show dirname(path)
    # @assert(isdir(path))
    class_path = joinpath(path, "analysis")
    !(isdir(class_path)) && (mkdir(class_path))

    stim, seq, net, (dend1, dend2), learn, store = read_network_params(path)
    spikes = read_network_spikes(store.data, tt0 = tt0)
    read.(spikes)
    idxs = rescale_with_tt0_ratio(store.data, stim, tt0)

    ## Array used to subselect the spike list

    ## Labels
    word_labels = seq.sequence[1, idxs]
    ph_labels = seq.sequence[2, idxs]
    wn_labels = sort(collect(Set(word_labels)))
    pn_labels = sort(collect(Set(ph_labels)))

    ## number of global inputs and single file inputs
    inputs_in_batch = store.interval / stim.duration
    n_batches = length(spikes)
    inputs = round(Int, inputs_in_batch * n_batches)
    n_points = round(Int, stim.duration / dt)

    ## Initialize variables
    rates = zeros(Float32, net.neurons, inputs)
    words_activity = zeros(Float32, length(wn_labels), inputs)
    phonemes_activity = zeros(Float32, length(pn_labels), inputs)

    @inbounds @simd for index in eachindex(spikes)
        # for index in 1:length(spikes)
        previous_inputs = inputs_in_batch * (index - 1)
        exc = spikes[index].exc # get excitatory neurons
        sst = spikes[index].sst # get excitatory neurons
        pv = spikes[index].pv # get excitatory neurons
        last_tt = spikes[index].tt
        input_time = last_tt - store.interval
        input = 0
        # @show index, input_time, store.interval
        while input_time < last_tt
            input += 1
            m = round(Int, input + previous_inputs)
            rates[1:net.tripod, m] =
                get_spike_rate(exc, interval = (input_time, input_time + stim.duration))
            for n in eachindex(wn_labels)
                words_activity[n, m] = mean(
                    get_spike_rate(
                        exc[seq.populations[wn_labels[n]]],
                        interval = (input_time, input_time + stim.duration),
                    ),
                )
            end
            for n in eachindex(pn_labels)
                phonemes_activity[n, m] = mean(
                    get_spike_rate(
                        exc[seq.populations[pn_labels[n]]],
                        interval = (input_time, input_time + stim.duration),
                    ),
                )
            end
            input_time += stim.duration
        end
    end

    results = joinpath(class_path, "pop_activity.h5")
    isfile(results) && rm(results)
    h5open(results, "w") do fid
        fid["word"] = words_activity
        fid["phonemes"] = phonemes_activity
        fid["base_rate"] = rates
        fid["seq"] = seq.sequence
    end
    return words_activity, phonemes_activity, rates, seq
end

function score_rate(; path, category)
    try
        if category == "phonemes"
            fid = h5open(joinpath(path, "analysis/pop_activity.h5"), "r")
            ph = read(fid["phonemes"])
            myseq = read(fid["seq"])
            return mean(
                length(get_words(myseq)) .+ argmax.(eachcol(ph)) .== myseq[2, 1:12000],
            )
        elseif category == "words"
            fid = h5open(joinpath(path, "analysis/pop_activity.h5"), "r")
            word = read(fid["word"])
            myseq = read(fid["seq"])
            return mean(argmax.(eachcol(word)) .== myseq[1, 1:12000])
        else
            return -1
        end
    catch
        -1.0
    end
end


function score_spikes(; path::String, tt0::Int = -1)
    @assert(isdir(path))
    class_path = joinpath(path, "analysis")
    !(isdir(class_path)) && (mkdir(class_path))

    stim, seq, net, (dend1, dend2), learn, store = read_network_params(path)
    save_path = joinpath(class_path, "logreg_spikes.h5")
    isfile(save_path) && rm(save_path)

    ## ================
    ## Prepare features
    ## Retrieve features and average points within the stimulus interval
    spikes = read_network_spikes(store.data, tt0 = tt0, cache = false)
    features, n_neurons = spikes_to_features(spikes, stim.duration, store.interval)
    idxs = rescale_with_tt0_ratio(store.data, stim, tt0)
    score_features(;
        features = features,
        n_neurons = n_neurons,
        seq = seq,
        idxs = idxs,
        save_path = save_path,
        feats_set = (;
            :cv => (n_neurons+1):(2*n_neurons),
            :rate => 1:n_neurons,
            :all => 1:(2*n_neurons),
        ),
    )
end


"""
Measure the spike rate of the target populations and compare it with the non-target populations
"""
function target_pop_rate(
    interval::Int64,
    seq::Encoding,
    network_state::Array{Float32,3},
    stim::StimParams,
    symbol::Int,
)
    if !(symbol ∈ [1, 2])
        throw("symbol: 1 words, 2 phonemes")
    end
    target_interval =
        (1+round(Int, stim.duration / dt*(interval-1))):round(
            Int,
            stim.duration / dt*interval,
        )
    println(target_interval)
    words = Set(seq.sequence[symbol, :])
    target_word = seq.sequence[symbol, interval]
    pop = seq.populations[target_word]
    non_pop = collect(Set(vcat(seq.populations[collect(delete!(words, target_word))]...)))
    target = mean(get_spikes(exc[pop, 1, target_interval]))
    non_target = mean(get_spikes(exc[non_pop, 1, target_interval]))
    return target / non_target
end
