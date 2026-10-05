using NPZ

"""
    import_bioseq_tasks(generator_path, task_path)

Read the BioSeq task definitions (JSON files) and pair each task with its generator description.

Returns a vector of `NamedTuple`s `(task, info)` with the parsed JSON dictionaries.

Note: the file names are listed in `generator_path` but read from `task_path` (and vice versa),
so the two folders must contain JSON files with the same names, in the same order.
"""
function import_bioseq_tasks(generator_path, task_path)
    task_list = []
    generators_list = []
    json_files = filter(x -> occursin(".json", x), readdir(generator_path))
    for json_file in json_files
        file_path = joinpath(task_path, json_file)
        file = open(file_path)
        dict_data = JSON.parse(file)
        close(file)
        push!(task_list, dict_data)
    end

    json_files = filter(x -> occursin(".json", x), readdir(task_path))
    for json_file in json_files
        file_path = joinpath(generator_path, json_file)
        file = open(file_path)
        dict_data = JSON.parse(file)
        close(file)
        push!(generators_list, dict_data)
    end

    experiments = []
    for (g, t) in zip(generators_list, task_list)
        push!(experiments, (task = t, info = g))
    end
    return experiments
end

"""
    bioseq_epochs(experiment, stage)

Return the list of epochs (the values of `experiment.task[stage]`) of a BioSeq experiment stage.
"""
function bioseq_epochs(experiment, stage)
    epochs = []
    for epoch in keys(experiment.task[stage])
        push!(epochs, experiment.task[stage][epoch])
    end
    return epochs
end

function make_unique_sequence(epochs, post_silence = 1)
    @assert all(length(epoch) == length(epochs[1]) for epoch in epochs)
    interval_length = maximum(length.(epochs)) + post_silence
    sequence = Vector{String}()
    items_in_epoch = Vector{Int}()
    for epoch in epochs
        append!(sequence, epoch)
        for _ = 1:post_silence
            append!(sequence, ["_"])
        end
        push!(items_in_epoch, length(epoch) + post_silence)
    end
    replace!(sequence, "#"=>"_")
    @assert(all(length(collection) == length(epochs[1]) for collection in epochs))
    return Symbol.(sequence), items_in_epoch
end


"""
    bioseq_lexicon(; experiment, duration::Float32 = 50.0f0, kwargs...)

Build a lexicon (same layout as [`generate_lexicon`](@ref)) from a BioSeq experiment: the words
are the strings of `experiment.info["task"]["test_string_set"]`, the phonemes their characters.
`ph_duration` is the scalar `duration` (ms) shared by all symbols; `silence = :_`.
"""
function bioseq_lexicon(; experiment, duration::Float32 = 50.0f0, kwargs...)
    dictionary = Dict{Symbol,Vector{Symbol}}()
    for w in experiment.info["task"]["test_string_set"]
        push!(dictionary, Symbol(join(w))=>[Symbol(p) for p in w])
    end
    words = keys(dictionary) |> collect |> sort
    phonemes = unique(Symbol.(experiment.info["task"]["g_strings"][1])) |> collect |> sort
    silence = :_

    @assert unique(phonemes) == union(vcat(values(dictionary)...)) "Phonemes do not match the dictionary"
    return (
        dict = dictionary,
        symbols = (phonemes = phonemes, words = words),
        ph_duration = duration,
        silence = silence,
    )
end

"""
    seq_bioseq(; experiment, stage::String, kwargs...)

Build the timed sequence of a BioSeq experiment stage. The epochs of `stage` are concatenated
with one silence element after each (`#` is replaced by silence), words of the lexicon are
located in the phoneme stream, and every element lasts `ph_duration`.

# Returns
The lexicon of [`bioseq_lexicon`](@ref) extended with `sequence` (a `3 x L` matrix: words,
phonemes, durations), `line_id = (phonemes = 2, words = 1, duration = 3)` and `timestamps` (the
duration of each epoch in ms). `kwargs` are passed to `bioseq_lexicon` (e.g. `duration`).
"""
function seq_bioseq(; experiment, stage::String, kwargs...)
    lexicon = bioseq_lexicon(experiment = experiment; kwargs...)
    @unpack phonemes, words = lexicon.symbols
    @unpack ph_duration, silence = lexicon

    ## Get the stage sequence
    epochs = bioseq_epochs(experiment, stage)
    sequence_phonemes, items_in_epochs = make_unique_sequence(epochs)
    #
    seq_length = length(sequence_phonemes)
    sequence = Matrix{Any}(fill(silence, 3, seq_length))
    sequence[2, :] = sequence_phonemes
    for (n, p) in enumerate(sequence_phonemes)
        for w in words
            _w = string(w)
            _p = string(p)
            !startswith(_w, _p) && continue
            (n + length(_w) > seq_length) && continue
            my_w = join(sequence_phonemes[n:(n+length(_w)-1)])
            if my_w == _w
                sequence[1, n:(n+length(_w)-1)] .= w
                break
            end
        end
    end
    sequence[3, :] .= ph_duration
    epoch_timestamps = items_in_epochs .* ph_duration

    line_id = (phonemes = 2, words = 1, duration = 3)
    sequence = (;
        lexicon...,
        sequence = sequence,
        line_id = line_id,
        timestamps = epoch_timestamps,
    )


end


"""
    root_path(path, exp)

Create (if needed) and return the folder `path/id-<seed>_seed-<seed_network>_<label>` for the
experiment `exp` (fields taken from `exp.info`).
"""
function root_path(path, exp)
    label = exp.info["label"]
    seed = exp.info["seed_network"]
    id = exp.info["seed"]
    return joinpath(path, "id-$(id)_seed-$(seed)_$(label)") |> mkpath
end

"""
    store_experiment_data(path, exp, network, seq)

Save the symbol mapping (`mapping.h5`), the experiment info (`info.h5`: seed, label, symbol
duration) and the neuron index ranges (`spikeinfo.h5`) of the populations `E`, `I1`, `I2` of
`network` into [`root_path`](@ref)`(path, exp)`, and return that folder.

Note: the ranges are built in the order `E`, then `I2.N` neurons labelled `sst`, then `I1.N`
neurons labelled `pv`, while [`store_activity_data`](@ref) concatenates spikes in the order
`E, I1, I2`.
"""
function store_experiment_data(path, exp, network, seq)
    ## Root
    _root = root_path(path, exp)

    ## Experiment data
    label = exp.info["label"]
    seed = exp.info["seed_network"]
    id = exp.info["seed"]
    mapping = Dict(string(k)=>string.(v) for (k, v) in seq.dict)
    exp_data = Dict("seed" => seed, "label" => label, "symbol_duration"=>seq.ph_duration)
    neurons_ranges = let
        exc = network.pop.E.N
        pv = network.pop.I1.N
        sst = network.pop.I2.N
        cumsum([1, exc, sst, pv]) |> x -> [collect(x[n]:(x[n+1]-1)) for n = 1:(length(x)-1)]
    end

    DrWatson.save(joinpath(_root, "mapping.h5"), mapping)
    DrWatson.save(joinpath(_root, "info.h5"), exp_data)
    DrWatson.save(
        joinpath(_root, "spikeinfo.h5"),
        @strdict exc = neurons_ranges[1] sst = neurons_ranges[2] pv = neurons_ranges[3]
    )
    return _root
end

"""
    store_target_pops(_root, seq, stim, targets)

Save to `target_pops.h5` in `_root` the indices of the neurons targeted by the stimulus of every
phoneme and word of `seq` (union over the compartments `targets`, read as
`getfield(getfield(stim, symbol), target).neurons`). Returns `_root`.
"""
function store_target_pops(_root, seq, stim, targets)
    folder = _root
    target_pops = Dict{String,Vector{Int}}()
    for k in seq.symbols.phonemes
        neurons = []
        for t in targets
            ph_stim = getfield(stim, k)
            ph_stim = getfield(ph_stim, t)
            push!(neurons, ph_stim.neurons)
        end
        push!(target_pops, string(k)=>Set(vcat(neurons...)) |> collect)
    end
    for k in seq.symbols.words
        neurons = []
        for t in targets
            ph_stim = getfield(stim, k)
            ph_stim = getfield(ph_stim, t)
            push!(neurons, ph_stim.neurons)
        end
        push!(target_pops, string(k)=>Set(vcat(neurons...)) |> collect)
    end
    DrWatson.save(joinpath(folder, "target_pops.h5"), target_pops)
    return _root
end

# function getstim(stim, field, target)
#     ph_stim = getfield(stim, field)
#     ph_stim = getfield(ph_stim, target)
#     return ph_stim
# end

"""
    store_labels(_root, stim, sequence, targets)

Save to `labels.h5` in `_root` a dictionary mapping the onset time of every phoneme presentation
(read from the intervals of the stimulus `stim[Symbol(phoneme, "_", targets[1])]`) to the
phoneme, and return the dictionary.
"""
function store_labels(_root, stim, sequence, targets)
    folder = _root
    # Get the labels
    labels = Dict{}()
    stim_id = []
    stim_time = []
    for k in sequence.symbols.phonemes
        # ph_stim = getstim(stim, k, targets[1])
        ph_stim = getfield(stim, Symbol(string(k, "_", targets[1])))
        for interval in ph_stim.param.variables[:intervals]
            push!(stim_time, interval[1])
            push!(stim_id, k)
        end
    end
    iid = sort(1:length(stim_id), by = x->stim_time[x])
    for (k, t) in zip(stim_id[iid], stim_time[iid])
        labels[string(t)] = string(k)
    end
    DrWatson.save(joinpath(folder, "labels.h5"), labels)
    return labels
end





"""
    store_activity_data(_root::String, stage::String, sequence, model; targets = [:d])

Save the activity of a BioSeq run in `_root/stage`: phoneme labels ([`store_labels`](@ref)),
spike times of `model.pop.E`, `I1`, `I2` (`spiketimes.h5`) and, for every epoch, the somatic
membrane potential `:v_s` of `E` at the end of each element and one element later
(`membrane_end/epoch_i.npz`, `membrane_delay/epoch_i.npz`).

Note: the function calls `SNN.record`, but the name `SNN` is not defined inside SNNUtils, so it
throws an `UndefVarError` when it reaches the membrane traces.
"""
function store_activity_data(_root::String, stage::String, sequence, model; targets = [:d])
    folder = joinpath(_root, stage) |> mkpath
    @unpack stim = model
    # Get the labels
    labels = store_labels(folder, stim, sequence, targets)

    # Get the spikes
    myspikes =
        vcat(spiketimes(model.pop.E), spiketimes(model.pop.I1), spiketimes(model.pop.I2))
    myspikes = myspikes |> d -> Dict("$n"=>d[n] for n in eachindex(d))
    DrWatson.save(joinpath(folder, "spiketimes.h5"), myspikes)

    # Membrane traces
    membrane, r_t = SNN.record(model.pop.E, :v_s, range = true)
    epoch_extrema =
        cumsum([0, sequence.timestamps...]) |>
        x -> [(x[n], (x[n+1])) for n = 1:(length(x)-1)]
    @unpack ph_duration = sequence
    for epoch in eachindex(epoch_extrema)
        _start, _end = epoch_extrema[epoch]
        offset = (_start+ph_duration):ph_duration:(_end-ph_duration)
        offset_delay = offset .+ ph_duration
        mkpath(joinpath(folder, "membrane_end"))
        membrane_path = joinpath(folder, "membrane_end", "epoch_$(epoch).npz")
        _timepoints = offset
        if offset_delay[end] < r_t[end]
            mem = membrane[:, _timepoints]
            timestamps = _timepoints
            npzwrite(membrane_path, Dict("membrane" => mem, "timestamp" => timestamps))
        end

        mkpath(joinpath(folder, "membrane_delay"))
        membrane_path = joinpath(folder, "membrane_delay", "epoch_$(epoch).npz")
        _timepoints = offset_delay
        if offset_delay[end] < r_t[end]
            mem = membrane[:, _timepoints]
            timestamps = _timepoints
            npzwrite(membrane_path, Dict("membrane" => mem, "timestamp" => timestamps))
        end
    end
end
##

export import_bioseq_tasks,
    seq_bioseq,
    bioseq_epochs,
    bioseq_lexicon,
    store_experiment_data,
    store_activity_data,
    root_path,
    store_target_pops,
    store_labels
