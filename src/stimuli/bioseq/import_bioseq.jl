using NPZ

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
        push!(experiments, (task=t,info=g))
    end
    return experiments
end

function bioseq_epochs(experiment, stage)
    epochs = []
    for epoch in keys(experiment.task[stage])
        push!(epochs, experiment.task[stage][epoch])
    end
    return  epochs
end

function make_unique_sequence(epochs, post_silence=1)
    @assert all(length(epoch) == length(epochs[1]) for epoch in epochs )
    interval_length = maximum(length.(epochs)) +post_silence
    sequence = Vector{String}()
    items_in_epoch = Vector{Int}()
    for epoch in epochs
        append!(sequence, epoch)
        for _ in 1:post_silence
            append!(sequence, ["_"])
        end
        push!(items_in_epoch, length(epoch) + post_silence)
    end
    @assert(all(length(collection) == length(epochs[1]) for collection in epochs))
    return Symbol.(sequence), items_in_epoch
end


function bioseq_lexicon(;experiment, duration::Float32=50.f0, kwargs...)
    dictionary = Dict{Symbol, Vector{Symbol}}()
    for w in experiment.info["task"]["test_string_set"]
        push!(dictionary, Symbol(join(w))=>[Symbol(p) for p in w])
    end
    words = keys(dictionary) |> collect |> sort
    phonemes = unique(Symbol.(experiment.info["task"]["g_strings"][1])) |> collect |> sort
    silence = :_

    @assert unique(phonemes) ==  union(vcat(values(dictionary)...)) "Phonemes do not match the dictionary"
    return (dict=dictionary, 
            symbols=(phonemes = phonemes, 
            words = words), 
            ph_duration = duration, 
            silence = silence)
end

function seq_bioseq(;experiment, stage::String, kwargs...) 
    lexicon = bioseq_lexicon(experiment=experiment; kwargs...)
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
        for  w in words
            _w = string(w)
            _p = string(p)
            !startswith(_w, _p) && continue
            (n + length(_w) > seq_length) && continue
            my_w = join(sequence_phonemes[n:n+length(_w)-1])
            if  my_w== _w
                sequence[1, n:n+length(_w)-1] .= w
                break
            end
        end
    end
    sequence[3, :] .= ph_duration
    epoch_timestamps = items_in_epochs .*ph_duration

    line_id = (phonemes=2, words=1, duration=3)
    sequence = (;lexicon...,
                sequence=sequence,
                line_id = line_id,
                timestamps= epoch_timestamps)
    

end


function root_path(path, exp)
    label = exp.info["label"]
    seed  = exp.info["seed_network"]
    id    = exp.info["seed"] 
    return joinpath(path, "id-$(id)_seed-$(seed)_$(label)") |> mkpath
end

function store_experiment_data(path, exp, network, seq)
    ## Root
    _root = root_path(path, exp)

    ## Experiment data
    label = exp.info["label"]
    seed  = exp.info["seed_network"]
    id    = exp.info["seed"] 
    mapping = Dict(string(k)=>string.(v) for (k,v) in seq.dict)
    exp_data = Dict(
            "seed"=> seed,
            "label"=> label,
            "symbol_duration"=>seq.ph_duration, 
    )
    neurons_ranges = let 
            exc = network.pop.E.N
            pv = network.pop.I1.N
            sst = network.pop.I2.N
            cumsum([1,exc, sst, pv]) |> x-> [collect(x[n]:(x[n+1]-1)) for n in 1:length(x)-1]
    end
    DrWatson.save(joinpath(_root, "mapping.h5"), mapping)
    DrWatson.save(joinpath(_root, "info.h5"), exp_data)
    DrWatson.save(joinpath(_root,"spikeinfo.h5"), @strdict exc = neurons_ranges[1] sst = neurons_ranges[2] pv = neurons_ranges[3])
    return _root
end

function store_activity_data(_root::String, stage::String, sequence, model; targets=[:d])
    folder = joinpath(_root, stage) |> mkpath
    @unpack stim = model
    myspikes = vcat(spiketimes(model.pop.E), spiketimes(model.pop.I1), spiketimes(model.pop.I2))
    myspikes = myspikes  |> d-> Dict("$n"=>d[n] for n in eachindex(d))
    labels, target_pops, = let
            stim_id =[]
            stim_time = []
            target_pops = Dict{String, Vector{Int}}()
            for k in sequence.symbols.phonemes
                cells = []
                for t in targets
                    t = Symbol(string(k,"_",t))
                    ph_stim = getfield(stim, t)
                    push!(cells, ph_stim.cells)
                end
                push!(target_pops, string(k)=>Set(vcat(cells...))|> collect)
            end
            for k in sequence.symbols.phonemes
                    ph_stim = getfield(stim,Symbol(string(k,"_",targets[1])))
                    for interval in ph_stim.param.variables[:intervals]
                            push!(stim_time, interval[1])
                            push!(stim_id, k)
                    end
            end
            iid = sort(1:length(stim_id), by=x->stim_time[x])
            labels = Dict{}()
            for (k,t) in zip(stim_id[iid], stim_time[iid])
                    labels[string(t)] = string(k)
            end
            labels, target_pops
    end
    DrWatson.save(joinpath(folder,"labels.h5"), labels) 
    DrWatson.save(joinpath(folder,"target_pops.h5"), target_pops) 
    DrWatson.save(joinpath(folder,"spiketimes.h5"), myspikes ) 

    membrane, r_t = SNN.interpolated_record(model.pop.E, :v_s)
    epoch_extrema =  cumsum([0,sequence.timestamps...])|> x-> [(x[n],(x[n+1])) for n in 1:length(x)-1]
    @unpack ph_duration = sequence
    for epoch in eachindex(epoch_extrema)
            _start, _end = epoch_extrema[epoch]
            offset = _start+ph_duration : ph_duration : _end-ph_duration
            offset_delay = offset .+ ph_duration
            mkpath(joinpath(folder, "membrane_end"))
            membrane_path = joinpath(folder, "membrane_end", "epoch_$(epoch).npz")
            _timepoints = offset
            if offset_delay[end] < r_t[end]
                    mem =  membrane[:,_timepoints]
                    timestamps = _timepoints
                    npzwrite(membrane_path, Dict("membrane" => mem,  "timestamp" => timestamps))
            end
            
            mkpath(joinpath(folder, "membrane_delay"))
            membrane_path = joinpath(folder, "membrane_delay", "epoch_$(epoch).npz") 
            _timepoints = offset_delay
            if offset_delay[end] < r_t[end]
                    mem =  membrane[:,_timepoints]
                    timestamps = _timepoints
                    npzwrite(membrane_path, Dict("membrane" => mem,  "timestamp" => timestamps))
            end
    end
end
##

export import_bioseq_tasks, seq_bioseq, bioseq_epochs, bioseq_lexicon, store_experiment_data, store_activity_data, root_path
