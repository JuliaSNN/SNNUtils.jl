using DataFrames

## Get all the experiments that are instatiated in _models_
function import_data(; note::String = "", dir = "models", root = "")
    dir = replace(dir, "data/" => "")
    _df = DataFrame(
        model = String[],
        id = UInt64[],
        path = String[],
        _params = Union{ExpParams,StoreParams}[],
    )
    (root == "") && (root = joinpath(__ROOT__, "data"))
    root = joinpath(root, dir)
    dirs = readdir(root)
    for dir in dirs
        (occursin("test", dir)) && (continue)
        ## model folder
        path = joinpath(root, dir)
        @show path
        if isdir(joinpath(path, "params"))
            id = hash(path)
            try
                _params = deserialize(joinpath(path, "params", "params.so"))
                push!(_df, (dir, id, path, _params))
            catch
                store = deserialize(joinpath(path, "params", "store.so"))
                push!(_df, (dir, id, path, store))
            end
        else
            df_temp = import_data(dir = dir, root = root)
            @show dir
            append!(_df, df_temp, cols = :setequal)
        end
    end
    return _df
end

function model_path_name(store, folder_name)
    path = joinpath(store.path, folder_name)
    (!isdir(path)) && (mkdir(path))
    return path
end

function expand_parameters!(df::DataFrame)
    try
        for k in fieldnames(ExpParams)
            transform!(df, :_params => ByRow(x -> getfield(x, k)) => k)
        end
    catch
        for k in fieldnames(StoreParams)
            @show k
            transform!(df, :_params => ByRow(x -> getfield(x, k)) => k)
        end
    end
end


function get_score_membrane(; path::String, category = "words")
    try
        fid = h5open(joinpath(path, "analysis/logreg_membrane.h5"))
        scores = read(fid[category]["scores"])
        # labels = read(fid[category]["labels"])
        close(fid)
        return scores
    catch
        return [0]
    end
end

function get_score_spikes(; path::String, category = "words")
    try
        fid = h5open(joinpath(path, "classify/logreg_spikes.h5"))
        scores = read(fid[category]["scores"])
        # labels = read(fid[category]["labels"])
        close(fid)
        return scores
    catch
        return [0]
    end
end

function get_activity(; path::String)
    fid = h5open(joinpath(path, "activity/pop_activity.h5"), "r")
    words = read(fid["word"])
    phs = read(fid["phonemes"])
    base = read(fid["base_rate"])
    seq = read(fid["seq"])
    close(fid)
    return words, phs, base, seq
end

function score_activity(; id::String)
    words, phs, base, sequence = get_activity(id = id)
    targets = MLJLinearModels.softmax(phs') |> preds -> map(x -> argmax(x), eachrow(preds))
    ph_scores = mean(targets[6000:end] .+ 28 .== sequence[2, 6000:length(targets)])
    targets =
        MLJLinearModels.softmax(words') |> preds -> map(x -> argmax(x), eachrow(preds))
    w_scores = mean(targets[6000:end] .== sequence[1, 6000:length(targets)])
    return ph_scores, w_scores
end

function score_data(data)
    scores = []
    labels = []
    for d = 1:length(data)
        sc, _ = score_logreg(id = data[d][2])
        ph, w = score_activity(id = data[d][2])
        push!(scores, [sc..., ph, w])
    end
    return scores
end

function get_rates(path)
    data = read_network_params(path).store.data
    rates = read_network_rates(data)
    intervals = length(rates)
    matrices = zeros(3, intervals)
    for i = 1:intervals
        for x = 2:4
            matrices[x-1, i] = mean(rates[i][x])
        end
    end
    return matrices
end
##
