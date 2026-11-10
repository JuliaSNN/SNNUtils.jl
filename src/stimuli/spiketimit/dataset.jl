"""
    _read_transcription(file::AbstractString, sr::Float64) -> Transcription{Float64}

Read a TIMIT `.wrd`/`.phn` file: one record `"t0 t1 label"` per line, `t0`, `t1` integer sample
indices. Times are returned in seconds (`sample / sr`, `sr` in Hz). Blank lines are skipped.
Throws an `ArgumentError` naming `file:line` for records without exactly 3 fields, with
non-integer sample indices or with `t0 > t1`; a missing file raises the `SystemError` of `open`.
"""
function _read_transcription(file::AbstractString, sr::Float64)
    intervals = NTuple{2,Float64}[]
    signs = String[]
    for (lineno, line) in enumerate(readlines(file))
        isempty(strip(line)) && continue
        fields = split(line)
        length(fields) == 3 ||
            throw(ArgumentError("$(file):$(lineno): expected 't0 t1 label', got \"$(line)\""))
        t0 = tryparse(Int, fields[1])
        t1 = tryparse(Int, fields[2])
        (t0 === nothing || t1 === nothing) && throw(ArgumentError(
            "$(file):$(lineno): sample indices must be integers, got \"$(line)\""))
        t0 <= t1 || throw(ArgumentError("$(file):$(lineno): t0 > t1 in \"$(line)\""))
        push!(intervals, (t0 / sr, t1 / sr))
        push!(signs, String(fields[3]))
    end
    return Transcription{Float64}(intervals, signs)
end

"""
    _parse_dialect(root::AbstractString) -> Int

Return the dialect region `k` of the speaker directory `root`, whose parent directory must be
named `dr<k>`. Throws an `ArgumentError` naming `root` otherwise.
"""
function _parse_dialect(root::AbstractString)
    parts = splitpath(root)
    m = length(parts) >= 2 ? match(r"^dr(\d+)$", parts[end-1]) : nothing
    m === nothing && throw(ArgumentError(
        "speaker directory \"$(root)\" is not inside a dialect directory named dr<k>"))
    return parse(Int, m.captures[1])
end

function _empty_dataset()
    return DataFrame(speaker = String[], senID = String[], dialect = Int[], gender = Char[],
                     path = String[], words = Transcription{Float64}[],
                     phones = Transcription{Float64}[], sentence = Vector{String}[])
end

"""
    create_dataset(dir::AbstractString; config = SpikeTimitConfig()) -> DataFrame
    create_dataset(; dir = dataset_path("train"), config = SpikeTimitConfig()) -> DataFrame

Read every sentence below `dir` (a split such as `<root>/train`) into a `DataFrame`, one row per
sentence, sorted by `(speaker, senID)`.

A sentence is any `<senID>.wav` file in a speaker directory `.../dr<k>/<speaker>/`; its
`<senID>.wrd` and `<senID>.phn` transcriptions are read (the audio is not). Symbolic links are
not followed. Other files are ignored.

# Arguments
- `dir::AbstractString`: directory to scan; must exist.
- `config::SpikeTimitConfig`: `config.sr` (Hz) converts sample indices to seconds.

# Returns
`DataFrame` with columns
- `speaker::String`, `senID::String`, `dialect::Int` (`k` of `dr<k>`), `gender::Char` (first
  letter of the speaker);
- `path::String`: sentence base path without extension (spike files are `path * suffix`);
- `words`, `phones`: [`Transcription{Float64}`](@ref Transcription), intervals in seconds;
- `sentence::Vector{String}`: the words of the sentence.

Throws an `ArgumentError` if `dir` is not a directory, if a speaker directory is not inside a
`dr<k>` directory, or for a malformed transcription line (naming `file:line`); a missing
transcription file raises a `SystemError`.

# Examples
```julia
using SNNUtils.SpikeTimit
df = create_dataset(dataset_path("train"))
df.words[1].signs, df.words[1].intervals   # words of the first sentence, times in s
```

See also [`find_word`](@ref), [`get_spiketimes`](@ref), [`dataset_path`](@ref).
"""
function create_dataset(dir::AbstractString; config::SpikeTimitConfig = SpikeTimitConfig())
    isdir(dir) || throw(ArgumentError("create_dataset: \"$(dir)\" is not a directory"))
    df = _empty_dataset()
    for (root, _, files) in walkdir(dir; topdown = true, follow_symlinks = false)
        for file in files
            endswith(file, ".wav") || continue
            senID = String(split(file, ".")[1])
            speaker = String(splitpath(root)[end])
            gender = speaker[1]
            dialect = _parse_dialect(root)
            path = joinpath(root, senID)
            words = _read_transcription(path * ".wrd", config.sr)
            phones = _read_transcription(path * ".phn", config.sr)
            push!(df, (speaker, senID, dialect, gender, path, words, phones, copy(words.signs)))
        end
    end
    return sort!(df, [:speaker, :senID])
end

create_dataset(; dir::AbstractString = dataset_path("train"),
               config::SpikeTimitConfig = SpikeTimitConfig()) = create_dataset(dir; config)

"""
    create_dictionary(file::AbstractString) -> Dict{String,String}
    create_dictionary(; file = dataset_path("DOC", "TIMITDIC.TXT")) -> Dict{String,String}

Read the TIMIT dictionary: entries `"<word>  /<phones>/"` become `word => phones` (phones with
stress digits, e.g. `"that" => "dh ae1 t"`). Lines starting with `;` and blank lines are skipped;
later duplicates overwrite earlier ones. Throws an `ArgumentError` naming `file:line` for an
entry without `/`.

# Examples
```julia
using SNNUtils.SpikeTimit
dict = create_dictionary()     # needs ENV["SPIKETIMIT_PATH"]
dict["that"]                   # "dh ae1 t"
```
"""
function create_dictionary(file::AbstractString)
    dict = Dict{String,String}()
    for (lineno, line) in enumerate(readlines(file))
        (startswith(line, ";") || isempty(strip(line))) && continue
        parts = split(line, "/")
        length(parts) >= 2 ||
            throw(ArgumentError("$(file):$(lineno): expected '<word>  /<phones>/', got \"$(line)\""))
        dict[String(strip(parts[1]))] = String(parts[2])
    end
    return dict
end

create_dictionary(; file::AbstractString = dataset_path("DOC", "TIMITDIC.TXT")) =
    create_dictionary(file)

"""
    find_word(df::AbstractDataFrame, words) -> DataFrame
    find_word(; df, words) -> DataFrame

Return a copy of the rows of `df` (from [`create_dataset`](@ref)) whose sentence contains at
least one of `words` (a `String` or a vector of strings; exact, case-sensitive match). Column
and row order are kept.

# Examples
```julia
using SNNUtils.SpikeTimit
df = create_dataset(dataset_path("train"))
find_word(df, "that")
find_word(df, ["she", "had"])
```

See also [`count_words`](@ref), [`select_words`](@ref).
"""
function find_word(df::AbstractDataFrame,
                   words::Union{AbstractString,AbstractVector{<:AbstractString}})
    ws = words isa AbstractString ? [words] : words
    return filter(:words => t -> any(in(t.signs), ws), df)
end

find_word(; df::AbstractDataFrame, words) = find_word(df, words)

"""
    count_words(df::AbstractDataFrame, words::AbstractVector{<:AbstractString}) -> Dict{String,Int}
    count_words(words::AbstractVector{Word}) -> Dict{String,Int}

Count, for every word in `words`, the number of sentences (rows) of `df` that contain it; or
count the occurrences of each word in a vector of [`Word`](@ref)s (e.g. the `labels` of
[`select_words`](@ref)).

# Examples
```julia
using SNNUtils.SpikeTimit
count_words(create_dataset(dataset_path("train")), ["that", "she"])
```
"""
function count_words(df::AbstractDataFrame, words::AbstractVector{<:AbstractString})
    return Dict{String,Int}(String(w) => nrow(find_word(df, w)) for w in words)
end

count_words(words::AbstractVector{Word}) =
    Dict{String,Int}(StatsBase.countmap(getfield.(words, :word)))
