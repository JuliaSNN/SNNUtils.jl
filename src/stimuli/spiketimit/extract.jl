"""
    word_intervals(df::AbstractDataFrame, word::AbstractString) -> Vector{Vector{NTuple{2,Float64}}}

Return, for every row of `df` (row order), the intervals `(t0, t1)` (s) of every occurrence of
`word` in the sentence. Rows without `word` give an empty vector.

# Examples
```julia
using SNNUtils.SpikeTimit
word_intervals(create_dataset(dataset_path("train")), "that")
```

See also [`get_spikes_in_interval`](@ref), [`get_word_spikes`](@ref).
"""
function word_intervals(df::AbstractDataFrame, word::AbstractString)
    out = Vector{Vector{NTuple{2,Float64}}}()
    for t in df.words
        push!(out, NTuple{2,Float64}[iv for (iv, s) in zip(t.intervals, t.signs) if s == word])
    end
    return out
end

"""
    get_spikes_in_interval(st, interval::NTuple{2,<:Real}) -> (SpikeTrains{Float64}, Float64)
    get_spikes_in_interval(st, intervals::AbstractVector{<:NTuple{2,<:Real}}) -> (Vector{SpikeTrains{Float64}}, Vector{Float64})
    get_spikes_in_interval(batch, df_intervals::AbstractVector{<:AbstractVector{<:NTuple{2,<:Real}}}) -> (Vector{SpikeTrains{Float64}}, Vector{Float64})

Cut the spikes of an interval `(t0, t1)` (s) out of the spike trains `st` (s).

Spikes with `t0 < t < t1` (strict bounds) are kept, shifted to `t - t0`; the duration is
`t1 - t0` (s). With a vector of intervals, one result per interval. With a batch (one sample
per row) and one vector of intervals per sample (as returned by [`word_intervals`](@ref)), the
results of all samples and intervals are concatenated in order.

# Returns
- spike trains (s, relative to the interval onset) and duration(s) (s).

Throws a `DimensionMismatch` if `batch` and `df_intervals` have different lengths.

# Examples
```julia
using SNNUtils.SpikeTimit
st = [[0.25, 0.5, 0.75], [0.5]]
get_spikes_in_interval(st, (0.25, 0.75))   # ([[0.25], [0.25]], 0.5)
```
"""
function get_spikes_in_interval(st::_Sample, interval::NTuple{2,<:Real})
    a, b = Float64(interval[1]), Float64(interval[2])
    out = SpikeTrains{Float64}(undef, length(st))
    for (i, channel) in enumerate(st)
        out[i] = Float64[Float64(t) - a for t in channel if a < t < b]
    end
    return out, b - a
end

function get_spikes_in_interval(st::_Sample, intervals::AbstractVector{<:NTuple{2,<:Real}})
    spikes = SpikeTrains{Float64}[]
    durations = Float64[]
    for interval in intervals
        s, d = get_spikes_in_interval(st, interval)
        push!(spikes, s)
        push!(durations, d)
    end
    return spikes, durations
end

function get_spikes_in_interval(batch::_Batch,
                                df_intervals::AbstractVector{<:AbstractVector{<:NTuple{2,<:Real}}})
    length(batch) == length(df_intervals) || throw(DimensionMismatch(
        "get_spikes_in_interval: $(length(batch)) samples for $(length(df_intervals)) interval vectors"))
    spikes = SpikeTrains{Float64}[]
    durations = Float64[]
    for (st, intervals) in zip(batch, df_intervals)
        s, d = get_spikes_in_interval(st, intervals)
        append!(spikes, s)
        append!(durations, d)
    end
    return spikes, durations
end

"""
    get_word_spikes(df, words; encoding, config = SpikeTimitConfig()) -> (Vector{SpikeTrains{Float64}}, Vector{Float64})

Return the spike trains (s, relative to the word onset) and the durations (s) of every
occurrence of `words` (a `String` or a vector of strings) in the sentences of `df`. The spikes
of `df` are read once; the results are ordered by word, then row, then occurrence.

# Arguments
- `df::AbstractDataFrame`: sentences from [`create_dataset`](@ref).
- `words`: word or vector of words.
- `encoding` (required): a [`SpikeEncoding`](@ref) or its name.
- `config::SpikeTimitConfig`: passed to [`read_spiketimes`](@ref).

# Examples
```julia
using SNNUtils.SpikeTimit
df = create_dataset(dataset_path("train"))
spikes, durations = get_word_spikes(find_word(df, "that"), "that"; encoding = Cochlea(70))
```

See also [`select_words`](@ref), [`get_word_labels`](@ref).
"""
function get_word_spikes(df::AbstractDataFrame,
                         words::Union{AbstractString,AbstractVector{<:AbstractString}};
                         encoding, config::SpikeTimitConfig = SpikeTimitConfig())
    ws = words isa AbstractString ? [words] : words
    all_spikes = get_spiketimes(df, encoding; config)
    spikes = SpikeTrains{Float64}[]
    durations = Float64[]
    for w in ws
        s, d = get_spikes_in_interval(all_spikes, word_intervals(df, w))
        append!(spikes, s)
        append!(durations, d)
    end
    return spikes, durations
end

"""
    get_word_labels(df::AbstractDataFrame, word::AbstractString) -> Vector{Vector{Word}}

Return, for every row of `df`, a [`Word`](@ref) for each occurrence of `word`, with the phones
that lie inside the word (`p.t0 >= t0 && p.t1 <= t1`, bounds included), their times relative to
the word onset (s).

# Examples
```julia
using SNNUtils.SpikeTimit
labels = get_word_labels(find_word(df, "that"), "that")
labels[1][1].phones
```
"""
function get_word_labels(df::AbstractDataFrame, word::AbstractString)
    out = Vector{Vector{Word}}()
    for row in eachrow(df)
        wt, pt = row.words, row.phones
        words = Word[]
        for ((t0, t1), s) in zip(wt.intervals, wt.signs)
            s == word || continue
            phones = Phone[Phone(ph, p0 - t0, p1 - t0)
                           for ((p0, p1), ph) in zip(pt.intervals, pt.signs) if p0 >= t0 && p1 <= t1]
            push!(words, Word(String(s), phones, t1 - t0, t0, t1))
        end
        push!(out, words)
    end
    return out
end

"""
    select_words(df, words; encoding, samples = nothing, replace = false,
                 rng = Random.default_rng(), config = SpikeTimitConfig())
        -> (durations, spikes, labels)

Collect the spike trains, durations and labels of the occurrences of each of `words` in `df`,
ready for [`get_ordered_spikes`](@ref).

For each word, the sentences that contain it are found with [`find_word`](@ref) (words absent
from `df` are skipped). With `samples = n`, `n` of these sentences are drawn with `rng` (without
replacement by default; `replace = true` draws with replacement, as SpikeTimit.jl 0.1 did);
if fewer than `n` sentences contain the word, all of them are used. Every occurrence of the
word in the selected sentences gives one sample.

# Arguments
- `df::AbstractDataFrame`: sentences from [`create_dataset`](@ref).
- `words::AbstractVector{<:AbstractString}`: words to select, in order.
- `encoding` (required): a [`SpikeEncoding`](@ref) or its name.
- `samples::Union{Int,Nothing} = nothing`: sentences per word; `nothing` uses all; must be `>= 1`.
- `replace::Bool = false`: draw sentences with replacement.
- `rng::AbstractRNG`: random number generator.
- `config::SpikeTimitConfig`: passed to [`read_spiketimes`](@ref).

# Returns
`NamedTuple` with
- `durations::Vector{Float64}`: duration (s) of each sample;
- `spikes::Vector{SpikeTrains{Float64}}`: spike trains (s, from the word onset);
- `labels::Vector{Word}`: word and phones of each sample.

# Examples
```julia
using SNNUtils.SpikeTimit, Random
df = create_dataset(dataset_path("train"))
inputs = select_words(df, ["that", "had"]; encoding = Cochlea(70), samples = 5,
                      rng = MersenneTwister(1))
```
"""
function select_words(df::AbstractDataFrame, words::AbstractVector{<:AbstractString};
                      encoding, samples::Union{Int,Nothing} = nothing, replace::Bool = false,
                      rng::AbstractRNG = Random.default_rng(),
                      config::SpikeTimitConfig = SpikeTimitConfig())
    samples !== nothing && samples < 1 &&
        throw(ArgumentError("samples must be >= 1 (or nothing), got $(samples)"))
    all_durations = Float64[]
    all_spikes = SpikeTrains{Float64}[]
    all_labels = Word[]
    for word in words
        df_word = find_word(df, word)
        n = nrow(df_word)
        if n == 0
            @debug "select_words: word \"$(word)\" not found, skipped"
            continue
        end
        inds = if samples !== nothing && samples <= n
            replace ? rand(rng, 1:n, samples) : StatsBase.sample(rng, 1:n, samples; replace = false)
        else
            1:n
        end
        sub = df_word[inds, :]
        spikes, durations = get_word_spikes(sub, word; encoding, config)
        labels = reduce(vcat, get_word_labels(sub, word); init = Word[])
        length(spikes) == length(labels) || throw(ErrorException(
            "select_words: $(length(spikes)) spike samples for $(length(labels)) labels of \"$(word)\""))
        append!(all_spikes, spikes)
        append!(all_durations, durations)
        append!(all_labels, labels)
    end
    return (durations = all_durations, spikes = all_spikes, labels = all_labels)
end

"""
    get_encoding_dimension(batch) -> Int

Return the number of input channels of a batch of spike trains (channels of its first sample).
Throws an `ArgumentError` for an empty batch.

# Examples
```julia
using SNNUtils.SpikeTimit
get_encoding_dimension(inputs.spikes)   # 620 for BAE, 35/70 for Cochlea
```
"""
function get_encoding_dimension(batch::AbstractVector)
    isempty(batch) && throw(ArgumentError("get_encoding_dimension: empty batch"))
    return length(batch[1])
end
