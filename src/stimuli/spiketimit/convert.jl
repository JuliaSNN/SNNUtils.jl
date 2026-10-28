# Conversion of SpikeTimit data (Float64, s) to SNNModels stimuli and SNNUtils sequences
# (Float32, ms).

"""
    to_snn_spiketimes(st) -> SNNModels.Spiketimes

Convert spike trains from seconds to the SNNModels format `Vector{Vector{Float32}}` in
milliseconds (`Float32(t * 1000)`). Channel order and spike order are kept; `st` is not
modified. Use it to plot SpikeTimit spikes with `SNNPlots.raster`.

# Examples
```jldoctest
julia> using SNNUtils.SpikeTimit

julia> to_snn_spiketimes([[0.5, 0.25], Float64[]])
2-element Vector{Vector{Float32}}:
 [500.0, 250.0]
 []
```

See also [`spiketime_parameter`](@ref).
"""
function to_snn_spiketimes(st::_Sample)
    return Vector{Float32}[Float32[Float32(t * S_TO_MS) for t in channel] for channel in st]
end

"""
    spiketime_parameter(st; delay::Real = 0.0) -> SNNModels.SpikeTimeStimulusParameter
    spiketime_parameter(seq::NamedTuple{(:ft, :neurons)}; delay::Real = 0.0) -> SNNModels.SpikeTimeStimulusParameter

Build the parameter of an `SNNModels.SpikeTimeStimulus` from SpikeTimit spikes: `Float32` spike
times in ms, sorted by time (the order of equal times is unspecified), each with its input
neuron (the channel index, 1-based).

# Arguments
- `st`: spike trains (s) of one sample; neuron `c` fires at the times of `st[c]`.
- `seq`: firing times (s) and channels, e.g. the first output of [`get_ordered_spikes`](@ref)
  or [`spiketimes2sequence`](@ref).
- `delay::Real = 0.0`: added to every spike time (ms).

# Returns
- `SNNModels.SpikeTimeStimulusParameter{Vector{Float32},Vector{Int}}`.

Throws an `ArgumentError` for a neuron id `< 1` in `seq`.

!!! note
    `SNNModels.SpikeTimeStimulus` takes the number of input neurons from the largest neuron id
    by default, which undercounts if the top channels never fire. Pass
    `N = get_encoding_dimension(spikes)` explicitly.

# Examples
```julia
using SNNModels, SNNUtils.SpikeTimit
inputs = select_words(df, ["that", "had"]; encoding = Cochlea(70))
spikes, trans = get_ordered_spikes(inputs; silence_time = 0.1, shift = 0.0, repetitions = 2)
E = SNNModels.IF(N = 100)
stim = SNNModels.SpikeTimeStimulus(E, :ge; param = spiketime_parameter(spikes),
                                   conn = (p = 0.2, μ = 1.0),
                                   N = get_encoding_dimension(inputs.spikes))
```

See also [`to_snn_spiketimes`](@ref), [`to_sequence`](@ref).
"""
function spiketime_parameter(st::_Sample; delay::Real = 0.0)
    param = SNNModels.SpikeTimeParameter(to_snn_spiketimes(st))
    delay != 0 && SNNModels.shift_spikes!(param, delay)
    return param
end

function spiketime_parameter(seq::NamedTuple{(:ft, :neurons)}; delay::Real = 0.0)
    times = Float32[]
    neurons = Int[]
    for (t, ns) in zip(seq.ft, seq.neurons), n in ns
        n >= 1 || throw(ArgumentError("spiketime_parameter: neuron ids must be >= 1, got $(n)"))
        push!(times, Float32(t * S_TO_MS))
        push!(neurons, n)
    end
    param = SNNModels.SpikeTimeParameter(times, neurons)
    delay != 0 && SNNModels.shift_spikes!(param, delay)
    return param
end

"""
    ms_intervals(trans::Transcription{<:AbstractFloat}) -> Vector{Vector{Float32}}

Return every interval of `trans` (s) as `[t0, t1]` in ms (`Float32`), the interval format of
SNNUtils (`sign_intervals`) and `SNNModels.set_intervals!`.

# Examples
```julia
using SNNUtils.SpikeTimit
ms_intervals(Transcription([(0.1, 0.25)], ["she"]))   # [[100.0, 250.0]]
```

See also [`label_intervals`](@ref).
"""
function ms_intervals(trans::Transcription{<:AbstractFloat})
    return Vector{Float32}[Float32[Float32(a * S_TO_MS), Float32(b * S_TO_MS)]
                           for (a, b) in trans.intervals]
end

"""
    label_intervals(trans::Transcription{<:AbstractFloat}, label) -> Vector{Vector{Float32}}

Return the intervals of `label` (a `String` or `Symbol`) in `trans` as `[t0, t1]` in ms
(`Float32`), in order; empty if the label is absent. Same format as SNNUtils `sign_intervals`:
use it to set the active intervals of stimuli or to select the activity during a word.

# Examples
```jldoctest
julia> using SNNUtils.SpikeTimit

julia> t = Transcription([(0.1, 0.25), (0.3, 0.5)], ["she", "had"]);

julia> label_intervals(t, :had)
1-element Vector{Vector{Float32}}:
 [300.0, 500.0]
```

See also [`ms_intervals`](@ref), [`to_sequence`](@ref).
"""
function label_intervals(trans::Transcription{<:AbstractFloat}, label::Union{AbstractString,Symbol})
    l = string(label)
    return Vector{Float32}[Float32[Float32(a * S_TO_MS), Float32(b * S_TO_MS)]
                           for ((a, b), s) in zip(trans.intervals, trans.signs) if s == l]
end

# Elements (word, phoneme, duration in s) of the sequence built by to_sequence.
function _sequence_elements(trans::Transcriptions{Float64}, tail::Float64)
    W, P = trans.words, trans.phones
    silence = :_
    elements = Tuple{Symbol,Symbol,Float64}[]
    t = 0.0
    k = 1
    np = length(P)
    for (i, ((w0, w1), ws)) in enumerate(zip(W.intervals, W.signs))
        w0 <= w1 || throw(ArgumentError("to_sequence: word $i (\"$(ws)\") has t0 > t1"))
        w0 >= t || throw(ArgumentError(
            "to_sequence: word $i (\"$(ws)\", $(w0)-$(w1) s) overlaps the previous word or is not sorted"))
        wsym = Symbol(ws)
        push!(elements, (silence, silence, w0 - t))
        c = w0
        while k <= np
            (p0, p1) = P.intervals[k]
            if p0 >= w0 && p1 <= w1
                (p0 >= c && p0 <= p1) || throw(ArgumentError(
                    "to_sequence: phone $k (\"$(P.signs[k])\", $(p0)-$(p1) s) overlaps the previous phone or is not sorted"))
                p0 > c && push!(elements, (wsym, silence, p0 - c))
                push!(elements, (wsym, Symbol(P.signs[k]), p1 - p0))
                c = p1
                k += 1
            elseif p0 >= w1
                break
            else
                throw(ArgumentError(
                    "to_sequence: phone $k (\"$(P.signs[k])\", $(p0)-$(p1) s) is not contained in word $i (\"$(ws)\", $(w0)-$(w1) s)"))
            end
        end
        w1 > c && push!(elements, (wsym, silence, w1 - c))
        t = w1
    end
    k <= np && throw(ArgumentError(
        "to_sequence: phone $k (\"$(P.signs[k])\", $(P.intervals[k][1])-$(P.intervals[k][2]) s) is outside every word"))
    push!(elements, (silence, silence, tail))
    return elements
end

# Type row of the sequence, with the rule of SNNUtils generate_sequence.
function _sequence_types(words::AbstractVector{Symbol}, phonemes::AbstractVector{Symbol}, silence::Symbol)
    n = length(words)
    types = fill(:silence, n)
    a = 1
    while a <= n
        if words[a] == silence
            a += 1
            continue
        end
        b = a
        while b < n && words[b + 1] == words[a]
            b += 1
        end
        types[b] = :offset
        for j in 1:(b - a - 1)
            types[b - j] = Symbol("offset_$(j + 1)")
        end
        types[a] = :onset
        a = b + 1
    end
    for i in 1:n
        phonemes[i] == silence && (types[i] = :silence)
    end
    return types
end

"""
    to_sequence(trans::Transcriptions{Float64}; tail::Real = 0.0) -> NamedTuple

Build an SNNUtils sequence from real word and phone timing (s), so that the SNNUtils sequence
functions (`sign_intervals`, `merge_intervals`, `all_intervals`, `sequence_end`,
`stimuli_names`, `update_stimuli!`, `set_stimuli!`) work on TIMIT speech.

Each word becomes a run of elements `(word, phone, duration)`: one per phone, plus `(word, :_)`
elements for the gaps between phones. Every word is preceded by a silence element `(:_, :_)`
(also of zero duration, so that repeated words stay separate presentations) and the sequence
ends with a silence of `tail` s. Durations are converted to `Float32` ms; onsets and offsets
are their cumulative sums (Float32 accumulation: drift < 1e-3 ms for sequences < 100 s).

# Arguments
- `trans::Transcriptions{Float64}`: words and phones (s), e.g. from
  [`get_ordered_spikes`](@ref). Words must be sorted and non-overlapping; every phone must lie
  inside a word (phones outside words, such as `h#` or `pau` of a raw sentence, must be removed
  first), sorted and non-overlapping.
- `tail::Real = 0.0`: silence (s) after the last word, `>= 0`.

# Returns
`NamedTuple` with the fields of `SNNUtils.generate_sequence`:
- `dict::Dict{Symbol,Vector{Symbol}}`: phones of the first occurrence of each word;
- `symbols = (phonemes, words)`: sorted unique symbols (silence `:_` excluded);
- `ph_duration::Dict{Symbol,Float32}`: mean duration (ms) of each phone and of silence
  (informational);
- `silence = :_`;
- `sequence::Matrix{Any}`: 6 x n, rows words, phonemes, duration (ms), type (`:onset`,
  `:offset_j`, `:offset`, `:silence`), onset (ms), offset (ms);
- `line_id = (words = 1, phonemes = 2, duration = 3, type = 4, onset = 5, offset = 6)`.

Throws an `ArgumentError` if the preconditions above are violated or `tail < 0`.

!!! warning
    `SNNUtils.sign_intervals` looks a symbol up among the phonemes first: a word spelled like a
    phone label resolves to the phone. `to_sequence` warns about such collisions.

# Examples
```julia
using SNNUtils, SNNUtils.SpikeTimit
spikes, trans = get_ordered_spikes(inputs; silence_time = 0.1, shift = 0.0, repetitions = 2)
seq = to_sequence(trans; tail = 0.2)
SNNUtils.sign_intervals(:that, seq)     # ms, same as label_intervals(trans.words, "that")
SNNUtils.sequence_end(seq)             # simulation length, ms
```

See also [`label_intervals`](@ref), [`spiketime_parameter`](@ref).
"""
function to_sequence(trans::Transcriptions{Float64}; tail::Real = 0.0)
    tail < 0 && throw(ArgumentError("to_sequence: tail (s) must be >= 0, got $(tail)"))
    silence = :_
    elements = _sequence_elements(trans, Float64(tail))
    words = Symbol[e[1] for e in elements]
    phonemes = Symbol[e[2] for e in elements]
    durations = Float32[Float32(e[3] * S_TO_MS) for e in elements]
    offsets = cumsum(durations)
    onsets = [0.0f0; offsets[1:(end - 1)]]

    sequence = Matrix{Any}(undef, 6, length(elements))
    sequence[1, :] .= words
    sequence[2, :] .= phonemes
    sequence[3, :] .= durations
    sequence[4, :] .= _sequence_types(words, phonemes, silence)
    sequence[5, :] .= onsets
    sequence[6, :] .= offsets

    # phones of the first occurrence (first run) of each word
    dict = Dict{Symbol,Vector{Symbol}}()
    i = 1
    while i <= length(words)
        w = words[i]
        if w == silence
            i += 1
            continue
        end
        j = i
        while j < length(words) && words[j + 1] == w
            j += 1
        end
        haskey(dict, w) || (dict[w] = Symbol[p for p in phonemes[i:j] if p != silence])
        i = j + 1
    end

    all_phonemes = sort(unique(p for p in phonemes if p != silence))
    all_words = sort(unique(w for w in words if w != silence))

    sums = Dict{Symbol,Float64}()
    counts = Dict{Symbol,Int}()
    for (p, d) in zip(phonemes, durations)
        d > 0 || continue
        sums[p] = get(sums, p, 0.0) + d
        counts[p] = get(counts, p, 0) + 1
    end
    ph_duration = Dict{Symbol,Float32}(p => Float32(sums[p] / counts[p]) for p in keys(sums))

    collisions = intersect(all_words, all_phonemes)
    isempty(collisions) || @warn "to_sequence: words spelled like phone labels resolve to the phone " *
                                 "in SNNUtils.sign_intervals: $(collisions)"

    return (dict = dict, symbols = (phonemes = Vector{Symbol}(all_phonemes), words = Vector{Symbol}(all_words)),
            ph_duration = ph_duration, silence = silence, sequence = sequence,
            line_id = (words = 1, phonemes = 2, duration = 3, type = 4, onset = 5, offset = 6))
end
