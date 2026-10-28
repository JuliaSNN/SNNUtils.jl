"""
    SpikeTrains{T} = Vector{Vector{T}}

Spike trains of one sample: `st[c]` holds the spike times of input channel (neuron) `c`. In
SpikeTimit `T = Float64` and the times are in seconds. Not to be confused with
`SNNModels.Spiketimes` (`Vector{Vector{Float32}}`, ms); convert with [`to_snn_spiketimes`](@ref).
"""
const SpikeTrains{T} = Vector{Vector{T}}

"""
    Phone(ph::String, t0::Float64, t1::Float64)

A TIMIT phone inside a [`Word`](@ref).

# Fields
- `ph::String`: TIMIT phone label (e.g. `"dh"`).
- `t0::Float64`: onset (s), relative to the onset of the word.
- `t1::Float64`: offset (s), relative to the onset of the word.
"""
Phone

struct Phone
    ph::String
    t0::Float64
    t1::Float64
end

"""
    Word(word, phones, duration, t0, t1)
    Word(; word = "", phones = Phone[], duration = -1.0, t0 = -1.0, t1 = -1.0)

One occurrence of a word in a sentence, with its phones.

# Fields
- `word::String`: the word.
- `phones::Vector{Phone}`: phones contained in the word, times relative to `t0` (s).
- `duration::Float64`: `t1 - t0` (s).
- `t0::Float64`: onset in the sentence (s).
- `t1::Float64`: offset in the sentence (s).

Two `Word`s are `==` (and hash equally) when all their fields are equal.

See also [`get_word_labels`](@ref), [`stack_labels`](@ref).
"""
Word

struct Word
    word::String
    phones::Vector{Phone}
    duration::Float64
    t0::Float64
    t1::Float64
end

Word(; word = "", phones = Phone[], duration = -1.0, t0 = -1.0, t1 = -1.0) =
    Word(word, phones, duration, t0, t1)

Base.:(==)(a::Word, b::Word) =
    a.word == b.word && a.phones == b.phones && a.duration == b.duration && a.t0 == b.t0 &&
    a.t1 == b.t1
Base.hash(w::Word, h::UInt) = hash((w.word, w.phones, w.duration, w.t0, w.t1), hash(:Word, h))

"""
    Transcription{T}(intervals, signs)
    Transcription(intervals::Vector{NTuple{2,T}}, signs)
    Transcription{T}()
    Transcription()

Labelled time intervals: `intervals[i] = (t0, t1)` is the interval of the label `signs[i]`.

The unit of the intervals is seconds for `T = Float64` (as read by [`create_dataset`](@ref) or
built by [`stack_labels`](@ref)) and integer time steps for `T = Int` (see
[`transcriptions_dt`](@ref)). `Transcription()` is an empty `Transcription{Float64}`.
`length(t)` is the number of labels. Throws an `ArgumentError` if `intervals` and `signs`
have different lengths.

# Fields
- `intervals::Vector{NTuple{2,T}}`: `(t0, t1)` of every label.
- `signs::Vector{String}`: the labels (words or phones).
"""
Transcription

struct Transcription{T<:Real}
    intervals::Vector{NTuple{2,T}}
    signs::Vector{String}
    function Transcription{T}(intervals::AbstractVector, signs::AbstractVector) where {T<:Real}
        length(intervals) == length(signs) || throw(ArgumentError(
            "Transcription: $(length(intervals)) intervals for $(length(signs)) labels"))
        return new{T}(intervals, signs)
    end
end

Transcription(intervals::AbstractVector{NTuple{2,T}}, signs::AbstractVector) where {T<:Real} =
    Transcription{T}(intervals, signs)
Transcription{T}() where {T<:Real} = Transcription{T}(NTuple{2,T}[], String[])
Transcription() = Transcription{Float64}()

Base.length(t::Transcription) = length(t.signs)
Base.:(==)(a::Transcription, b::Transcription) = a.intervals == b.intervals && a.signs == b.signs
Base.hash(t::Transcription, h::UInt) = hash((t.intervals, t.signs), hash(:Transcription, h))

"""
    Transcriptions{T} = NamedTuple{(:words, :phones), Tuple{Transcription{T}, Transcription{T}}}

Word and phone transcriptions of the same time line, e.g. the output of
[`stack_labels`](@ref) (`T = Float64`, s) or [`transcriptions_dt`](@ref) (`T = Int`, steps).
"""
const Transcriptions{T} = NamedTuple{(:words, :phones),Tuple{Transcription{T},Transcription{T}}}
