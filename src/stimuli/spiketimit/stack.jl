# Stacking of word samples into one long input, and conversion to time steps.

"""
    _times_x_neurons(st) -> (Vector{Float64}, Vector{Vector{Int}})

Return the unique spike times of `st`, sorted, and for each time the channels that fire at it
(ascending).
"""
function _times_x_neurons(st::_Sample)
    d = Dict{Float64,Vector{Int}}()
    for (n, channel) in enumerate(st), t in channel
        push!(get!(Vector{Int}, d, Float64(t)), n)
    end
    times = sort!(collect(keys(d)))
    return times, Vector{Int}[d[t] for t in times]
end

"""
    _sort_spikes!(ft, neurons)

Sort `ft` and `neurons` in place by `ft`; equal times keep their order (stable).
"""
function _sort_spikes!(ft::AbstractVector, neurons::AbstractVector)
    p = sortperm(ft)
    ft .= ft[p]
    neurons .= neurons[p]
    return ft, neurons
end

"""
    _remove_duplicate_ft!(ft, neurons)

Merge every run of equal consecutive times of `ft` into its first entry, appending the neuron
lists in order; the other entries are removed. Mutates the vectors of `neurons`.
"""
function _remove_duplicate_ft!(ft::AbstractVector, neurons::AbstractVector)
    length(ft) == length(neurons) || throw(DimensionMismatch("ft and neurons differ in length"))
    isempty(ft) && return ft, neurons
    k = 1
    for i in 2:length(ft)
        if ft[i] == ft[k]
            append!(neurons[k], neurons[i])
        else
            k += 1
            ft[k] = ft[i]
            neurons[k] = neurons[i]
        end
    end
    resize!(ft, k)
    resize!(neurons, k)
    return ft, neurons
end

"""
    stack_spiketimes(spikes, durations, silence_time::Real, shift::Real) -> (ft, neurons)

Concatenate samples of spike trains into one sequence of firing times.

Sample `i` starts at `g_i`, with `g_1 = shift` and `g_{i+1} = (g_i + durations[i]) + silence_time`.
Its unique spike times are shifted by `g_i`; the result is sorted by time (stable). Equal times
of different samples are not merged.

# Arguments
- `spikes`: vector of spike trains (s), one per sample.
- `durations::AbstractVector{<:Real}`: duration (s) of each sample.
- `silence_time::Real`: silence (s) after each sample, `>= 0`.
- `shift::Real`: onset (s) of the first sample, `>= 0`.

# Returns
- `(ft = Vector{Float64}, neurons = Vector{Vector{Int}})`: firing times (s) and the channels
  firing at each time.

Throws a `DimensionMismatch` if `spikes` and `durations` differ in length, an `ArgumentError`
for negative `silence_time` or `shift`.

# Examples
```julia
using SNNUtils.SpikeTimit
stack_spiketimes([[[0.01], [0.02]], [[0.01]]], [0.1, 0.1], 0.05, 0.0)
# (ft = [0.01, 0.02, 0.16], neurons = [[1], [2], [1]])
```

See also [`stack_labels`](@ref), [`get_ordered_spikes`](@ref).
"""
function stack_spiketimes(spikes::_Batch, durations::AbstractVector{<:Real}, silence_time::Real,
                          shift::Real)
    length(spikes) == length(durations) || throw(DimensionMismatch(
        "stack_spiketimes: $(length(spikes)) samples for $(length(durations)) durations"))
    silence_time < 0 && throw(ArgumentError("silence_time (s) must be >= 0, got $(silence_time)"))
    shift < 0 && throw(ArgumentError("shift (s) must be >= 0, got $(shift)"))
    all_ft = Float64[]
    all_neurons = Vector{Int}[]
    g = Float64(shift)
    for (st, dd) in zip(spikes, durations)
        times, neurons = _times_x_neurons(st)
        append!(all_ft, times .+ g)
        append!(all_neurons, neurons)
        g += Float64(dd)
        g += Float64(silence_time)
    end
    _sort_spikes!(all_ft, all_neurons)
    return (ft = all_ft, neurons = all_neurons)
end

"""
    stack_labels(labels::AbstractVector{Word}, durations, silence_time::Real, shift::Real) -> Transcriptions{Float64}

Build the word and phone transcriptions (s) of the samples stacked by
[`stack_spiketimes`](@ref) with the same arguments.

Word `i` starts at `g_i`, with `g_1 = shift` and `g_{i+1} = g_i + (durations[i] + silence_time)`;
its phones are shifted by `g_i`. (The rounding differs from [`stack_spiketimes`](@ref) by at most
one ulp, as in SpikeTimit.jl 0.1.)

# Arguments
- `labels::AbstractVector{Word}`: labels of the samples (e.g. `select_words(...).labels`).
- `durations`: duration (s) of each sample; must equal `labels[i].duration`.
- `silence_time::Real`, `shift::Real`: as in [`stack_spiketimes`](@ref) (s).

Throws an `ArgumentError` if a label duration differs from the duration, a `DimensionMismatch`
if the lengths differ.
"""
function stack_labels(labels::AbstractVector{Word}, durations::AbstractVector{<:Real},
                      silence_time::Real, shift::Real)
    length(labels) == length(durations) || throw(DimensionMismatch(
        "stack_labels: $(length(labels)) labels for $(length(durations)) durations"))
    words = Transcription()
    phones = Transcription()
    silence = Float64(silence_time)
    g = Float64(shift)
    for (label, dd) in zip(labels, durations)
        label.duration == dd || throw(ArgumentError(
            "stack_labels: label \"$(label.word)\" has duration $(label.duration) s, expected $(dd) s"))
        push!(words.intervals, (g, g + dd))
        push!(words.signs, label.word)
        for ph in label.phones
            push!(phones.intervals, (g + ph.t0, g + ph.t1))
            push!(phones.signs, ph.ph)
        end
        g += dd + silence
    end
    return (words = words, phones = phones)
end

"""
    mix_inputs([rng::AbstractRNG,] samples::Int, repetitions::Int) -> Vector{Int}

Return the indices `1:samples`, each repeated `repetitions` times, in random order.

# Examples
```jldoctest
julia> using SNNUtils.SpikeTimit, Random

julia> sort(mix_inputs(MersenneTwister(1), 3, 2))
6-element Vector{Int64}:
 1
 1
 2
 2
 3
 3
```
"""
function mix_inputs(rng::AbstractRNG, samples::Int, repetitions::Int)
    (samples < 0 || repetitions < 0) && throw(ArgumentError(
        "mix_inputs: samples and repetitions must be >= 0, got $(samples), $(repetitions)"))
    return shuffle(rng, repeat(1:samples, repetitions))
end

mix_inputs(samples::Int, repetitions::Int) = mix_inputs(Random.default_rng(), samples, repetitions)

"""
    get_ordered_spikes(word_inputs; silence_time, shift, ids = Int[], repetitions = -1,
                       rng = Random.default_rng()) -> (ordered_spikes, transcriptions)

Stack the samples `ids` of `word_inputs` (the output of [`select_words`](@ref)) into one input
sequence, with [`stack_spiketimes`](@ref) and [`stack_labels`](@ref).

If `ids` is empty, every sample is used `repetitions` times in random order
([`mix_inputs`](@ref) with `rng`).

# Arguments
- `word_inputs::NamedTuple`: fields `spikes`, `durations`, `labels`.
- `silence_time::Real`: silence (s) after each sample.
- `shift::Real`: onset (s) of the first sample.
- `ids::AbstractVector{<:Integer} = Int[]`: samples to stack, in order.
- `repetitions::Int = -1`: repetitions per sample when `ids` is empty.
- `rng::AbstractRNG`: random number generator for the order.

# Returns
- `ordered_spikes = (ft, neurons)`: firing times (s) and channels;
- `transcriptions::Transcriptions{Float64}`: word and phone intervals (s).

Throws an `ArgumentError` if `ids` is empty and `repetitions < 0`.

# Examples
```julia
using SNNUtils.SpikeTimit, Random
inputs = select_words(df, ["that", "had"]; encoding = Cochlea(70))
spikes, trans = get_ordered_spikes(inputs; silence_time = 0.1, shift = 0.0, repetitions = 3,
                                   rng = MersenneTwister(1))
```
"""
function get_ordered_spikes(word_inputs::NamedTuple; silence_time::Real, shift::Real,
                            ids::AbstractVector{<:Integer} = Int[], repetitions::Int = -1,
                            rng::AbstractRNG = Random.default_rng())
    if isempty(ids)
        repetitions < 0 &&
            throw(ArgumentError("Define 'repetitions' per sample or set the 'ids' variable"))
        ids = mix_inputs(rng, length(word_inputs.durations), repetitions)
    end
    (; spikes, durations, labels) = word_inputs
    ordered = stack_spiketimes(spikes[ids], durations[ids], silence_time, shift)
    transcriptions = stack_labels(labels[ids], durations[ids], silence_time, shift)
    return ordered, transcriptions
end

function _transcription_dt(t::Transcription{Float64}, dt::Real)
    intervals = NTuple{2,Int}[(round(Int, a * 1000 / dt), round(Int, b * 1000 / dt))
                              for (a, b) in t.intervals]
    return Transcription{Int}(intervals, copy(t.signs))
end

"""
    transcriptions_dt(data::Transcriptions{Float64}; dt::Real = 0.1) -> Transcriptions{Int}

Convert word and phone intervals from seconds to integer time steps of `dt` ms:
`round(Int, t * 1000 / dt)`.

# Examples
```julia
using SNNUtils.SpikeTimit
_, trans = get_ordered_spikes(inputs; silence_time = 0.1, shift = 0.0, repetitions = 1)
transcriptions_dt(trans; dt = 0.1)
```
"""
function transcriptions_dt(data::Transcriptions{Float64}; dt::Real = 0.1)
    return (words = _transcription_dt(data.words, dt), phones = _transcription_dt(data.phones, dt))
end

"""
    ft_dt(data::NamedTuple{(:ft, :neurons)}; dt::Real = 0.1, unit = :s) -> (ft = Vector{Int}, neurons = Vector{Vector{Int}})

Convert firing times to integer time steps of `dt` ms (`round(Int, ft * 1000 / dt)` for `unit =
:s`, `round(Int, ft / dt)` for `unit = :ms`) and merge the entries that fall in the same step
(neuron lists concatenated in order). The input is not modified.

# Arguments
- `data`: `(ft, neurons)`, e.g. the output of [`stack_spiketimes`](@ref) (sorted).
- `dt::Real = 0.1`: time step (ms).
- `unit = :s`: unit of `data.ft`, `:s`/`"s"` or `:ms`/`"ms"`; other values throw a
  `DomainError`.

# Examples
```julia
using SNNUtils.SpikeTimit
ft_dt((ft = [0.001, 0.00102, 0.002], neurons = [[1], [2], [3]]))   # (ft = [10, 20], neurons = [[1, 2], [3]])
```
"""
function ft_dt(data::NamedTuple{(:ft, :neurons)}; dt::Real = 0.1,
               unit::Union{Symbol,AbstractString} = :s)
    u = String(unit)
    factor = u == "s" ? 1000 / dt :
             u == "ms" ? 1 / dt : throw(DomainError(unit, "Valid units are 's' and 'ms'"))
    ft = Int[round(Int, t * factor) for t in data.ft]
    neurons = Vector{Int}[copy(n) for n in data.neurons]
    _remove_duplicate_ft!(ft, neurons)
    return (ft = ft, neurons = neurons)
end

function _savepoints(t::Transcription{Int}, n::Int)
    n >= 1 || throw(ArgumentError("number of save points must be >= 1, got $(n)"))
    measures = Matrix{Int}(undef, length(t), n)
    for (i, (a, b)) in enumerate(t.intervals)
        measures[i, :] = a .+ (1:n) .* floor(Int, (b - a) / n)
    end
    return measures
end

"""
    get_savepoints(trans::Transcriptions{Int}; per_word::Int, per_phone::Int) -> (words = Matrix{Int}, phones = Matrix{Int})

Return evenly spaced time steps inside every word and phone interval (in steps, from
[`transcriptions_dt`](@ref)): for an interval `(a, b)` and `n` points, `a .+ (1:n) .* floor(Int,
(b - a) / n)`. One row per interval. Throws an `ArgumentError` if `per_word` or `per_phone` is
`< 1`.

# Examples
```julia
using SNNUtils.SpikeTimit
trans = (words = Transcription([(0, 100)], ["a"]), phones = Transcription([(0, 100)], ["x"]))
get_savepoints(trans; per_word = 4, per_phone = 2).words   # [25 50 75 100]
```
"""
function get_savepoints(trans::Transcriptions{Int}; per_word::Int, per_phone::Int)
    return (words = _savepoints(trans.words, per_word), phones = _savepoints(trans.phones, per_phone))
end

"""
    matrix2spiketimes(matrix::AbstractMatrix{Bool}, dt::Real) -> (spiketimes = SpikeTrains{Float64}, duration = Float64)

Convert a binary matrix (rows = channels, columns = time bins of width `dt`) into spike trains:
a spike in column `j` is at time `j * dt`. The duration is `size(matrix, 2) * dt`. Times are in
the unit of `dt` (s if `dt` is in s).

# Examples
```jldoctest
julia> using SNNUtils.SpikeTimit

julia> st, T = matrix2spiketimes(BitMatrix([1 0 1; 0 1 0]), 0.5);

julia> st
2-element Vector{Vector{Float64}}:
 [0.5, 1.5]
 [1.0]

julia> T
1.5
```
"""
function matrix2spiketimes(matrix::AbstractMatrix{Bool}, dt::Real)
    spiketimes = SpikeTrains{Float64}([Float64[i * dt for i in findall(r)] for r in eachrow(matrix)])
    return (spiketimes = spiketimes, duration = Float64(size(matrix, 2) * dt))
end

"""
    spiketimes2sequence(st) -> (ft = Vector{Float64}, neurons = Vector{Vector{Int}})

Return the sorted unique spike times of the spike trains `st` and the channels firing at each
time; same as `stack_spiketimes([st], [0.0], 0.0, 0.0)`.

# Examples
```julia
using SNNUtils.SpikeTimit
spiketimes2sequence([[0.5, 0.25], [0.25]])   # (ft = [0.25, 0.5], neurons = [[1, 2], [1]])
```
"""
spiketimes2sequence(st::_Sample) = stack_spiketimes([st], [0.0], 0.0, 0.0)
