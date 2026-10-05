using StatsBase

"""
    generate_lexicon(config)

Build a lexicon from a word-to-phoneme dictionary.

# Arguments
- `config`: any object with the fields (or keys accessible with `@unpack`)
    - `dictionary::Dict{Symbol,Vector{Symbol}}`: word => its sequence of phonemes;
    - `ph_duration`: duration of each phoneme in ms, normally a `Dict{Symbol,Float32}` that
      also contains the silence symbol `:_` (see [`getduration`](@ref)).

# Returns
A `NamedTuple` with fields
- `dict`: the input dictionary;
- `symbols = (phonemes, words)`: sorted vectors of the unique phonemes and words (the silence
  symbol is not included);
- `ph_duration`: the input durations;
- `silence = :_`: the silence symbol.

# Example
```julia
using SNNUtils
dictionary = getdictionary([:AB, :BA])
lexicon = generate_lexicon((ph_duration = getduration(dictionary, 50.0), dictionary = dictionary))
```
"""
function generate_lexicon(config)
    @unpack ph_duration, dictionary = config

    all_words = collect(keys(dictionary)) |> Set |> collect |> sort |> Vector{Symbol}
    all_phonemes =
        collect(values(dictionary)) |>
        Iterators.flatten |>
        Set |>
        collect |>
        sort |>
        Vector{Symbol}
    symbols = collect(union(all_words, all_phonemes))

    ## Add the silence symbol
    silence_symbol = :_

    return (
        dict = dictionary,
        symbols = (phonemes = all_phonemes, words = all_words),
        ph_duration = ph_duration,
        silence = silence_symbol,
    )
end

"""
    generate_sequence(seq_function::Function; lexicon::NamedTuple, seed = -1, kwargs...)

Generate a timed sequence of words and phonemes.

`seq_function(; lexicon, kwargs...)` must return `(words, phonemes, seq_length)`: two vectors of
equal length with the word and the phoneme of every sequence element, and their length
([`word_phonemes_sequence`](@ref) is the generator provided by SNNUtils). If `seed > 0` the
global RNG is seeded with `Random.seed!(seed)` first.

The sequence is stored as a `6 x (seq_length + 2)` `Matrix{Any}`, with a silence element added at
the beginning and at the end. Rows (see `line_id`):
1. `words`: word symbol of each element (`:_` for silence);
2. `phonemes`: phoneme symbol;
3. `duration`: duration in ms, `lexicon.ph_duration[phoneme]`;
4. `type`: `:onset` for the first phoneme of a word, `:offset` for the last one, `:offset_j`
   for the intermediate ones of longer words (counted backwards from the end), `:silence`;
5. `onset`: start time (ms) of the element;
6. `offset`: end time (ms) of the element.

# Returns
The lexicon `NamedTuple` extended with `sequence` (the matrix above) and
`line_id = (words = 1, phonemes = 2, duration = 3, type = 4, onset = 5, offset = 6)`.

# Example
```julia
using SpikingNeuralNetworks, SNNUtils
SNN.@load_units
lexicon = get_lexicon([:AB, :BA, :CC], 50ms)
seq = generate_sequence(word_phonemes_sequence; lexicon, presentations = 6, mode = :fixed,
                        weights = Dict(:AB => 1.0, :BA => 1.0, :CC => 1.0), seed = 1)
seq.sequence[seq.line_id.words, :]
```
"""
function generate_sequence(
    seq_function::Function;
    lexicon::NamedTuple,
    seed = -1,
    kwargs...,
)
    (seed > 0) && (Random.seed!(seed))

    words, phonemes, seq_length = seq_function(; lexicon = lexicon, kwargs...)

    @unpack dict, symbols, silence, ph_duration = lexicon
    ## create the populations
    ## sequence from the initial word sequence
    word_line = 1
    phoneme_line = 2
    duration_line = 3
    type_line = 4
    onset_line = 5
    offset_line = 6
    max_word_length = 20

    sequence = Matrix{Any}(fill(silence, 6, seq_length+2))
    sequence[word_line, 1] = silence
    sequence[phoneme_line, 1] = silence
    sequence[duration_line, 1] = ph_duration[silence]
    for (n, (w, p)) in enumerate(zip(words, phonemes))
        sequence[word_line, 1+n] = w
        sequence[phoneme_line, 1+n] = p
        sequence[duration_line, 1+n] = ph_duration[p]
    end
    sequence[word_line, end] = silence
    sequence[phoneme_line, end] = silence
    sequence[duration_line, end] = ph_duration[silence]

    sequence[type_line, :] .= :silence
    n = 2
    while n < size(sequence, 2)
        word_offset = (sequence[word_line, n+1] !== sequence[word_line, n]) && (sequence[word_line, n] !== silence)
        if word_offset
            sequence[type_line, n] = :offset
            j = 1
            while sequence[word_line, n] == sequence[word_line, n-j]
                sequence[type_line, n-j] = Symbol("offset_$(j+1)")
                j += 1
            end
            j -= 1
            sequence[type_line, n-j] = :onset
        end
        if sequence[phoneme_line, n] == :_
            sequence[type_line, n] = :silence
        end
        n += 1
    end


    sequence[5, :] .= [0ms, cumsum(sequence[3, 1:end])[1:end-1]...]
    sequence[6, :] .= [cumsum(sequence[3, 1:end])...]

    line_id = (words = 1, phonemes = 2, duration = 3, type = 4, onset = 5, offset = 6)
    sequence = (; lexicon..., sequence = sequence, line_id = line_id)
end


"""
    sign_intervals(sign::Symbol, sequence)

Return the time intervals during which the word or phoneme `sign` is presented in `sequence`.

The row of the sequence is selected by looking `sign` up in `sequence.symbols` (`words` or
`phonemes`). Each sequence element equal to `sign` contributes one interval
`[t_start, t_end]` (ms); consecutive elements are not merged, so a two-phoneme word yields two
adjacent intervals per presentation (see [`merge_intervals`](@ref)). Throws an error if `sign` is
neither a word nor a phoneme of the lexicon.

# Returns
`Vector{Vector{Float32}}` of `[start, end]` intervals in ms.

# Example
```julia
using SpikingNeuralNetworks, SNNUtils
SNN.@load_units
lexicon = get_lexicon([:AB, :BA], 50ms)
seq = generate_sequence(word_phonemes_sequence; lexicon, presentations = 4, mode = :fixed,
                        weights = Dict(:AB => 1.0, :BA => 1.0))
sign_intervals(:AB, seq)
```
"""
function sign_intervals(sign::Symbol, sequence)
    @unpack dict, sequence, symbols, line_id = sequence

    ## Identify the line of the sequence that contains the sign
    sign_line_id = -1
    for k in keys(symbols)
        if sign in getfield(symbols, k)
            sign_line_id = getfield(line_id, k)
            break
        end
    end
    if sign_line_id == -1
        throw(ErrorException("Sign index not found"))
    end

    ## Find the intervals where the sign is present
    intervals = Vector{Vector{Float32}}()

    my_seq = sequence[sign_line_id, :]
    time_counter = 0
    for i in eachindex(my_seq)
        interval_start = time_counter
        if my_seq[i] == sign
            interval_end = time_counter + sequence[line_id.duration, i]
        else
            interval_end = time_counter
        end
        if interval_end > interval_start
            interval = [interval_start, interval_end]
            push!(intervals, interval)
        end
        time_counter += sequence[line_id.duration, i]
    end
    # while !isnothing(_end)  || !isnothing(_start)
    #     _start = findfirst(x -> x == sign, my_seq[_end:end])
    #     if isnothing(_start)
    #         break
    #     else
    #         _start += _end-1
    #     end
    #     _end  = findfirst(x -> x != sign, my_seq[_start:end]) + _start - 1
    #     interval[1] = cum_duration[_start] - sequence[line_id.duration,_start]
    #     interval[2] = cum_duration[_end-1]
    # end
    return intervals
end


"""
    merge_intervals(intervals::Vector{Vector{Float32}}, skip = nothing)

Merge consecutive intervals that touch (`end` of one equal to `start` of the next) into a single
interval. `skip` is unused.

`merge_intervals([[0, 1], [3, 4], [5, 6]])` returns the three intervals. (Up to SNNUtils 0.2.9
the last interval was dropped when it did not touch the previous one.)

# Example
```julia
using SNNUtils
merge_intervals([[0f0, 1f0], [1f0, 2f0]])   # [[0.0, 2.0]]
```
"""
function merge_intervals(intervals::Vector{Vector{Float32}}, skip=nothing)
    merged_intervals = Vector{Vector{Float32}}()
    all_intervals = length(intervals)
    all_intervals == 0 && return merged_intervals

    current_start = :new_item
    current_end = nothing
    i = 0 
    while i < all_intervals
        i+=1
        local_start, local_end = intervals[i]
        if current_start == :new_item
            current_start = local_start
            current_end = local_end
            continue
        end
        ## if the current end is the same as the local start, we merge the intervals by updating the current end to the local end
        if current_end == local_start
            current_end = local_end
        ## if the current end is different from the local start, we push the current interval to the merged intervals and start a new interval with the local start and local end
        else
            push!(merged_intervals, [current_start, current_end])
            current_start = local_start
            current_end = local_end
            # @info "Merged $(length(merged_intervals)) intervals: $current_start, $current_end at index $i"
        end
    end
    # @info "Last interval: $current_start, $current_end"
    push!(merged_intervals, [current_start, current_end])   
    merged_intervals
end


"""
    all_intervals(sym::Symbol, sequence; interval::Vector = [-50ms, 100ms])

For every symbol of the class `sym` (`:words` or `:phonemes`) and every interval returned by
[`sign_intervals`](@ref), return the window `interval_end .+ interval` (ms) around the end of the
interval, together with the symbol.

# Returns
`(offsets, ys)`: a `Vector{Vector{Float32}}` of windows and the `Vector{Symbol}` of the
corresponding symbols, grouped by symbol.
"""
function all_intervals(sym::Symbol, sequence; interval::Vector = [-50ms, 100ms])
    offsets = Vector{Vector{Float32}}()
    ys = Vector{Symbol}()
    symbols = getfield(sequence.symbols, sym)
    for word in symbols
        for myinterval in sign_intervals(word, sequence)
            offset = myinterval[end] .+ interval
            push!(offsets, offset)
            push!(ys, word)
        end
    end
    return offsets, ys
end



"""
    sequence_end(seq)

Return the total duration of the sequence in ms, i.e. the sum of the `duration` row of
`seq.sequence`.
"""
function sequence_end(seq)
    @unpack line_id, sequence = seq
    return sum(sequence[line_id.duration, :])
end

"""
    time_in_interval(x::Float32, intervals::Vector{Vector{Float32}})

Return `true` if `interval[1] <= x <= interval[2]` for at least one interval (bounds included),
`false` otherwise.
"""
function time_in_interval(x::Float32, intervals::Vector{Vector{Float32}})
    for interval in intervals
        if x >= interval[1] && x <= interval[2]
            return true
        end
    end
    return false
end

"""
    start_interval(x::Float32, intervals::Vector{Vector{Float32}})

Return the start of the first interval that contains `x` (bounds included), or `-1` if `x` is in
none of the intervals.
"""
function start_interval(x::Float32, intervals::Vector{Vector{Float32}})
    for interval in intervals
        if x >= interval[1] && x <= interval[2]
            return interval[1]
        end
    end
    return -1
end

"""
    getdictionary(words::Vector{T}, insert = nothing) where {T<:Union{String,Symbol}}

Create a `Dict{Symbol,Vector{Symbol}}` mapping each word to the vector of its characters, used as
phonemes. If `insert` is given, `Symbol(insert)` is placed between consecutive characters
(e.g. `getdictionary(["ab"], :_)` gives `:ab => [:a, :_, :b]`).

# Example
```julia
using SNNUtils
getdictionary([:AB, :CD])   # Dict(:AB => [:A, :B], :CD => [:C, :D])
```
"""
function getdictionary(words::Vector{T}, insert=nothing) where {T<:Union{String,Symbol}}
    dict = Dict{Symbol,Vector{Symbol}}()
    if !isnothing(insert)
        for word in words
            phonemes = vcat([[Symbol(x), Symbol(insert)] for x in collect(string(word))]...)[1:end-1]
            push!(dict, Symbol(word) => phonemes)
        end
    else 
        for word in words
            phonemes = [Symbol(x) for x in collect(string(word))]
            push!(dict, Symbol(word) => phonemes)
        end
    end
    dict
end

"""
    getphonemes(dictionary::Dict{Symbol,Vector{Symbol}})

Return the unique phonemes occurring in `dictionary`, followed by the silence symbol `:_`.
"""
function getphonemes(dictionary::Dict{Symbol,Vector{Symbol}})
    phs = collect(unique(vcat(values(dictionary)...)))
    push!(phs, :_)
    return phs
end

function getwords(dictionary::Dict{Symbol,Vector{Symbol}})
    phs = collect(unique(vcat(keys(dictionary)...)))
    push!(phs, :_)
    return phs
end


"""
    get_lexicon(words, duration, insert = nothing)

Convenience constructor of a lexicon: `getdictionary(words, insert)`, then
`getduration(dictionary, duration)`, then [`generate_lexicon`](@ref).

`duration` is either a number (the same duration, in ms, for every phoneme and for silence) or a
`NamedTuple` with one entry per phoneme plus `silence` (see [`getduration`](@ref)).

# Example
```julia
using SpikingNeuralNetworks, SNNUtils
SNN.@load_units
lexicon = get_lexicon([:AB, :BA, :CC], 50ms)
lexicon.symbols   # (phonemes = [:A, :B, :C], words = [:AB, :BA, :CC])
```
"""
function get_lexicon(words, duration, insert = nothing)
    dictionary = getdictionary(words, insert)
    duration = getduration(dictionary, duration)
    config_lexicon = (ph_duration = duration, dictionary = dictionary)
    return generate_lexicon(config_lexicon)
end

"""
    getduration(dictionary::Dict{Symbol,Vector{Symbol}}, duration::Real)
    getduration(dictionary::Dict{Symbol,Vector{Symbol}}, duration::NamedTuple)

Return a `Dict{Symbol,Float32}` with the duration (ms) of every phoneme of `dictionary` and of
the silence symbol `:_`.

With a number, every phoneme (and silence) gets `Float32(duration)`. With a `NamedTuple`, each
phoneme `ph` gets `duration[ph]` and the silence gets `duration.silence`; an error is thrown if a
phoneme is missing.
"""
function getduration(dictionary::Dict{Symbol,Vector{Symbol}}, duration::R) where {R<:Real}
    phonemes = getphonemes(dictionary)
    Dict(Symbol(phoneme) => Float32(duration) for phoneme in phonemes)
end

function getduration(dictionary::Dict{Symbol,Vector{Symbol}}, duration::NamedTuple)
    dict = Dict{Symbol,Float32}()
    phonemes = getphonemes(dictionary)
    for phoneme in phonemes
        if haskey(duration, Symbol(phoneme))
            push!(dict, Symbol(phoneme) => getfield(duration, Symbol(phoneme)))
        elseif phoneme == :_
            push!(dict, Symbol(phoneme) => getfield(duration, :silence))
        else
             throw(ErrorException("Duration for phoneme $phoneme not found in duration NamedTuple"))
        end
    end
    dict
end


"""
    getneurons(stim, symbol, target = nothing)

Return the unique indices of the neurons targeted by the stimulus `stim[Symbol(symbol, "_", target)]`
(or `stim[symbol]` when `target` is `nothing` or `:s`).
"""
function getneurons(stim, symbol, target = nothing)
    target = (target == :s) || isnothing(target) ? "" : "_$target"
    target = Symbol(string(symbol, target))
    return collect(Set(getfield(stim, target).neurons))
end

"""
    getstim(stim, word, target)

Return the field [`getstimsym`](@ref)`(word, target)` of the stimulus collection `stim`.
"""
function getstim(stim, word, target)
    return getfield(stim, getstimsym(word, target))
end

"""
    getstimsym(word, target)

Return the stimulus name `Symbol(word, "_", target)`, or `Symbol(word)` when `target` is
`nothing` or `:s` (soma).
"""
function getstimsym(word, target)
    target = (target == :s) || isnothing(target) ? "" : "_$target"
    return Symbol(string(word)*target)
end


"""
    stimuli_names(lexicon)

Return the stimulus names used by [`step_input`](@ref), [`set_stimuli!`](@ref) and
[`update_stimuli!`](@ref): words prefixed with `w_`, phonemes prefixed with `p_`.

# Returns
`(words, phonemes, all)`, where `all = vcat(words, phonemes)`.

# Example
```julia
using SpikingNeuralNetworks, SNNUtils
SNN.@load_units
stimuli_names(get_lexicon([:AB], 50ms))  # (words = [:w_AB], phonemes = [:p_A, :p_B], all = ...)
```
"""
function stimuli_names(lexicon)
    words = map(lexicon.symbols.words) do word
         Symbol(string("w_", word))
    end
    phonemes = map(lexicon.symbols.phonemes) do phoneme
         Symbol(string("p_", phoneme))
    end
    return (;words, phonemes, all=vcat(words, phonemes))
end

"""
    symbol_names(seq)

Return the phoneme and word symbols of a lexicon or sequence, without prefixes, as
`(phonemes, words, all)` with `all = vcat(words, phonemes)`. (Use [`stimuli_names`](@ref) for the
`w_`/`p_`-prefixed stimulus names.)
"""
function symbol_names(seq)
    phonemes = Symbol[]
    words = Symbol[]
    [push!(phonemes, ph) for ph in seq.symbols.phonemes]
    [push!(words, w) for w in seq.symbols.words]
    return (phonemes = phonemes, words = words, all = vcat(words, phonemes))
end


export getstim, getstimsym

export generate_sequence,
    sign_intervals,
    time_in_interval,
    sequence_end,
    generate_lexicon,
    start_interval,
    getdictionary,
    getduration,
    getphonemes,
    symbol_names,
    get_lexicon,
    getneurons,
    all_intervals,
    stimuli_names,
    generate_balanced_sequence,
    merge_intervals


"""
    generate_balanced_sequence(sounds, sequence_length)

Return a shuffled vector of length `sequence_length` in which every element of `sounds` appears
`sequence_length ÷ length(sounds)` times, plus the first `sequence_length % length(sounds)`
sounds once more.
"""
function generate_balanced_sequence(sounds, sequence_length)
    num_sounds = length(sounds)
    target_count = sequence_length ÷ num_sounds
    remainder = sequence_length % num_sounds

    # Create a list with the target count of each sound
    sequence = repeat(sounds, inner = target_count)

    # Add the remainder sounds to balance the sequence
    sequence = vcat(sequence, sounds[1:remainder])

    # Shuffle the sequence to randomize the order
    shuffled_sequence = shuffle(sequence)

    return shuffled_sequence
end

