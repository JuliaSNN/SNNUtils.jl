using StatsBase

"""
    generate_lexicon(config)

Generate a lexicon based on the given configuration.

# Arguments
- `config`: A dictionary containing the configuration parameters.
    - `ph_duration`: The duration of each phoneme.
    - `dictionary`: A dictionary mapping words to phonemes.

# Returns
A named tuple containing the following fields:
- `dict`: The input dictionary.
- `symbols`: A tuple containing the phonemes and words.
- `ph_duration`: The duration of each phoneme.
- `silence_symbol`: The symbol representing silence.

"""
function generate_lexicon(config)
    @unpack ph_duration, dictionary = config

    all_words = collect(keys(dictionary)) |> Set |> collect |> sort |> Vector{Symbol}
    all_phonemes = collect(values(dictionary)) |> Iterators.flatten |> Set |> collect |> sort |> Vector{Symbol}
    symbols = collect(union(all_words,all_phonemes))

    ## Add the silence symbol
    silence_symbol = :_

    return (dict=dictionary, 
            symbols=(phonemes = all_phonemes, 
            words = all_words), 
            ph_duration = ph_duration, 
            silence = silence_symbol)
end

"""
    generate_sequence(lexicon, config, seed=nothing)

Generate a sequence of words and phonemes based on the provided lexicon and configuration.

# Arguments
- `lexicon`: A dictionary containing the lexicon information.
    - `dict`: A dictionary mapping words and phonemes to their corresponding IDs.
    - `symbols`: A list of symbols in the lexicon.
    - `silence_symbol`: The symbol representing silence.
    - `ph_duration`: A dictionary mapping phonemes to their corresponding durations.

# Returns
A named tuple containing the lexicon information and the generated sequence.

"""
function generate_sequence(seq_function::Function; init_silence=1s, lexicon::NamedTuple, kwargs...)

    words, phonemes, seq_length = seq_function(;
                        lexicon=lexicon,
                        kwargs...
                    )

    @unpack dict, symbols, silence, ph_duration = lexicon
    ## create the populations
    ## sequence from the initial word sequence
    sequence = Matrix{Any}(fill(silence, 3, seq_length+1))
    sequence[1, 1] = silence
    sequence[2, 1] = silence
    sequence[3, 1] = init_silence
    for (n, (w, p)) in enumerate(zip(words, phonemes))
        sequence[1, 1+n] = w
        sequence[2, 1+n] = p
        sequence[3, 1+n] = ph_duration[p]
    end

    line_id = (phonemes=2, words=1, duration=3)
    sequence = (;lexicon...,
                sequence=sequence,
                line_id = line_id)

end


"""
    sign_intervals(sign::Symbol, sequence)

Given a sign symbol and a sequence, this function identifies the line of the sequence that contains the sign and finds the intervals where the sign is present. The intervals are returned as a vector of vectors, where each inner vector represents an interval and contains two elements: the start time and the end time of the interval.

# Arguments
- `sign::Symbol`: The sign symbol to search for in the sequence.
- `sequence`: The sequence object containing the sign and other information.

# Returns
- `intervals`: A vector of vectors representing the intervals where the sign is present in the sequence.

# Example
"""
function sign_intervals(sign::Symbol, sequence)
    @unpack dict, sequence, symbols, line_id = sequence
    ## Identify the line of the sequence that contains the sign
    sign_line_id = -1
    for k in keys(symbols)
        if sign in getfield(symbols,k)
            sign_line_id = getfield(line_id,k)
            break
        end
    end
    if sign_line_id == -1
        throw(ErrorException("Sign index not found"))
    end

    ## Find the intervals where the sign is present
    intervals = Vector{Vector{Float32}}()
    cum_duration = cumsum(sequence[line_id.duration,:])
    _end = 1
    interval = [-1, -1]
    my_seq = sequence[sign_line_id, :]
    while !isnothing(_end)  || !isnothing(_start)
        _start = findfirst(x -> x == sign, my_seq[_end:end])
        if isnothing(_start)
            break
        else
            _start += _end-1
        end
        _end  = findfirst(x -> x != sign, my_seq[_start:end]) + _start - 1
        interval[1] = cum_duration[_start] - sequence[line_id.duration,_start]
        interval[2] = cum_duration[_end-1]
        push!(intervals, interval)
    end
    return intervals
end


function all_intervals(sym::Symbol, sequence; interval::Vector=[-50ms, 100ms] )
    offsets = Vector{Vector{Float32}}()
    ys = Vector{Symbol}()
    symbols = getfield(sequence.symbols, sym)
    @show symbols
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

Return the end of the sequence.

# Arguments
- `seq`: A sequence object containing `line_id` and `sequence` fields.

# Returns
- The sum of the values in the `sequence` array at the `line_id.duration` index.

"""
function sequence_end(seq)
    @unpack line_id, sequence = seq
    return sum(sequence[line_id.duration, :])
end

"""
    time_in_interval(x, intervals)

Return true if the time `x` is in any of the intervals.

# Arguments
- `x`: A Float32 value representing the time.
- `intervals`: A vector of vectors, where each inner vector represents an interval with two Float32 values.

# Returns
- `true` if `x` is in any of the intervals, `false` otherwise.

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
    start_interval(x, intervals)

Return the start of the interval that contains the time `x`.

# Arguments
- `x`: A Float32 value representing the time.
- `intervals`: A vector of vectors, where each inner vector represents an interval with two Float32 values.

# Returns
- The start of the interval that contains `x`, or -1 if `x` is not in any of the intervals.

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
    getdictionary(words::Vector{Union{String, Symbol}})

Create a dictionary mapping each word in `words` to a vector of symbols representing its letters.

# Arguments
- `words`: A vector of strings or symbols representing the words.

# Returns
A dictionary mapping each word to a vector of symbols representing its letters.
"""
function getdictionary(words::Vector{T }) where T <: Union{String, Symbol}
    Dict(Symbol(word) => [Symbol(letter) for letter in string(word)] for word in words)
end

"""
    getphonemes(dictionary::Dict{Symbol, Vector{Symbol}})

Get a vector of symbols representing all the unique phonemes in the given `dictionary`.

# Arguments
- `dictionary`: A dictionary mapping words to vectors of symbols representing their letters.

# Returns
A vector of symbols representing all the unique phonemes in the given `dictionary`.
"""
function getphonemes(dictionary::Dict{Symbol, Vector{Symbol}})
    phs = collect(unique(vcat(values(dictionary)...)))
    push!(phs, :_)
    return phs
end

"""
    getduration(dictionary::Dict{Symbol, Vector{Symbol}}, duration::R) where R <: Real

Create a dictionary mapping each phoneme in the given `dictionary` to the specified `duration`.

# Arguments
- `dictionary`: A dictionary mapping words to vectors of symbols representing their letters.
- `duration`: The duration to assign to each phoneme.

# Returns
A dictionary mapping each phoneme to the specified `duration`.
"""
function getduration(dictionary::Dict{Symbol, Vector{Symbol}}, duration::R) where R <: Real
    phonemes = getphonemes(dictionary)
    Dict(Symbol(phoneme) => Float32(duration) for phoneme in phonemes)
end

"""
    symbolnames(seq)

    Get the names of phonemes and words from the given sequence.
    Words are prefixed with 'w_'.

"""
function symbolnames(seq)
    phonemes = String[]
    words = String[]
    [push!(phonemes, string.(ph)) for ph in seq.symbols.phonemes]
    [push!(words, "w_"*string(w)) for w in seq.symbols.words]
    return (phonemes=phonemes, words=words)
end

function getcells(stim, symbol, target)
    target = target =="" ? "" : "_$target"
   return collect(Set(getfield(stim,Symbol(string(symbol, target ))).cells))
end

function getstim(stim, word, target)
    return getfield(stim, getstimsym(word, target))
end

function getstimsym(word, target)
    return Symbol(string(word)*"_$target")
end

export getstim, getstimsym

export generate_sequence, sign_intervals, time_in_interval, sequence_end, generate_lexicon, start_interval, getdictionary, getduration, getphonemes, symbolnames, getcells, all_intervals
