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

function generate_lexicon_vot(config)
    @unpack ph_duration, dictionary = config

    all_words = collect(keys(dictionary)) |> Set |> collect |> sort |> Vector{Symbol}
    all_phonemes = collect(values(dictionary)) |> Iterators.flatten |> Set |> collect |> sort |> Vector{Symbol}
    for word in all_words
        push!(all_phonemes, Symbol("#$word"))
    end
    ## Add the silence symbolstarts
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
function generate_sequence(seq_function::Function; lexicon::NamedTuple, seed=-1, kwargs...)
    (seed > 0) && (Random.seed!(seed))

    words, phonemes, seq_length = seq_function(;
                        lexicon=lexicon,
                        kwargs...
                    )

    @unpack dict, symbols, silence, ph_duration = lexicon
    ## create the populations
    ## sequence from the initial word sequence
    sequence = Matrix{Any}(fill(silence, 6, seq_length+1))
    sequence[1, 1] = silence
    sequence[2, 1] = silence
    sequence[3, 1] = ph_duration[silence]
    for (n, (w, p)) in enumerate(zip(words, phonemes))
        sequence[1, 1+n] = w
        sequence[2, 1+n] = p
        sequence[3, 1+n] = ph_duration[p]
    end

    sequence[4, :] .= :mid
    for n in 1:(size(sequence, 2)-2)
        if !(sequence[1, n+1] == sequence[1, n])
            sequence[4, n+1] = :onset
        end
        if !(sequence[1, n+1] == sequence[1, n+2])
            sequence[4, n+1] = :offset
            j = 2
            while (n+1-j > 0) && !(sequence[4, n+1-j] == :onset)
                sequence[4, n+1-j] = Symbol("offset$j")
                j += 1
            end
        end
        (sequence[1, n] == :_) && (sequence[4, n] = :silence)
    end

    sequence[5,:] .= [0ms,cumsum(sequence[3,2:end])...]
    sequence[6,:] .= [cumsum(sequence[3,1:end])...]

    line_id = (words=1, phonemes=2, duration=3, type=4, onset=5, offset=6)
    sequence = (;lexicon...,
                sequence=sequence,
                line_id = line_id)

end

function generate_sequence_vot(seq_function::Function; lexicon::NamedTuple, init_silence::Real, end_silence::Real, seed=-1, init_time = 0ms, kwargs...)
    (seed > 0) && (Random.seed!(seed))

    words, phonemes, seq_length = seq_function(;
                        lexicon=lexicon,
                        kwargs...
                    )

    @unpack dict, symbols, silence, ph_duration = lexicon
    ## create the populations
    ## sequence from the initial word sequence
    sequence = Matrix{Any}(fill(silence, 6, seq_length+2))
    sequence[1, 1] = silence
    sequence[2, 1] = silence
    sequence[3, 1] = init_silence
    for (n, (w, p)) in enumerate(zip(words, phonemes))
        if startswith(String(p), "#")
            sequence[1, 1+n] = w # silence
            sequence[2, 1+n] = p
            min, max = ph_duration[p]
            vot_duration = rand(range(min, max; step=10))
            sequence[3, 1+n] = Float32(vot_duration)
        # elseif sequence[2, n]  == silence
        #     sequence[1, 1+n] = silence
        #     sequence[2, 1+n] = p
        #     sequence[3, 1+n] = ph_duration[p]
        else
            sequence[1, 1+n] = w
            sequence[2, 1+n] = p
            sequence[3, 1+n] = ph_duration[p]
        end
    end

    sequence[1, end] = silence
    sequence[2, end] = silence
    sequence[3, end] = end_silence

    sequence[4, :] .= :mid
    for n in 1:(size(sequence, 2)-2)
        if !(sequence[1, n+1] == sequence[1, n])
            sequence[4, n+1] = :onset
        end
        if !(sequence[1, n+1] == sequence[1, n+2])
            sequence[4, n+1] = :offset
            j = 2
            while (n+1-j > 0) && !(sequence[4, n+1-j] == :onset)
                sequence[4, n+1-j] = Symbol("offset$j")
                j += 1
            end
        end
        (sequence[1, n] == :_) && (sequence[4, n] = :silence)
    end

    sequence[5,:] .= [0ms, cumsum(sequence[3,2:end])...] .+ init_time
    sequence[6,:] .= [cumsum(sequence[3,1:end])...] .+ init_time

    line_id = (words=1, phonemes=2, duration=3, type=4, onset=5, offset=6)
    sequence = (;lexicon...,
                sequence=sequence,
                line_id = line_id)

end



function generate_serial_sequence(repetition::Int, lexicon::NamedTuple, init_silence::Real, end_silence::Real, init_time = 0ms, seed=-1, kwargs...)
    (seed > 0) && (Random.seed!(seed))

    @unpack dict, symbols, silence, ph_duration = lexicon

    # Number of columns for one pattern: 7 (as in your example)
    ncols = 7 * repetition + 2  # +2 for initial and final silence
    sequence = Matrix{Any}(fill(silence, 6, ncols))
    col = 1

    # Initial silence
    sequence[1, col] = silence
    sequence[2, col] = silence
    sequence[3, col] = init_silence
    col += 1

    for _ in 1:repetition
        # :_  :B   20.0
        sequence[1, col] = silence
        sequence[2, col] = :B
        sequence[3, col] = 20.0
        col += 1

        # :_  :_   20.0
        sequence[1, col] = silence
        sequence[2, col] = silence
        sequence[3, col] = 20.0
        col += 1

        # :_  :V   20.0
        sequence[1, col] = silence
        sequence[2, col] = :V
        sequence[3, col] = 20.0
        col += 1

        # :_  :_   20.0
        sequence[1, col] = silence
        sequence[2, col] = silence
        sequence[3, col] = 20.0
        col += 1

        # :COINC  :_   20.0
        sequence[1, col] = :COINC
        sequence[2, col] = silence
        sequence[3, col] = 20.0
        col += 1

        # :_  :_   20.0
        sequence[1, col] = silence
        sequence[2, col] = silence
        sequence[3, col] = 20.0
        col += 1

        # :GAP  :_   20.0
        sequence[1, col] = :GAP
        sequence[2, col] = silence
        sequence[3, col] = 20.0
        col += 1
    end

    # Final silence
    sequence[1, col] = silence
    sequence[2, col] = silence
    sequence[3, col] = end_silence

    # Fill the rest of the lines/types as needed (optional)
    sequence[4, :] .= :mid
    sequence[5,:] .= [0ms, cumsum(sequence[3,2:end])...] .+ init_time
    sequence[6,:] .= [cumsum(sequence[3,1:end])...] .+ init_time

    line_id = (words=1, phonemes=2, duration=3, type=4, onset=5, offset=6)
    return (;lexicon..., sequence=sequence, line_id=line_id)
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
function sign_intervals(sign::Symbol, sequence; all_simulation::Bool=false)
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
        if all_simulation
            interval[1] +=  sequence[line_id.onset, 1]
            interval[2] +=  sequence[line_id.onset, 1]
        end
        push!(intervals, interval)
    end
    return intervals 
end
# function sign_intervals(sign::Symbol, sequence; all_simulation::Bool=false)
#     @unpack dict, sequence, symbols, line_id = sequence
#     ## Identify the line of the sequence that contains the sign (either words or phonemes)
#     sign_line_id = -1
#     for k in keys(symbols)
#         if sign in getfield(symbols,k)
#             sign_line_id = getfield(line_id,k)
#             break
#         end
#     end
#     if sign_line_id == -1
#         throw(ErrorException("Sign index not found"))
#     end

#     ## Find the intervals where the sign is present
#     intervals = Vector{Vector{Float32}}()
#     my_seq = sequence[sign_line_id, :]
#     if all_simulation
#         time_counter = sequence[5, 1]
#     else
#         time_counter = 0
#     end
    
#     for i in eachindex(my_seq)
#         interval_start = time_counter
#         if my_seq[i] == sign
#             interval_end = time_counter + sequence[line_id.duration, i]
#         else
#             interval_end = time_counter
#         end
#         if  interval_end > interval_start
#             interval = [interval_start, interval_end]
#             push!(intervals, interval)
#         end
#         time_counter += sequence[line_id.duration, i]
#     end
#     # while !isnothing(_end)  || !isnothing(_start)
#     #     _start = findfirst(x -> x == sign, my_seq[_end:end])
#     #     if isnothing(_start)
#     #         break
#     #     else
#     #         _start += _end-1
#     #     end
#     #     _end  = findfirst(x -> x != sign, my_seq[_start:end]) + _start - 1
#     #     interval[1] = cum_duration[_start] - sequence[line_id.duration,_start]
#     #     interval[2] = cum_duration[_end-1]
#     # end
#     return intervals
# end


function all_intervals(sym::Symbol, sequence; interval::Vector=[-50ms, 100ms] )
    offsets = Vector{Vector{Float32}}()
    ys = Vector{Symbol}()
    symbols = getfield(sequence.symbols, sym)
    @show symbols
    for word in symbols
        for myinterval in sign_intervals(word, sequence; all_simulation=true)
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

function getdictionary_vot(words::Vector{T}, phonemes_list::Vector{Vector{String}}) where T <: Union{String, Symbol}
    dictionary = Dict{Symbol,Any}()
    for (word, phonemes) in zip(words, phonemes_list)
        push!(dictionary, Symbol(word) => [Symbol(phoneme) for phoneme in phonemes])
    end
    return dictionary
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

function getphonemes_vot(dictionary::Dict{Symbol, Any})
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

function getduration_vot(dictionary::Dict{Symbol, Any}, duration::R, vot_durations) where R <: Real
    phonemes = getphonemes_vot(dictionary)
    durations = Dict{Symbol,Any}()
    for phoneme in phonemes
        if phoneme == Symbol("_")
            push!(durations, Symbol(phoneme) => Float32(200ms))
        else
            push!(durations, Symbol(phoneme) => Float32(duration))
        end
    end

    for (word, range) in zip(keys(dictionary), vot_durations)
        vot_symbol = Symbol("#" * string(word))
        push!(durations, vot_symbol => range)
    end
    return durations
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

function getneurons(stim, symbol, target=nothing)
    target = (target ==:s) || isnothing(target) ? "" : "_$target" 
    target = Symbol(string(symbol, target ))
    @show target
   return collect(Set(getfield(stim,target).neurons))
end

function getstim(stim, word, target)
    return getfield(stim, getstimsym(word, target))
end

function getstimsym(word, target)
    target = (target ==:s) || isnothing(target) ? "" : "_$target" 
    return Symbol(string(word)*target)
end

export getstim, getstimsym

export generate_sequence, generate_sequence_vot, generate_serial_sequence, sign_intervals, time_in_interval, sequence_end, generate_lexicon, generate_lexicon_vot, start_interval, getdictionary, getdictionary_vot, getduration, getduration_vot, getphonemes, getphonemes_vot, symbolnames, getneurons, all_intervals