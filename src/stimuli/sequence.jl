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
- `id2string`: A mapping from numbers to symbols.
- `string2id`: A mapping from symbols to numbers.
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

    ## create mapping between numbers and phonemes
    mapping = Dict{Int, Symbol}()
    r_mapping = Dict{Symbol, Int}()
    for (n, s) in enumerate(all_words)
        push!(mapping, n => s)
        push!(r_mapping, s => n)
    end
    for (n, s) in enumerate(all_phonemes)
        n = n + length(all_words)
        push!(mapping, n => s)
        push!(r_mapping, s => n)
    end

    ## Add the silence symbol
    silence_symbol = :_
    silence = length(symbols) + 1
    push!(mapping, silence => silence_symbol)
    push!(r_mapping, silence_symbol => silence)

    return (id2string = mapping, 
            string2id = r_mapping, 
            dict=dictionary, 
            symbols=(phonemes = all_phonemes, 
            words = all_words), 
            ph_duration = ph_duration, 
            silence_symbol = silence_symbol)
end

"""
    generate_sequence(lexicon, config, seed=nothing)

Generate a sequence of words and phonemes based on the provided lexicon and configuration.

# Arguments
- `lexicon`: A dictionary containing the lexicon information.
    - `dict`: A dictionary mapping words and phonemes to their corresponding IDs.
    - `id2string`: A dictionary mapping IDs to their corresponding words and phonemes.
    - `string2id`: A dictionary mapping words and phonemes to their corresponding IDs.
    - `symbols`: A list of symbols in the lexicon.
    - `silence_symbol`: The symbol representing silence.
    - `ph_duration`: A dictionary mapping phonemes to their corresponding durations.

- `config`: A dictionary containing the configuration information.
    - `seq_length`: The length of the sequence to generate.
    - `init_silence` (optional): The initial silence symbol to use in the sequence.

- `seed` (optional): The seed value for the random number generator.

# Returns
A named tuple containing the lexicon information and the generated sequence.

"""
function generate_sequence(lexicon, config, seed=nothing)

    if seed !== nothing
        Random.seed!(seed)
    end 

    @unpack seq_length = config
    @unpack dict, id2string, string2id, symbols, silence_symbol, ph_duration = lexicon
    silent_intervals = 1
    words, phonemes = generate_random_word_sequence(
        seq_length,
        dict,
        silence_symbol,
        silent_intervals=silent_intervals;
    )

    ## create the populations
    ## sequence from the initial word sequence
    silence = string2id[silence_symbol]
    if haskey(config,:init_silence)
        sequence = fill(silence, 3, seq_length+1)
        sequence[1, 1] = silence
        sequence[2, 1] = silence
        sequence[3, 1] = config.init_silence
        for (n, (w, p)) in enumerate(zip(words, phonemes))
            sequence[1, 1+n] = string2id[w]
            sequence[2, 1+n] = string2id[p]
            sequence[3, 1+n] = ph_duration[p]
        end
    else
        sequence = fill(silence, 3, seq_length)
        for (n, (w, p)) in enumerate(zip(words, phonemes))
            sequence[1, n] = string2id[w]
            sequence[2, n] = string2id[p]
            sequence[3, n] = ph_duration[p]
        end
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
    @unpack dict, id2string, string2id, sequence, symbols, line_id = sequence
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
    cum_duration = cumsum(sequence[line_id.duration,:]) .+ time 
    _end = 1
    interval = [-1.0, -1.0]
    my_seq = sequence[sign_line_id, :]
    while !isnothing(_end)  || !isnothing(_start)
        _start = findfirst(x -> x == string2id[sign], my_seq[_end:end])
        if isnothing(_start)
            break
        else
            _start += _end-1
        end
        _end  = findfirst(x -> x != string2id[sign], my_seq[_start:end]) + _start - 1
        interval[1] = cum_duration[_start] - sequence[line_id.duration,_start]
        interval[2] = cum_duration[_end-1]
        push!(intervals, interval)
    end
    return intervals
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
    generate_random_word_sequence(sequence_length, dictionary, silence_symbol; silent_intervals=1, weights=nothing)

Generate a random word sequence of a given length using a dictionary of words and their corresponding phonemes.

# Arguments
- `sequence_length::Int`: The desired length of the word sequence.
- `dictionary::Dict{Symbol, Vector{Symbol}}`: A dictionary mapping words to their corresponding phonemes.
- `silence_symbol::Symbol`: The symbol representing silence in the word sequence.

# Optional Arguments
- `silent_intervals::Int = 1`: The number of silent intervals between words.
- `weights::Union{Nothing, Vector{Float64}} = nothing`: The weights assigned to each word in the dictionary. If `nothing`, all words have equal weight.

# Returns
- `words::Vector{Symbol}`: The generated word sequence.
- `phonemes::Vector{Symbol}`: The corresponding phonemes for each word in the sequence.
"""
function generate_random_word_sequence(
    sequence_length::Int,
    dictionary::Dict{Symbol, Vector{Symbol}},
    silence_symbol::Symbol;
    silent_intervals = 1,
    weights = nothing,
    vot_duration = nothing,
)
    dict_words = collect(keys(dictionary))

    weights = isnothing(weights) ? fill(1, length(dict_words)) : [weights[word] for word in dict_words]
    weights = StatsBase.Weights(weights)

    word_frequency = Dict{Symbol,Int}()
    words, phonemes = [], []

    remaining_words = copy(dict_words)
    make_equal = true

    while length(words) < sequence_length
        current_word = choose_word(make_equal, remaining_words, dict_words, weights, word_frequency)
        word_phonemes = dictionary[current_word]

        if haskey(word_frequency, current_word)
            word_frequency[current_word] += 1
        else
            word_frequency[current_word] = 1
        end

        if should_fill_with_silence(word_phonemes, silent_intervals, sequence_length, length(words))
            fill_with_silence!(words, phonemes, silence_symbol, sequence_length - length(words))
        else
            append_word_and_phonemes!(words, phonemes, current_word, word_phonemes, silence_symbol, silent_intervals)
        end
    end

    return words, phonemes
end

"""
    choose_word(make_equal, remaining_words, dict_words, weights, word_frequency)

Choose a word from the given list of words based on the specified criteria.

Arguments:
- `make_equal`: A boolean indicating whether the words should be chosen with equal probability.
- `remaining_words`: A list of words that have not been chosen yet.
- `dict_words`: A list of all available words.
- `weights`: A list of weights corresponding to each word in `dict_words`.
- `word_frequency`: A dictionary mapping words to their frequencies.

Returns:
- A word chosen based on the specified criteria.
"""
function choose_word(make_equal, remaining_words, dict_words, weights, word_frequency)
    if make_equal
        return !isempty(remaining_words) ? pop!(remaining_words) :
               StatsBase.sample(dict_words, StatsBase.Weights([exp(-word_frequency[word]) for word in dict_words]))
    else
        return StatsBase.sample(dict_words, weights)
    end
end

"""
    should_fill_with_silence(word_phonemes, silent_intervals, sequence_length, current_length)

Check if the remaining sequence should be filled with silence.

Arguments:
- `word_phonemes`: The number of phonemes in the current word.
- `silent_intervals`: The number of silent intervals.
- `sequence_length`: The total length of the sequence.
- `current_length`: The current length of the sequence.

Returns:
- A boolean indicating whether the remaining sequence should be filled with silence.
"""
function should_fill_with_silence(word_phonemes, silent_intervals, sequence_length, current_length)
    return length(word_phonemes) + silent_intervals > sequence_length - current_length
end

"""
    fill_with_silence!(words, phonemes, silence_symbol, fill_count)

Fill the given lists with silence symbols.

Arguments:
- `words`: A list of words.
- `phonemes`: A list of phonemes.
- `silence_symbol`: The symbol representing silence.
- `fill_count`: The number of times to fill with silence.
"""
function fill_with_silence!(words, phonemes, silence_symbol, fill_count)
    for _ in 1:fill_count
        push!(words, silence_symbol)
        push!(phonemes, silence_symbol)
    end
end

"""
    append_word_and_phonemes!(words, phonemes, word, phonemes_list, silence_symbol, silent_intervals)

Append a word and its corresponding phonemes to the given lists.

Arguments:
- `words`: A list of words.
- `phonemes`: A list of phonemes.
- `word`: The word to append.
- `phonemes_list`: A list of phonemes corresponding to the word.
- `silence_symbol`: The symbol representing silence.
- `silent_intervals`: The number of silent intervals to append after the word.
"""
function append_word_and_phonemes!(words, phonemes, word, phonemes_list, silence_symbol, silent_intervals)
    for ph in phonemes_list
        push!(phonemes, ph)
        push!(words, word)
    end

    for _ = 1:silent_intervals
        push!(words, silence_symbol)
        push!(phonemes, silence_symbol)
    end
end

# function silence_sequence!(seq::Encoding)
#     @assert(length(seq.populations) == seq.silence)
#     @assert(seq.populations[seq.silence] == [])
#     fill!(seq.sequence, seq.silence)
# end

# function silence_sequence(seq::Encoding)
#     new_seq = deepcopy(seq)
#     silence_sequence!(new_seq)
#     return new_seq
# end

export generate_sequence, sign_intervals, time_in_interval, sequence_end, generate_lexicon, start_interval, generate_random_word_sequence 