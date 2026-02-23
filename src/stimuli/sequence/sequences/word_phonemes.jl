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
function word_phonemes_sequence(;
    lexicon,
    weights = nothing,
    seed = nothing,
    silent_intervals = 1,
    presentations,
    kwargs...,
)

    @unpack dict, symbols, silence, ph_duration = lexicon
    if seed !== nothing
        Random.seed!(seed)
    end

    lexicon_words = collect(keys(dict))

    word_count = Dict(word => 1 for word in lexicon_words)
    weight_list = nothing
    balanced = nothing
    if isnothing(weights) 
        balanced = true
        weight_list = map(lexicon_words) do word
                        exp(-1/word_count[word])
                    end
    else
        balanced = false
        weight_list = map(lexicon_words) do word
                        haskey(weights, word) ? weights[word] : 0
                    end
    end

    words, phonemes = [], []
    while sum(values(word_count)) < presentations
        current_word = StatsBase.sample(lexicon_words, StatsBase.Weights(weight_list))
        word_phonemes = dict[current_word]
        word_count[current_word] += 1

        for ph in word_phonemes
            push!(phonemes, ph)
            push!(words, current_word)
        end

        for _ = 1:silent_intervals
            push!(words, silence)
            push!(phonemes, silence)
        end

        push!(words, silence)
        push!(phonemes, silence)

        if balanced 
            weight_list =[1/word_count[word] for word in lexicon_words]
        end
    end
    seq_length = length(words)

    return words, phonemes, seq_length
end


export word_phonemes_sequence
