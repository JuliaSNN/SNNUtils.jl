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
    mode = :fixed,
    seed = nothing,
    silent_intervals = 1,
    presentations,
    kwargs...,
)
    @assert mode in [:fixed, :random, :balanced] "Mode must be one of :fixed, :random, or :balanced"
    @info "Generating word sequence with mode: $mode, presentations: $presentations"

    @unpack dict, symbols, silence, ph_duration = lexicon
    if seed !== nothing
        Random.seed!(seed)
    end

    lexicon_words = collect(keys(dict))

    word_count = Dict(word => 0 for word in lexicon_words)
    weight_list = nothing
    @show mode
    if  mode == :balanced
        weight_list = map(lexicon_words) do word
                        exp(-1/word_count[word])
                    end
    elseif mode == :random
        weight_list = map(lexicon_words) do word
                        haskey(weights, word) ? weights[word] : 0
                    end
    elseif mode == :fixed
        total_weight = sum(values(weights))
        word_list = []
        for (word, weight) in pairs(weights)
            count = floor(Int, weight * presentations / total_weight)
            append!(word_list, fill(word, count))
        end
        shuffle!(word_list)
    end

    words, phonemes = [], []
    while sum(values(word_count)) < presentations
        if mode == :fixed
            current_word = pop!(word_list)
        elseif mode == :balanced
            weight_list =[exp(-word_count[word]) for word in lexicon_words]
            current_word = StatsBase.sample(lexicon_words, StatsBase.Weights(weight_list))
        else
            current_word = StatsBase.sample(lexicon_words, StatsBase.Weights(weight_list))
        end
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
    end
    push!(words, silence)
    push!(phonemes, silence)
    seq_length = length(words)
    @show word_count

    return words, phonemes, seq_length
end


export word_phonemes_sequence
