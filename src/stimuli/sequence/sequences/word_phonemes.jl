"""
    word_phonemes_sequence(; lexicon, weights = nothing, mode = :fixed, seed = nothing,
                           silent_intervals = 1, presentations, kwargs...)

Sequence generator for [`generate_sequence`](@ref): draw `presentations` words from
`lexicon.dict` and expand each into its phonemes, followed by `silent_intervals` silence elements.

Modes:
- `:fixed`: every word `w` in `weights` (a `Dict` word => weight) is presented
  `floor(Int, weights[w] * presentations / sum(values(weights)))` times, in random order. If the
  rounded counts sum to less than `presentations`, the function errors when the list is exhausted
  (`pop!` on an empty vector).
- `:random`: words are sampled independently with probabilities proportional to `weights[w]`
  (0 for words not in `weights`).
- `:balanced`: words are sampled with weights `exp(-count(w))`, favouring words presented less
  often; `weights` is ignored.

If `seed !== nothing`, `Random.seed!(seed)` is called. One final silence element is appended.
The function prints `mode` and the final word counts (`@show`).

# Returns
`(words, phonemes, seq_length)`: the word of each element, the phoneme of each element, and
their common length.

# Example
```julia
using SpikingNeuralNetworks, SNNUtils
SNN.@load_units
lexicon = get_lexicon([:AB, :BA], 50ms)
words, phonemes, L = word_phonemes_sequence(; lexicon, presentations = 4, mode = :balanced)
```
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
