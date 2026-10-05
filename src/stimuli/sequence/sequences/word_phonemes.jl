"""
    word_phonemes_sequence(; lexicon, weights = nothing, mode = :fixed, seed = nothing,
                           silent_intervals = 1, presentations, kwargs...)

Sequence generator for [`generate_sequence`](@ref): draw `presentations` words from
`lexicon.dict` and expand each into its phonemes, followed by `silent_intervals` silence elements.

Modes:
- `:fixed`: every word `w` in `weights` (a `Dict` word => weight) is presented
  `floor(Int, weights[w] * presentations / sum(values(weights)))` times, plus one extra
  presentation for the words with the largest remainders until the total is `presentations`,
  in random order.
- `:random`: words are sampled independently with probabilities proportional to `weights[w]`
  (0 for words not in `weights`).
- `:balanced`: words are sampled with weights `exp(-count(w))`, favouring words presented less
  often; `weights` is ignored.

`weights = nothing` means equal weights for all the words of the lexicon. If `seed !== nothing`,
`Random.seed!(seed)` is called. One final silence element is appended. The word counts are
logged at debug level.

(Up to SNNUtils 0.2.9 `:fixed` failed with `pop!` on an empty list when the rounded counts summed
to less than `presentations`, `weights = nothing` failed for `:fixed`/`:random`, and the function
printed with `@show`.)

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
    weights = isnothing(weights) ? Dict(word => 1.0 for word in lexicon_words) : weights

    word_count = Dict(word => 0 for word in lexicon_words)
    weight_list = nothing
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
        wkeys = collect(keys(weights))
        exact = [weights[w] * presentations / total_weight for w in wkeys]
        counts = floor.(Int, exact)
        # distribute the remaining presentations to the largest remainders
        for k in sortperm(exact .- counts, rev = true)[1:(presentations-sum(counts))]
            counts[k] += 1
        end
        for (word, count) in zip(wkeys, counts)
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
    @debug "word counts" word_count

    return words, phonemes, seq_length
end


export word_phonemes_sequence
