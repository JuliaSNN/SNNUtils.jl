"""
    step_input(;
    lexicon, # optionally provide a sequence    
    network::NamedTuple, # network object

    ## Projection parameters
    sym::Symbol = :glu,
    targets::Union{Vector{Symbol},Vector{Nothing}},  # target neuron's compartments
    pop::Symbol = :E,  # target population
    p_post::Real,  # probability of post_synaptic projection
    peak_rate::Real, # peak rate of the stimulus
    start_rate::Real, # start rate of the stimulus
    decay_rate::Real, # decay rate of attack-peak function
    proj_strength::Real, # strength of the synaptic projection
    rate_function = attack_decay, # function to generate the rate
    kwargs...,
)

Create a step input stimulus for a neural network simulation.

# Arguments
- `lexicon`: A lexicon object containing symbols for words and phonemes
- `network::NamedTuple`: The neural network object containing populations
- `sym::Symbol = :glu`: The synaptic type (default: :glu)
- `targets::Union{Vector{Symbol},Vector{Nothing}}`: Target compartments for projections
- `pop::Symbol = :E`: Target population symbol (default: :E)
- `p_post::Real`: Probability of post-synaptic projection
- `peak_rate::Real`: Peak rate of the stimulus
- `start_rate::Real`: Start rate of the stimulus
- `decay_rate::Real`: Decay rate for the attack-peak function
- `proj_strength::Real`: Strength of the synaptic projection
- `rate_function`: Function to generate the rate (default: attack_decay)
- `kwargs...`: Additional keyword arguments

# Returns
- A NamedTuple containing PoissonStimulus objects organized by symbol

This function creates Poisson stimulus inputs for both words and phonemes in the provided lexicon,
connecting them to the specified target population in the network. Each stimulus is configured
with the specified parameters and rate function.
"""
function step_input(;
    lexicon, # optionally provide a sequence
    network::NamedTuple, # network object

    ## Projection parameters
    sym::Symbol = :glu,
    targets::Union{Vector{Symbol},Vector{Nothing}},  # target neuron's compartments
    pop::Symbol = :E,  # target population
    p_post::Real,  # probability of post_synaptic projection
    peak_rate::Real, # peak rate of the stimulus
    start_rate::Real, # start rate of the stimulus
    decay_rate::Real, # decay rate of attack-peak function
    proj_strength::Real, # strength of the synaptic projection
    rate_function = attack_decay, # function to generate the rate
    kwargs...,
    )

    target_pop = getfield(network.pop, pop)
    stim = Dict{Symbol,Any}()
    variables = Dict(
        :decay=>decay_rate,
        :peak=>peak_rate,
        :start=>start_rate,
        :intervals=>Float32[],
    )

    for s in lexicon.symbols.words
        param = PSParam(rate = rate_function, variables = copy(variables))
        _my_targets = Dict{Symbol,Any}()
        for t in targets
            key = isnothing(t) ? :v : t
            my_input = PoissonStimulus(
                target_pop,
                sym,
                t,
                μ = proj_strength,
                param = param,
                name = "w_$s",
                p_post = p_post,
            )
            push!(_my_targets, key => my_input)
        end
        if length(_my_targets) > 1
            push!(stim, s => _my_targets |> dict2ntuple)
        else
            push!(stim, s => first(_my_targets)[2])
        end
    end
    for s in lexicon.symbols.phonemes
        param = PSParam(rate = rate_function, variables = copy(variables))
        push!(stim, s => Dict{Symbol,Any}())
        _my_targets = Dict{Symbol,Any}()
        for t in targets
            key = isnothing(t) ? :v : t
            my_input = PoissonStimulus(
                target_pop,
                sym,
                t,
                μ = proj_strength,
                param = param,
                name = "p_$s",
                p_post = p_post,
            )
            push!(_my_targets, key => my_input)
        end

        if length(_my_targets) > 1
            push!(stim, s => _my_targets |> dict2ntuple)
        else
            push!(stim, s => first(_my_targets)[2])
        end
    end
    return (stim |> dict2ntuple)
end

"""
    set_stimuli!(; model, targets::Vector{Symbol}, seq, words = true, phonemes = true)

Activate or deactivate stimuli for words and phonemes.

# Arguments
- `model`: The neural network model containing stimuli to be configured
- `targets::Vector{Symbol}`: Vector of target compartments for stimuli
- `seq`: Sequence object containing word and phoneme symbols
- `words::Bool = true`: Whether to activate stimuli for words (default: true)
- `phonemes::Bool = true`: Whether to activate stimuli for phonemes (default: true)

# Returns
- The modified model with updated stimulus activation states

This function sets the activation state of stimuli in the model for both words and phonemes.
For each target compartment, it activates or deactivates stimuli based on the `words` and
`phonemes` boolean flags. The stimuli are identified by combining the symbol with the target
compartment name.
"""
function set_stimuli!(; model, targets::Vector{Symbol}, seq, words = true, phonemes = true)
    @unpack stim = model
    for target in targets
        for s in seq.symbols.words
            word = Symbol(string(s, "_", target))
            stim[word].param.active[1] = words
        end
        for s in seq.symbols.phonemes
            ph = Symbol(string(s, "_", target))
            stim[ph].param.active[1] = phonemes
        end
    end
end

"""
    update_stimuli!(; seq, model, targets::Vector{Symbol})

Update the stimulus intervals for words and phonemes.

# Arguments
- `seq`: Sequence object containing word and phoneme symbols with timing information
- `model`: The neural network model containing stimuli to be updated
- `targets::Vector{Symbol}`: Vector of target compartments for stimuli

# Returns
- The modified model with updated stimulus intervals

This function updates the timing intervals for stimuli in the model for both words and phonemes.
For each target compartment, it retrieves the timing intervals from the sequence object and
updates the corresponding stimulus parameters. The stimuli are identified by combining the
symbol with the target compartment name.
"""
function update_stimuli!(; seq, model, targets::Vector{Symbol})
    for target in targets
        for w in seq.symbols.words
            s = Symbol(string(w, "_", target))
            ints = copy(sign_intervals(w, seq))
            model.stim[s].param.variables[:intervals] = ints
        end
        for p in seq.symbols.phonemes
            s = Symbol(string(p, "_", target))
            ints = copy(sign_intervals(p, seq))
            model.stim[s].param.variables[:intervals] = ints
        end
    end
    return model
end
