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
    proj_strength::Real, # strength of the synaptic projection
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
- `proj_strength::Real`: Strength of the synaptic projection
- `kwargs...`: Additional keyword arguments

# Returns
- A NamedTuple containing PoissonStimulus objects organized by symbol

This function creates Poisson stimulus inputs for both words and phonemes in the provided lexicon,
connecting them to the specified target population in the network. Each stimulus is configured
with the specified parameters and rate function.
"""
function step_input(;
    inputs, # optionally provide a sequence
    network::NamedTuple, # network object

    ## Projection parameters
    sym::Symbol = :glu,
    targets::Union{Vector{Symbol},Vector{Nothing}} = [nothing],  # target neuron's compartments
    pop::Symbol = :Exc,  # target population
    p_post::Real,  # probability of post_synaptic projection
    peak_rate::Real, # peak rate of the stimulus
    proj_strength::Real, # strength of the synaptic projection
    kwargs...,
)
    @info "Creating step input stimulus for symbols: $(inputs)"
    @info "Projection parameters: sym=$(sym), targets=$(targets), pop=$(pop), p_post=$(p_post), peak_rate=$(peak_rate), proj_strength=$(proj_strength)"

    target_pop = getfield(network.pop, pop)
    stim = Dict{Symbol,Any}()
    for s in inputs
        param = PoissonInterval(rate = peak_rate, μ = proj_strength)
        my_input = StimulusGroup(
            param, 
            target_pop,
            sym,
            targets;
            p_post,
            neurons = :p_post,
            name = "$(s)",
        )
        push!(stim, Symbol(s) => my_input)
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
function set_stimuli!(; model, seq, words = true, phonemes = true)
    @unpack stim = model
    for w in seq.symbols.words
        w = Symbol(string("w_", w))
        haskey(stim, w) || continue
        SNNModels.set_active!(stim[w], words)
    end
    for p in seq.symbols.phonemes
        ph = Symbol(string("p_", p))
        haskey(stim, ph) || continue
        SNNModels.set_active!(stim[ph], phonemes)
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
function update_stimuli!(; seq, model)
    for w in seq.symbols.words
        s = Symbol(string("w_", w))
        ints = copy(sign_intervals(w, seq))
        haskey(model.stim, s) || continue
        set_intervals!(model.stim[s], ints)
    end
    for p in seq.symbols.phonemes
        s = Symbol(string("p_", p))
        ints = copy(sign_intervals(p, seq))
        haskey(model.stim, s) || continue
        set_intervals!(model.stim[s], ints)
    end
    return model
end

export step_input, set_stimuli!, update_stimuli!