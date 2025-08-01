

"""
    step_input_sequence(; network, targets=[:d], lexicon, config_sequence, seed=1234)

Generate a sequence input for a spiking neural network.

# Arguments
- `network`: The spiking neural network object.
- `targets`: An array of target neurons to stimulate. Default is `[:d]`.
- `lexicon`: The lexicon object containing the sequence symbols.
- `config_sequence`: The configuration for generating the sequence.

# Returns
- `stim`: A named tuple of stimuli for each symbol in the sequence.
- `seq`: The generated sequence.

"""
function step_input(;
    lexicon, # optionally provide a sequence    
    network::NamedTuple, # network object
    # words::Bool,  # active or inactive word inputs
    # phonemes::Bool=true, # active or inactive phoneme inputs

    ## Projection parameters
    targets::Union{Vector{Symbol},Vector{Nothing}},  # target neuron's compartments
    p_post::Real,  # probability of post_synaptic projection
    peak_rate::Real, # peak rate of the stimulus
    start_rate::Real, # start rate of the stimulus
    decay_rate::Real, # decay rate of attack-peak function
    proj_strength::Real, # strength of the synaptic projection
    kwargs...,
)

    @unpack E = network.pop
    # seq = isnothing(seq) ? 

    stim = Dict{Symbol,Any}()
    variables = Dict(
        :decay=>decay_rate,
        :peak=>peak_rate,
        :start=>start_rate,
        :intervals=>Float32[],
    )

    for s in lexicon.symbols.words
        param = PSParam(rate = attack_decay, variables = copy(variables))
        _my_targets = Dict{Symbol,Any}()
        for t in targets
            key = isnothing(t) ? :v : t
            my_input = PoissonStimulus(
                E,
                :he,
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
        param = PSParam(rate = attack_decay, variables = copy(variables))
        push!(stim, s => Dict{Symbol,Any}())
        _my_targets = Dict{Symbol,Any}()
        for t in targets
            key = isnothing(t) ? :v : t
            my_input = PoissonStimulus(
                E,
                :he,
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


function dummy_input(x, param::PSParam)
    return 0kHz
end


"""
    attack_decay(x, param::PSParam)

    Generate an attack-decay function for the PoissonStimulus. 
    It requires these parameters in the PoissonStimulusParameter object:
    - `intervals`: The intervals for the attack-decay function.
    - `decay`: The decay rate for the function.
    - `peak`: The peak rate for the function.
    - `start`: The start rate for the function.
    
    The attack decay function is defined as:

    f(x) = peak + (start-peak) *(exp(-(x-my_interval)/decay))

"""
function attack_decay(x, param::PSParam)
    intervals::Vector{Vector{Float32}} = param.variables[:intervals]
    decay::Float32 = param.variables[:decay]
    peak::Float32 = param.variables[:peak]
    start::Float32 = param.variables[:start]
    if time_in_interval(x, intervals)
        my_interval::Float32 = start_interval(x, intervals)
        return peak + (start-peak)*(exp(-(x-my_interval)/decay))
        # return 0kHz
    else
        return 0kHz
    end
end
# scatter(new_seq.sequence[1,:], seq.sequence[1,:], label="New sequence", c=:black, alpha=0.01, ms=10)


export step_input_sequence,
    randomize_sequence!,
    dummy_input,
    attack_decay,
    update_stimuli!,
    set_stimuli!,
    sign_intervals,
    all_intervals
