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
# Sequence input
function step_input_sequence(;network, targets=[:d], lexicon, 
    config_sequence, p_post, 
    peak_rate=4kHz, start_rate=2kHz, decay_rate=10ms,
    kwargs...)

    @unpack E = network.pop
    seq = generate_sequence(lexicon, word_phonemes_sequence; config_sequence..., kwargs...)


    function step_input(x, param::PSParam) 
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

    stim = Dict{Symbol,Any}()
    parameters = Dict(:decay=>decay_rate, :peak=>peak_rate, :start=>start_rate)
    for s in seq.symbols.words
        variables = merge(parameters, Dict(:intervals=>sign_intervals(s, seq)))
        param = PSParam(rate=step_input, 
                    variables=variables)
        for t in targets
            push!(stim, Symbol(string(s,"_",t))  => SNN.PoissonStimulus(E, :he, t, μ=4.f0, param=param, name="w_$s", p_post=p_post))
        end
    end
    for s in seq.symbols.phonemes
        variables = merge(parameters, Dict(:intervals=>sign_intervals(s, seq)))
        param = PSParam(rate=step_input, 
                    variables=variables)
        for t in targets
            push!(stim,Symbol(string(s,"_",t))  => SNN.PoissonStimulus(E, :he, t, μ=4.f0, param=param, name="$s", p_post=p_post) 
            )
        end
    end
    stim = dict2ntuple(stim)
    stim, seq
end

export step_input_sequence