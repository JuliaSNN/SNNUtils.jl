"""
    step_input_sequence(; network, targets=[:d], lexicon, config_sequence, seed=1234)

Generate a sequence input for a spiking neural network.

# Arguments
- `network`: The spiking neural network object.
- `targets`: An array of target neurons to stimulate. Default is `[:d]`.
- `lexicon`: The lexicon object containing the sequence symbols.
- `config_sequence`: The configuration for generating the sequence.
- `seed`: The seed value for random number generation. Default is `1234`.

# Returns
- `stim`: A named tuple of stimuli for each symbol in the sequence.
- `seq`: The generated sequence.

"""
# Sequence input
function step_input_sequence(;network, targets=[:d], lexicon, config_sequence, seed=1234)
    @unpack E = network.pop
    seq = generate_sequence(lexicon, config_sequence, seed)

    function step_input(x, param::PSParam) 
        intervals::Vector{Vector{Float32}} = param.variables[:intervals]
        if time_in_interval(x, intervals)
            my_interval = start_interval(x, intervals)
            return 2kHz * 3kHz*(1-exp(-(x-my_interval)/10))
            # return 0kHz
        else
            return 0kHz
        end
    end

    stim = Dict{Symbol,Any}()
    for s in seq.symbols.words
        param = PSParam(rate=step_input, variables=Dict(:intervals=>sign_intervals(s, seq)))
        for t in targets
            push!(stim,s  => SNN.PoissonStimulus(E, :he, t, μ=4.f0, param=param, name="w_$s"))
        end
    end
    for s in seq.symbols.phonemes
        param = PSParam(rate=step_input, variables=Dict(:intervals=>sign_intervals(s, seq)))
        for t in targets
            push!(stim,s  => SNN.PoissonStimulus(E, :he, t, μ=4.f0, param=param, name="$s"))
        end
    end
    stim = dict2ntuple(stim)
    stim, seq
end

export step_input_sequence