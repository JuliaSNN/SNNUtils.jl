

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
