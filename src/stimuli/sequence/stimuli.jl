"""
    step_input(; inputs, network::NamedTuple, sym::Symbol = :glu,
               targets::Union{Vector{Symbol},Vector{Nothing}} = [nothing], pop::Symbol = :Exc,
               p_post::Real, peak_rate::Real, proj_strength::Real, kwargs...)

Create one Poisson input group per symbol in `inputs`, each projecting onto population `pop` of
`network`.

For every symbol `s` in `inputs` the function builds a
`MultiCompartmentStimulusGroup(PoissonInterval(rate = peak_rate, μ = proj_strength),
network.pop[pop], sym, targets; p_post, neurons = :p_post, name = "s")`: a Poisson source with
rate `peak_rate` that is active only inside its intervals (set them later with
[`update_stimuli!`](@ref) or `set_intervals!`), connected with strength `proj_strength` to a
random fraction `p_post` of the neurons of the target population, on the receptor `sym` of each
compartment in `targets`.

With the default `targets = [nothing]` the input targets a point neuron (one element per
group); pass the compartments for dendritic neurons (e.g. `[:d1, :d2]` for a `Tripod`). (Up to
SNNUtils 0.2.9, with SNNModels 1.8.4, the default raised a `MethodError`.)

# Arguments
- `inputs`: iterable of stimulus names (e.g. `stimuli_names(lexicon).all`).
- `network::NamedTuple`: model with a `pop` field.
- `sym::Symbol = :glu`: target receptor/variable.
- `targets = [nothing]`: target compartments (`[nothing]` for a point neuron, e.g. `[:d1, :d2]`
  for a Tripod).
- `pop::Symbol = :Exc`: key of the target population in `network.pop`.
- `p_post::Real`: probability that a neuron of `pop` receives the input.
- `peak_rate::Real`: Poisson rate inside the active intervals (library unit: kHz, write e.g.
  `8Hz`).
- `proj_strength::Real`: synaptic strength `μ` of the projection.
- `kwargs...`: ignored.

# Returns
A `NamedTuple` mapping `Symbol(s)` to the corresponding stimulus group.

# Example
```julia
using SpikingNeuralNetworks, SNNUtils
SNN.@load_units
Exc = SNN.Tripod(N = 100, name = "Exc")
network = SNN.compose(; Exc)
lexicon = get_lexicon([:AB, :BA], 50ms)
stim = step_input(; inputs = stimuli_names(lexicon).all, network, pop = :Exc,
                  targets = [:d1, :d2], p_post = 0.1, peak_rate = 8Hz, proj_strength = 2.0)
keys(stim)  # :w_AB, :w_BA, :p_A, :p_B (in some order)
```
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
        my_input = MultiCompartmentStimulusGroup(
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
    set_stimuli!(; model, seq, words = true, phonemes = true)

Activate or deactivate the word and phoneme stimuli of `model`.

For every word `w` in `seq.symbols.words` the stimulus `model.stim[:w_<w>]` (if present) is set
active with `set_active!(stim, words)`; likewise `model.stim[:p_<ph>]` for every phoneme with
`phonemes`. Stimuli that are not in `model.stim` are skipped. The names follow
[`stimuli_names`](@ref). Returns `nothing`; the model is modified in place.

# Arguments
- `model`: model whose `stim` field contains the stimuli.
- `seq`: lexicon or sequence with `symbols.words` and `symbols.phonemes`.
- `words::Bool = true`, `phonemes::Bool = true`: activation state to set.
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
    update_stimuli!(; seq, model)

Set the active intervals of the word and phoneme stimuli of `model` from the sequence `seq`.

For every word `w` (phoneme `ph`) of `seq.symbols`, the intervals returned by
[`sign_intervals`](@ref)`(w, seq)` are assigned with `set_intervals!` to
`model.stim[:w_<w>]` (`model.stim[:p_<ph>]`); stimuli not present in `model.stim` are skipped.
Returns `model`.

# Arguments
- `seq`: sequence created by [`generate_sequence`](@ref).
- `model`: model whose `stim` field contains the stimuli (e.g. created with
  [`step_input`](@ref)).
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