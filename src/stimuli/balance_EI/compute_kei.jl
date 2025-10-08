"""
    get_model(L, NAR, Nd; Vs = -55)

Construct a neural model with specified parameters and return its physiological properties.

# Arguments
- `L`: Length parameter for the neuron model
- `NAR`: NAR parameter for synapse configuration
- `Nd`: Number of dendrites
- `Vs`: Somatic voltage (default: -55)

# Returns
A dictionary containing:
- `gax`: Axial conductance
- `gm`: Membrane conductance
- `gl`: Leak conductance
- `a`: Adaptation parameter
- `Vs`: Somatic voltage
- `Vr`: Resting potential
- `C`: Membrane capacitance
- `dend_syn`: Dendritic synapse array
- `Nd`: Number of dendrites

# Implementation Notes
- First attempts to create a Tripod neuron model
- Falls back to Multipod model if Tripod creation fails
- Both models use Eyal NMDA parameters
- Synapses are configured using EyalEquivalentNAR with the given NAR parameter
"""
function get_model(L, NAR, Nd; Vs = -55)
    try
        neuron = DendNeuronParameter(
            C = 281pF,
            gl = 40nS,
            Vr = -70.6,
            El = -70.6,
            ΔT = 2,
            Vt = 1000.0f0,
            a = 4,
            b = 80.5,
            τw = 144,
            up = 1ms,
            τabs = 1ms,
            ds = [L, L],
            postspike = PostSpike(A = 10.0, τA = 30.0),
            NMDA = EyalNMDA
        )
        E = Tripod( N = 1,  param = neuron)
        dend_syn = EyalEquivalentNAR(NAR) |> synapsearray
        gax = E.d1.gax[1, 1]
        gm = E.d1.gm[1, 1]
        C = E.param.C
        gl = E.param.gl
        a = E.param.a
        Vr = E.param.Vr
        return @symdict gax gm gl a Vs Vr C dend_syn Nd
    catch e
        @error "Error creating Tripod neuron: $e"
        ps = PostSpike(A = 10.0, τA = 30.0)
        adex = AdExSoma(
            C = 281pF,
            gl = 40nS,
            Vr = -70.6,
            El = -70.6,
            ΔT = 2,
            Vt = 1000.0f0,
            a = 4,
            b = 80.5,
            τw = 144,
            up = 1ms,
            τabs = 1ms,
        )
        ls = repeat([L], Nd)
        E = Multipod(ls; N = 1, NMDA = EyalNMDA, param = adex, postspike = ps)
    
        dend_syn = EyalEquivalentNAR(NAR) |> synapsearray
        gax = E.gax[1, 1]
        gm = E.gm[1, 1]
        C = E.param.C
        gl = E.param.gl
        a = E.param.a
        Vr = E.param.Vr
        return @symdict gax gm gl a Vs Vr C dend_syn Nd
    end
end


"""
    nmda_curr(V)

Compute the NMDA current based on the given membrane potential.

# Arguments
- `V`: Membrane potential (in mV)

# Returns
The NMDA current value, calculated using the Eyal NMDA parameters:
- `mg`: Magnesium concentration
- `b`: Binding constant
- `k`: Voltage scaling factor

The calculation follows the formula:
(1 + (mg/b) * exp(k*V))^-1

# Implementation Notes
- Uses the EyalNMDA parameters (mg, b, k) which should be defined elsewhere
- Converts the input voltage to Float32 for numerical stability
- Returns a Float32 value representing the NMDA current
"""
function nmda_curr(V)
    @unpack mg, b, k = EyalNMDA
    return (1.0f0 + (mg / b) * exp(k * Float32(V)))^-1
end

"""
    residual_current(; λ, kIE, L, NAR, Nd, currents = false, Vs = -55mV)

Compute the residual current at the dendrites for a neural model with given parameters.

# Arguments
- `λ`: Firing rate parameter
- `kIE`: Inhibitory current scaling factor
- `L`: Length parameter for the neuron model
- `NAR`: NAR parameter for synapse configuration
- `Nd`: Number of dendrites
- `currents`: Boolean flag to return individual currents (default: false)
- `Vs`: Somatic voltage (default: -55mV)

# Returns
If `currents` is false (default):
- The total residual current (sum of excitatory synaptic current, inhibitory synaptic current, and compartmental current)

If `currents` is true:
- A tuple containing:
  1. Array of excitatory synaptic currents
  2. Array of inhibitory synaptic currents
  3. Compartmental current

# Implementation Notes
- Uses the `get_model` function to obtain neuron parameters
- Calculates the target dendritic voltage based on somatic and resting potentials
- Computes currents using the Eyal NMDA parameters for NMDA synapses
- The inhibitory current is scaled by the `kIE` parameter
- The function includes debug logging for tracking parameter values
"""
function residual_current(;
    λ = λ,
    kIE = kIE,
    L = L,
    NAR = NAR,
    Nd = Nd,
    currents = false,
    Vs = -55mV,
)
    @unpack gax, gm, gl, a, Vs, Vr, C, dend_syn, Nd = get_model(L, NAR, Nd, Vs = Vs)
    @debug "Computing residual current for λ=$λ, kIE=$kIE, NAR=$NAR, Nd=$Nd"

    ## Target dendritic voltage
    Vd = (gl*(Vs - Vr) + a*(Vs - Vr) + Nd*gax*(Vs))/(Nd*gax)

    ## Currents
    comp_curr = (gax*(Vs - Vd) + gm*(Vd - Vr))
    exc_syn_curr = map(dend_syn[1:2]) do syn
        (
            - syn.gsyn *
            (syn.τd - syn.τr) *
            λ *
            (syn.nmda>0 ? nmda_curr(Vd) : 1.0f0) *
            (Vd - syn.E_rev)
        )
    end
    inh_syn_curr = map(dend_syn[3:4]) do syn
        (- syn.gsyn * (syn.τd - syn.τr) * λ * kIE * (Vd - syn.E_rev))
    end
    if currents
        return exc_syn_curr, inh_syn_curr, comp_curr
    else
        return (sum(exc_syn_curr) + sum(inh_syn_curr) + comp_curr)
    end
end

"""
    compute_kei(L, rate; NAR = 1.8, Nd = 2, Vs = -55mV)

Compute the optimal inhibitory current scaling factor (kei) for a neural model with given parameters such that the soma is at the required voltage and the residual dendritic current is zero.

# Arguments
- `L`: Length parameter for the neuron model
- `rate`: Firing rate parameter
- `NAR`: NAR parameter for synapse configuration (default: 1.8)
- `Nd`: Number of dendrites (default: 2)
- `Vs`: Somatic voltage (default: -52mV)

# Returns
The optimal inhibitory current scaling factor (kei), calculated as:
kei = max(0, - (sum(exc_syn_curr) + sum(comp_curr)) / sum(inh_syn_curr))

# Implementation Notes
- Uses the `get_model` function to obtain neuron parameters
- Calculates the target dendritic voltage based on somatic and resting potentials
- Computes currents using the Eyal NMDA parameters for NMDA synapses
- The function ensures the returned value is non-negative
- The calculation follows the principle of balancing excitatory and inhibitory inputs
"""
function compute_kei(L, rate; NAR = 1.8, Nd = 2, Vs = -55mV)
    @unpack gax, gm, gl, a, Vs, Vr, C, dend_syn, Nd = get_model(L, NAR, Nd, Vs = Vs)
    ## Target dendritic voltage
    Vd = (gl*(Vs - Vr) + a*(Vs - Vr))/(Nd*gax) + Vs

    ## Currents
    comp_curr = (-gax*(Vd - Vs) - gm*(Vd - Vr))
    exc_syn_curr = map(dend_syn[1:2]) do syn
        (
            - syn.gsyn *
            (syn.τd - syn.τr) *
            rate *
            (syn.nmda>0 ? nmda_curr(Vd) : 1.0f0) *
            (Vd - syn.E_rev)
        )
    end
    inh_syn_curr = map(dend_syn[3:4]) do syn
        (- syn.gsyn * (syn.τd - syn.τr) * rate * (Vd - syn.E_rev))
    end
    λ = - (sum(exc_syn_curr) + sum(comp_curr))/sum(inh_syn_curr)
    return maximum([0.0, λ])
end
function optimal_kei(l, NAR, Nd; kwargs...)
    rates = exp10.(range(-2, stop = 3, length = 100))
    [compute_kei(l, rate; NAR = NAR, Nd = Nd) for rate in rates]
end



export get_model, residual_current, optimal_kei, compute_kei
