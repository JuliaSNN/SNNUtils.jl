@doc raw"""
    get_model(L, NAR, Nd; Vs = -55)

Build a single dendritic neuron (two dendrites of length `L`, via `Tripod`; if that fails, a
`Multipod` with `Nd` dendrites) with fixed AdEx-like somatic parameters
(``C = 281`` pF, ``g_L = 40`` nS, ``E_L = V_r = -70.6`` mV, ``a = 4`` nS, ``b = 80.5`` pA,
``\tau_w = 144`` ms, spike threshold disabled with ``V_t = 1000`` mV) and Eyal NMDA kinetics,
and return its passive properties.

# Arguments
- `L`: dendritic length (library length unit, e.g. `200um`).
- `NAR`: NMDA/AMPA ratio passed to `EyalEquivalentNAR`.
- `Nd`: number of dendrites (used by the `Multipod` fallback and returned).
- `Vs = -55`: target somatic potential (mV), returned unchanged.

# Returns
A `Dict{Symbol,Any}` (from `@symdict`) with `gax` (axial conductance), `gm` (dendritic membrane
conductance), `gl`, `a`, `Vs`, `Vr`, `C`, `dend_syn` (dendritic receptors from
`EyalEquivalentNAR(NAR)`), `Nd`.

# Status
Does not run with SNNModels 1.8.4: `get_model` calls `PostSpike(A = 10.0, τA = 30.0)`, whose
keyword `A` does not exist (nor do the `C`, `gl`, ... keywords passed to `DendNeuronParameter`) (the error is caught, but the fallback branch calls `PostSpike` in the
same way and would then need the undefined `AdExSoma`/`Multipod`); it also needs `EyalEquivalentNAR` (defined only in the unloaded file
`models/quaresima_2024_updown.jl`) and `synapsearray` (exported but not defined by SNNModels).
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
            NMDA = EyalNMDA,
        )
        E = Tripod(N = 1, param = neuron)
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


@doc raw"""
    nmda_curr(V)

NMDA magnesium-block factor at membrane potential `V` (mV), with the `EyalNMDA` parameters
(`mg`, `b`, `k`) of SNNModels:

```math
B(V) = \left(1 + \frac{[\mathrm{Mg}]}{b}\, e^{k V}\right)^{-1}
```

Returns a `Float32` (dimensionless, between 0 and 1). Not exported.
"""
function nmda_curr(V)
    @unpack mg, b, k = EyalNMDA
    return (1.0f0 + (mg / b) * exp(k * Float32(V)))^-1
end

@doc raw"""
    residual_current(; λ, kIE, L, NAR, Nd, currents = false, Vs = -55mV)

Residual current at the dendrite of the neuron built by [`get_model`](@ref) when the soma is held
at `Vs`, with excitatory input rate `λ` and inhibitory input rate `kIE * λ` on each dendrite.

The dendritic potential is set to
``V_d = \left(g_L (V_s - V_r) + a (V_s - V_r) + N_d g_{ax} V_s
ight) / (N_d g_{ax})``, and with
``I_{comp} = g_{ax}(V_s - V_d) + g_m (V_d - V_r)`` the currents are, for every receptor
``r`` of the dendritic synapse (first two receptors excitatory, last two inhibitory),
``I_r = -\bar g_r (\tau_{d,r} - \tau_{r,r})\, \lambda_r\, B_r(V_d)\, (V_d - E_r)``, where ``B_r``
is [`nmda_curr`](@ref) for NMDA receptors and 1 otherwise.

All keyword arguments except `currents` and `Vs` are required (their defaults refer to
themselves and raise `UndefVarError` if omitted).

# Returns
`sum(exc) + sum(inh) + I_comp`, or `(exc, inh, I_comp)` if `currents = true`.

# Status
Does not run with SNNModels 1.8.4: `get_model` calls `PostSpike(A = 10.0, τA = 30.0)`, whose
keyword `A` does not exist (nor do the `C`, `gl`, ... keywords passed to `DendNeuronParameter`) (the error is caught, but the fallback branch calls `PostSpike` in the
same way and would then need the undefined `AdExSoma`/`Multipod`); it also needs `EyalEquivalentNAR` (defined only in the unloaded file
`models/quaresima_2024_updown.jl`) and `synapsearray` (exported but not defined by SNNModels).
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

@doc raw"""
    compute_kei(L, rate; NAR = 1.8, Nd = 2, Vs = -55mV)

Inhibitory-to-excitatory rate ratio ``k_{EI}`` that makes the net dendritic current zero when
the soma of the neuron built by [`get_model`](@ref) is held at `Vs` and every dendrite receives
excitatory input at rate `rate`.

With ``V_d = (g_L + a)(V_s - V_r)/(N_d g_{ax}) + V_s``, ``I_{comp} = -g_{ax}(V_d - V_s) - g_m (V_d - V_r)``
and the receptor currents ``I_r`` defined as in [`residual_current`](@ref) (rate `rate` for all
receptors), it returns

```math
k_{EI} = \max\left(0,\; -\frac{\sum_{exc} I_r + I_{comp}}{\sum_{inh} I_r}
ight).
```

# Arguments
- `L`: dendritic length; `rate`: excitatory input rate (kHz in library units).
- `NAR = 1.8`: NMDA/AMPA ratio; `Nd = 2`: number of dendrites; `Vs = -55mV`: somatic potential.

# Status
Does not run with SNNModels 1.8.4: `get_model` calls `PostSpike(A = 10.0, τA = 30.0)`, whose
keyword `A` does not exist (nor do the `C`, `gl`, ... keywords passed to `DendNeuronParameter`) (the error is caught, but the fallback branch calls `PostSpike` in the
same way and would then need the undefined `AdExSoma`/`Multipod`); it also needs `EyalEquivalentNAR` (defined only in the unloaded file
`models/quaresima_2024_updown.jl`) and `synapsearray` (exported but not defined by SNNModels).
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
"""
    optimal_kei(l, NAR, Nd; kwargs...)

Evaluate [`compute_kei`](@ref)`(l, rate; NAR, Nd)` on 100 log-spaced rates between `1e-2` and
`1e3` (library rate unit, kHz) and return the vector of results. `kwargs` are ignored.
Inherits the status of `compute_kei` (does not run with SNNModels 1.8.4).
"""
function optimal_kei(l, NAR, Nd; kwargs...)
    rates = exp10.(range(-2, stop = 3, length = 100))
    [compute_kei(l, rate; NAR = NAR, Nd = Nd) for rate in rates]
end



export get_model, residual_current, optimal_kei, compute_kei
