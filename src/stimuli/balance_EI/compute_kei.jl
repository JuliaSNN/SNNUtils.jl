@doc raw"""
    get_model(L, NAR, Nd; Vs = -55)

Passive properties of a dendritic neuron with dendrites of length `L` (human dendritic
physiology, `create_dendrite(L)`), the fixed AdEx-like somatic parameters of the Tripod model
(``C = 281`` pF, ``g_L = 40`` nS, ``E_L = V_r = -70.6`` mV, ``a = 4`` nS, ``b = 80.5`` pA,
``\tau_w = 144`` ms) and dendritic receptors with NMDA/AMPA ratio `NAR`
(`EyalEquivalentNAR`: AMPA ``g_0 = 0.73(1 + 1.31/0.73 - NAR)`` nS, NMDA ``g_0 = 0.73\,NAR`` nS
with ``\tau_d = 35`` ms, and the `MilesGabaDend` GABA receptors).

# Arguments
- `L`: dendritic length (library length unit, e.g. `200um`).
- `NAR`: NMDA/AMPA ratio.
- `Nd`: number of dendrites (returned; used by `residual_current` and `compute_kei`).
- `Vs = -55`: target somatic potential (mV), returned unchanged.

# Returns
A `Dict{Symbol,Any}` (from `@symdict`) with `gax` (axial conductance), `gm` (dendritic membrane
conductance), `gl`, `a`, `Vs`, `Vr`, `C`, `dend_syn` (vector of the four dendritic `Receptor`s,
AMPA, NMDA, GABAa, GABAb), `Nd`.

!!! note "Changed after SNNUtils 0.2.9"
    Up to 0.2.9 `get_model` (and therefore `residual_current`, `compute_kei`, `optimal_kei`)
    always threw: it used keywords and types removed from SNNModels (`PostSpike(A = ...)`,
    `DendNeuronParameter(C = ..., ...)`, `AdExSoma`, `Multipod`, `synapsearray`) and
    `EyalEquivalentNAR`, defined only in an unloaded file. It now computes the same quantities
    from `create_dendrite` and `AdExParameter`.
"""
function get_model(L, NAR, Nd; Vs = -55)
    adex = AdExParameter(
        C = 281pF,
        gl = 40nS,
        Vr = -70.6mV,
        El = -70.6mV,
        ΔT = 2mV,
        Vt = 1000.0f0,
        a = 4nS,
        b = 80.5pA,
        τw = 144ms,
    )
    d = SNNModels.create_dendrite(L)
    gax = d.gax
    gm = d.gm
    C = adex.C
    gl = adex.gl
    a = adex.a
    Vr = adex.Vr
    dend_syn = _eyal_equivalent_nar(NAR)
    return @symdict gax gm gl a Vs Vr C dend_syn Nd
end

# Dendritic receptors with NMDA/AMPA ratio `NAR` (same values as `EyalEquivalentNAR` in
# models/quaresima_2024_updown.jl): AMPA, NMDA, GABAa, GABAb.
function _eyal_equivalent_nar(NAR, τd = 35ms)
    NAR0 = 1.31 / 0.73
    glu = SNNModels.Glutamatergic(
        SNNModels.Receptor(E_rev = 0.0, τr = 0.25, τd = 2.0, g0 = 0.73(1 + NAR0 - NAR)),
        SNNModels.ReceptorVoltage(E_rev = 0.0, τr = 8, τd = τd, g0 = 0.73 * NAR, nmda = 1.0f0),
    )
    return SNNModels.Receptors(glu, SNNModels.MilesGabaDend)
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
``V_d = \left(g_L (V_s - V_r) + a (V_s - V_r) + N_d g_{ax} V_s\right) / (N_d g_{ax})``, and with
``I_{comp} = g_{ax}(V_s - V_d) - g_m (V_d - V_r)`` (the current needed to hold the dendrite at
``V_d``, the same expression as in `compute_kei`; up to SNNUtils 0.2.9 the leak term had the
opposite sign, so `residual_current` was not zero at the ``k_{EI}`` returned by `compute_kei`) the currents are, for every receptor
``r`` of the dendritic synapse (first two receptors excitatory, last two inhibitory),
``I_r = -\bar g_r (\tau_{d,r} - \tau_{r,r})\, \lambda_r\, B_r(V_d)\, (V_d - E_r)``, where ``B_r``
is [`nmda_curr`](@ref) for NMDA receptors and 1 otherwise.

All keyword arguments except `currents` and `Vs` are required. (Up to SNNUtils 0.2.9 their
defaults referred to themselves and raised `UndefVarError` when omitted.)

# Returns
`sum(exc) + sum(inh) + I_comp`, or `(exc, inh, I_comp)` if `currents = true`.

The neuron is built by [`get_model`](@ref) (up to SNNUtils 0.2.9 this function always threw,
see there).
"""
function residual_current(;
    λ,
    kIE,
    L,
    NAR,
    Nd,
    currents = false,
    Vs = -55mV,
)
    @unpack gax, gm, gl, a, Vs, Vr, C, dend_syn, Nd = get_model(L, NAR, Nd, Vs = Vs)
    @debug "Computing residual current for λ=$λ, kIE=$kIE, NAR=$NAR, Nd=$Nd"

    ## Target dendritic voltage
    Vd = (gl*(Vs - Vr) + a*(Vs - Vr) + Nd*gax*(Vs))/(Nd*gax)

    ## Currents
    comp_curr = (gax*(Vs - Vd) - gm*(Vd - Vr))
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
k_{EI} = \max\left(0,\; -\frac{\sum_{exc} I_r + I_{comp}}{\sum_{inh} I_r}\right).
```

# Arguments
- `L`: dendritic length; `rate`: excitatory input rate (kHz in library units).
- `NAR = 1.8`: NMDA/AMPA ratio; `Nd = 2`: number of dendrites; `Vs = -55mV`: somatic potential.

The neuron is built by [`get_model`](@ref) (up to SNNUtils 0.2.9 this function always threw,
see there).
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
"""
function optimal_kei(l, NAR, Nd; kwargs...)
    rates = exp10.(range(-2, stop = 3, length = 100))
    [compute_kei(l, rate; NAR = NAR, Nd = Nd) for rate in rates]
end



export get_model, residual_current, optimal_kei, compute_kei
