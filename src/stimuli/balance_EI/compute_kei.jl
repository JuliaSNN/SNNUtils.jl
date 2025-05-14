
function get_model(L, NAR, Nd; Vs=-55) 
    ps = SNN.PostSpike(A= 10.0,τA= 30.0)
    adex = AdExSoma( C=281pF, gl=40nS, Vr = -70.6, Er = -70.6, ΔT = 2, Vt = 1000.f0, a = 4, b = 80.5, τw = 144, up = 1ms, τabs= 1ms)
    dend_syn = EyalEquivalentNAR(NAR) |> SNN.synapsearray
    ls = repeat([L], Nd)
    E = Multipod(
        ls;
        N=1,
        param= adex,
        postspike = ps
    )

    gax = E.gax[1,1]
    gm = E.gm[1,1]
    C = E.param.C
    gl = E.param.gl
    a = E.param.a
    Vr = E.param.Vr
    return SNN.@symdict gax gm gl a Vs Vr C dend_syn Nd
end



function nmda_curr(V) 
    @unpack mg, b, k = SNN.EyalNMDA
    return (1.0f0 + (mg / b) * SNN.exp32(k * Float32(V)))^-1
end

function residual_current(;λ=λ, kIE=kIE, L=L, NAR=NAR, Nd=Nd, currents=false, Vs=-55mV)
    @unpack gax, gm, gl, a, Vs, Vr, C, dend_syn, Nd = get_model(L, NAR, Nd, Vs=Vs)
    @debug "Computing residual current for λ=$λ, kIE=$kIE, NAR=$NAR, Nd=$Nd"

    ## Target dendritic voltage
    Vd = (gl*(Vs - Vr) + a*(Vs - Vr) + Nd*gax*(Vs))/(Nd*gax)

    ## Currents
    comp_curr = (gax*(Vs - Vd) + gm*(Vd -Vr))
    exc_syn_curr = map(dend_syn[1:2]) do syn
        (- syn.gsyn *(syn.τd - syn.τr) * λ * (syn.nmda>0 ? nmda_curr(Vd) : 1f0) * (Vd - syn.E_rev))
    end
    inh_syn_curr = map(dend_syn[3:4]) do syn
        (- syn.gsyn *(syn.τd - syn.τr) * λ * kIE * (Vd - syn.E_rev))
    end
    if currents
        return exc_syn_curr, inh_syn_curr, comp_curr
    else
       return (sum(exc_syn_curr)+  sum(inh_syn_curr)+ comp_curr)
    end
end

# function optimal_kei(L, rate; NAR=1.8, Nd=2, Vs=-52mV)
# V    kIEs = range(0, stop=2, length=100)
#     residuals =zeros(length(kIEs))
#     for (i, kIE) in enumerate(kIEs)
#         residuals[i] = residual_current(λ=rate, kIE = kIE, L=L,NAR= NAR,Nd= Nd, currents=false, Vs=Vs)
#     end
#     return kIEs[argmin(abs.(residuals))]
# end

function optimal_kei(l, NAR, Nd; kwargs...)
    rates = exp10.(range(-2, stop=3, length=100))
    [compute_kei(l, rate; NAR=NAR, Nd=Nd) for rate in rates]
end

function compute_kei(L, rate; NAR=1.8, Nd=2, Vs=-52mV)
    @unpack gax, gm, gl, a, Vs, Vr, C, dend_syn, Nd = get_model(L, NAR, Nd, Vs=Vs)
    ## Target dendritic voltage
    Vd = (gl*(Vs - Vr) + a*(Vs - Vr))/(Nd*gax) + Vs

    ## Currents
    comp_curr = (-gax*(Vd - Vs) - gm*(Vd -Vr))
    exc_syn_curr = map(dend_syn[1:2]) do syn
        (- syn.gsyn *(syn.τd - syn.τr) * rate * (syn.nmda>0 ? nmda_curr(Vd) : 1f0) * (Vd - syn.E_rev))
    end
    inh_syn_curr = map(dend_syn[3:4]) do syn
        (- syn.gsyn *(syn.τd - syn.τr) * rate *  (Vd - syn.E_rev))
    end
    curr =  - (sum(exc_syn_curr) + sum(comp_curr))/sum(inh_syn_curr)
    return maximum([0.0, curr])
end


export get_model, residual_current, optimal_kei, compute_kei