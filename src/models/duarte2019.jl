# Duarte2019 model parameters without NMDA and GABA_B synapses
duarte2019 = (
    PV = SNN.IFParameterGsyn(
        τm = 104.52pF / 9.75nS,
        El = -64.33mV,
        Vt = -38.97mV,
        Vr = -57.47mV,
        τabs = 0.5ms,
        τre = 0.18ms,
        τde = 0.70ms,
        τri = 0.19ms,
        τdi = 2.50ms,
        gsyn_e = 1.04nS, # ADDED 
        gsyn_i = 0.84nS, # ADDED 
    ),
    
    SST = SNN.IFParameterGsyn(
        τm = 102.86pF / 4.61nS,
        El = -61mV,
        Vt = -34.4mV,
        Vr = -47.11mV,
        τabs = 1.3ms,
        τre = 0.18ms,
        τde = 1.80ms,
        τri = 0.19ms,
        τdi = 5.00ms,
        gsyn_e = 0.56nS, # 0.8nS,
        gsyn_i = 0.59nS, # 0.7nS,
        a = 4nS,
        b = 80.5pA,       #(pA) 'sra' current increment
        τw = 144ms,        #(s) adaptation time constant (~Ca-activated K current inactivation)
    ),
    AdEx = AdExParameterGsyn(El = -76.43, 
                            τm = 116.5pF/ 4.64nS, 
                        Vt = -44.45, 
                        τabs = 2.05ms,
                        τre = 0.25ms,
                        τde = 2.0ms,
                        τri = 0.5ms, 
                        τdi = 6.0ms,
                        gsyn_e = 0.73,
                        gsyn_i = 0.265
                        )
)
export duarte2019