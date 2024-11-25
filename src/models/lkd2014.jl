
"""
Litwin-Kumar, A., & Doiron, B. (2014). Formation and maintenance of neuronal assemblies through synaptic plasticity. Nature Communications, 5(1). https://doi.org/10.1038/ncomms6319
"""

LKD2014 = (
    AdEx = AdExParameter(
                        El = -70mV, 
                        Vt = -52.0mV, 
                        τm = 300pF /15.0nS, 
                        R = 1/(15.0nS),
                        Vr = -60.0f0mV,
                        τabs = 1ms,       
                        τri=0.5,
                        τdi=2.0,
                        τre=1.0,
                        τde=6.0,
                        E_i = -75mV,
                        E_e = 0mV,
                        At = 10mV
                        ),
    PV = IFParameter(
        El = -62.0mV,
        Vr = -57.47mV,   #(mV)
        Vt = -52.0mV,
        τm = 20ms,
        a = 0.0,
        b = 0.0,
        τw = 144,
        τri=0.5,
        τdi=2.0,
        τre=1.0,
        τde=6.0,
    )
)


export LKD2014