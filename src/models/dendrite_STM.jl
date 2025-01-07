dendritic_stp_network = let 
    EyalGluDend = Glutamatergic(
                Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73),
                ReceptorVoltage(E_rev = 0.0, τr = 8, τd = 35.0, g0 = 1.31, nmda = 1.0f0),
            )
    DuarteGluSoma =  Glutamatergic(
            Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73), 
            ReceptorVoltage(E_rev = 0.0, nmda = 0.0f0),
        )
    MilesGabaDend =  GABAergic(
            Receptor(E_rev = -70.0, τr = 4.8, τd = 29.0, g0 = 0.27), 
            Receptor(E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.006), # τd = 100.0
        )
    MilesGabaSoma =  GABAergic(Receptor(E_rev = -70.0, τr = 0.1, τd = 15.0, g0 = 0.38), Receptor()) 
    exc = (
        dends = [(150um, 400um), (150um, 400um)],  # dendritic lengths
        NMDA = NMDAVoltageDependency(
            b = 3.36,  # NMDA voltage dependency parameter
            k = -0.077,  # NMDA voltage dependency parameter
            mg = 1.0f0,  # NMDA voltage dependency parameter
        ),
            # After spike timescales and membrane
        param= AdExSoma(
            C = 281pF,  # membrane capacitance
            gl = 40nS,  # leak conductance
            R = nS / 40nS * SNN.GΩ,  # membrane resistance
            τm = 281pF / 40nS,  # membrane time constant
            Er = -70.6mV,  # reset potential
            Vr = -55.6mV,  # resting potential
            Vt = -50.4mV,  # threshold potential
            ΔT = 2mV,  # slope factor
            τw = 144ms,  # adaptation time constant
            a = 4nS,  # subthreshold adaptation conductance
            b = 10.5pA,  # spike-triggered adaptation current
            AP_membrane = 2.0f0mV,  # action potential membrane potential
            BAP = 1.0f0mV,  # burst afterpotential
            up = 1ms,  # refractory period
            τabs = 2ms,  # absolute refractory period
        ),
        dend_syn = Synapse(EyalGluDend, MilesGabaDend), # defines glutamaterbic and gabaergic receptors in the dendrites
        soma_syn=  Synapse(DuarteGluSoma, MilesGabaSoma)  # connect EyalGluDend to MilesGabaDend
    )
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
        gsyn_e = 1.04nS,
        gsyn_i = 0.84nS, 
    )

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
        gsyn_e = 0.56nS, 
        gsyn_i = 0.59nS, 
        a = 4nS,
        b = 80.5pA,       #(pA) 'sra' current increment
        τw = 144ms,        #(s) adaptation time constant (~Ca-activated K current inactivation)
    )
    plasticity = (
        iSTDP_rate = SNN.iSTDPParameterRate(η = 0.2, τy = 10ms, r=10Hz, Wmax = 200.0pF, Wmin = 2.78pF),
        iSTDP_potential =SNN.iSTDPParameterPotential(η = 0.2, v0 = -70mV, τy = 20ms, Wmax = 200.0pF, Wmin = 2.78pF),
        vstdp = SNN.vSTDPParameter(
                A_LTD = 4.0f-4,  #ltd strength
                A_LTP = 14.0f-4, #ltp strength
                θ_LTD = -40.0,  #ltd voltage threshold # set higher
                θ_LTP = -20.0,  #ltp voltage threshold
                τu = 15.0,  #timescale for u variable
                τv = 45.0,  #timescale for v variable
                τx = 20.0,  #timescale for x variable
                Wmin = 2.78,  #minimum ee strength
                Wmax = 81.4,   #maximum ee strength
            ),
        stm=SNN.STPParameter()

    )
    connectivity = (
        EdE = (p = 0.2,  μ = 10.78, dist = Normal, σ = 1),
        IfE = (p = 0.2,  μ = log(15.27),  dist = LogNormal, σ = 0.),
        IsE = (p = 0.2,  μ = log(15.27),  dist = LogNormal, σ = 0.),

        EIf = (p = 0.2,  μ = log(15.8), dist = LogNormal, σ = 0.),
        IsIf = (p = 0.2, μ = log(0.83),  dist = LogNormal, σ = 0.),
        IfIf = (p = 0.2, μ = log(16.2), dist = LogNormal, σ = 0.),

        EdIs = (p = 0.2, μ = log(15.8), dist = LogNormal, σ = 0.),
        IfIs = (p = 0.2, μ = log(1.47), dist = LogNormal, σ = 0.),
        IsIs = (p = 0.2, μ = log(16.2), dist = LogNormal, σ = 0.),
    )

    noise_params = let
        exc_soma = (param=4.0kHz,  μ=2.8f0,  cells=:ALL, name="noise_exc_soma")
        exc_dend = (param=0.0kHz,  μ=0.f0,  cells=:ALL, name="noise_exc_dend")
        inh1 = (param=2.5kHz,  μ=2.8f0,  cells=:ALL,     name="noise_inh1")
        inh2 = (param=3.5kHz,  μ=2.8f0, cells=:ALL,     name="noise_inh2")
        (exc_soma=exc_soma, exc_dend=exc_dend, inh1=inh1, inh2=inh2)
    end

    (exc=exc, pv=PV, sst=SST, plasticity,connectivity, noise_params)
end