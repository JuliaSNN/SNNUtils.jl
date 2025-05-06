quaresima2023 = (
    plasticity = (
        iSTDP_rate = SNN.iSTDPRate(η = 0.2, τy = 5ms, r=10Hz, Wmax = 243.4pF, Wmin = 0.1pF), 
        iSTDP_potential =SNN.iSTDPPotential(η = 0.2, v0 = -70mV, τy = 5ms, Wmax = 243.4pF, Wmin = 0.1pF),        
        vstdp = SNN.vSTDPParameter(
                A_LTD = 14.0f-4,  #ltd strength
                A_LTP = 8.0f-4, #ltp strength
                θ_LTD = -60.0,  #ltd voltage threshold # set higher
                θ_LTP = -25.0,  #ltp voltage threshold
                τu = 15.0,  #timescale for u variable
                τv = 45.0,  #timescale for v variable
                τx = 20.0,  #timescale for x variable
                Wmin = 2.78,  #minimum ee strength # CHANGED Wmin = 1.78
                Wmax = 41.4,   #maximum ee strength
            )
    ),
    connectivity = (
        E_to_Ed = (p = 0.2,  μ = 10.78, dist = Normal, σ = 1),
        E_to_If = (p = 0.2,  μ = log(16.),  dist = LogNormal, σ = 0.),
        E_to_Is = (p = 0.2,  μ = log(16.),  dist = LogNormal, σ = 0.),

        If_to_E = (p = 0.2,  μ = log(16.8), dist = LogNormal, σ = 0),
        If_to_Is = (p = 0.2, μ = log(5.83),  dist = LogNormal, σ = 0.),
        If_to_If = (p = 0.2, μ = log(16.2), dist = LogNormal, σ = 0.),

        Is_to_Ed = (p = 0.2, μ = log(16.0), dist = LogNormal, σ = 0),
        Is_to_If = (p = 0.2, μ = log(5.47), dist = LogNormal, σ = 0.),
        Is_to_Is = (p = 0.2, μ = log(16.2), dist = LogNormal, σ = 0.),
    )
)
export quaresima2023, ballstick_network