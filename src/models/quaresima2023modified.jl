# dend_stdp = STDP(
#     a⁻ = 4.0f-5,  #ltd strength          # made 10 times slower
#     a⁺ = 14.0f-5, #ltp strength
#     θ⁻ = -40.0,  #ltd voltage threshold # set higher
#     θ⁺ = -20.0,  #ltp voltage threshold
#     τu = 15.0,  #timescale for u variable
#     τv = 45.0,  #timescale for v variable
#     τx = 20.0,  #timescale for x variable
#     τ1 = 5,    # filter for delayed voltage
#     j⁻ = 1.78,  #minimum ee strength
#     j⁺ = 41.4,   #maximum ee strength
# )


quaresima2023modified = (
    plasticity = (
        iSTDP_rate = SNN.iSTDPParameterRate(η = 0.2, τy = 5ms, r=10Hz, Wmax = 273.4pF, Wmin = 0.1pF), 
        iSTDP_potential =SNN.iSTDPParameterPotential(η = 0.2, v0 = -70mV, τy = 20ms, Wmax = 273.4pF, Wmin = 0.1pF),        
        vstdp = SNN.vSTDPParameter(
                A_LTD = 4.0f-5,  #ltd strength          # made 10 times slower
                A_LTP = 14.0f-5, #ltp strength
                θ_LTD = -40.0,  #ltd voltage threshold # set higher
                θ_LTP = -20.0,  #ltp voltage threshold
                τu = 15.0,  #timescale for u variable
                τv = 45.0,  #timescale for v variable
                τx = 20.0,  #timescale for x variable
                Wmin = 1.78,  #minimum ee strength
                Wmax = 41.4,   #maximum ee strength
            )
    ),
    connectivity = (
        EdE = (p = 0.2,  μ = 10., dist = Normal, σ = 1),
        IfE = (p = 0.2,  μ = log(2.),  dist = LogNormal, σ = 0.),
        IsE = (p = 0.2,  μ = log(2.),  dist = LogNormal, σ = 0.),

        EIf = (p = 0.2,  μ = log(10.8), dist = LogNormal, σ = 0), # this
        IsIf = (p = 0.2, μ = log(5.4),  dist = LogNormal, σ = 0.25),
        IfIf = (p = 0.2, μ = log(15.2), dist = LogNormal, σ = 0.14),

        EdIs = (p = 0.2, μ = log(10.0), dist = LogNormal, σ = 0), # this
        IfIs = (p = 0.2, μ = log(5.83), dist = LogNormal, σ = 0.1),
        IsIs = (p = 0.2, μ = log(5.83), dist = LogNormal, σ = 0.1),
    )
)


export quaresima2023modified, ballstick_network