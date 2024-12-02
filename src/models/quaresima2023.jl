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

# # Alessio's network
# Ne = 2000         # number of excitatory neurons
# p = 0.2           # connection probability
# initialEE = 2.86  # pF
# numSynapses = p*Ne^2
# totalMass = initialEE*numSynapses

# # Smaller network
# NeSmall = 1000
# newP = 0.2
# newInitialEE = totalMass/(newP*NeSmall^2)

quaresima2023 = (
    plasticity = (
        iSTDP_rate = SNN.iSTDPParameterRate(η = 0.2, τy = 5ms, r=10Hz, Wmax = 273.4pF, Wmin = 0.1pF),
        iSTDP_potential =SNN.iSTDPParameterPotential(η = 0.2, v0 = -70mV, τy = 5ms, Wmax = 273.4pF, Wmin = 0.1pF),
        vstdp = SNN.vSTDPParameter(
                A_LTD = 4.0f-4,  #ltd strength
                A_LTP = 14.0f-4, #ltp strength
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

        EIf = (p = 0.2,  μ = log(5.), dist = LogNormal, σ = 0),
        IsIf = (p = 0.2, μ = log(0.5),  dist = LogNormal, σ = 0.25),
        IfIf = (p = 0.2, μ = log(6.), dist = LogNormal, σ = 0.14),

        EdIs = (p = 0.2, μ = log(5.), dist = LogNormal, σ = 0),
        IfIs = (p = 0.2, μ = log(1.), dist = LogNormal, σ = 0.1),
        IsIs = (p = 0.2, μ = log(6.), dist = LogNormal, σ = 0.1),
    )
    # connectivity = (
    #     EdE = (p = 0.3,  μ = (10.78*(0.2*2000^2))/(0.3*1000^2), dist = Normal, σ = 1),
    #     IfE = (p = 0.3,  μ = log((5.27*(0.2*2000^2))/(0.3*1000^2)),  dist = LogNormal, σ = 0.),
    #     IsE = (p = 0.3,  μ = log((5.27*(0.2*2000^2))/(0.3*1000^2)),  dist = LogNormal, σ = 0.),

    #     EIf = (p = 0.3,  μ = log((15.8*(0.2*2000^2))/(0.3*1000^2)), dist = LogNormal, σ = 0), # this
    #     IsIf = (p = 0.3, μ = log((0.83*(0.2*2000^2))/(0.3*1000^2)),  dist = LogNormal, σ = 0.25),
    #     IfIf = (p = 0.3, μ = log((16.2*(0.2*2000^2))/(0.3*1000^2)), dist = LogNormal, σ = 0.14),

    #     EdIs = (p = 0.3, μ = log((15.8*(0.2*2000^2))/(0.3*1000^2)), dist = LogNormal, σ = 0), # this
    #     IfIs = (p = 0.3, μ = log((1.47*(0.2*2000^2))/(0.3*1000^2)), dist = LogNormal, σ = 0.1),
    #     IsIs = (p = 0.3, μ = log((16.2*(0.2*2000^2))/(0.3*1000^2)), dist = LogNormal, σ = 0.1),
    # )
)


export quaresima2023