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

quaresima2023 = (
    plasticity = (
        iSTDP_rate = SNN.iSTDPParameterRate(η = 1., τy = 5ms, r=5Hz, Wmax = 273.4pF, Wmin = 0.1pF), 
        iSTDP_potential =SNN.iSTDPParameterPotential(η = 0.1, v0 = -70mV, τy = 20ms, Wmax = 273.4pF, Wmin = 0.1pF),        
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
        IfE = (p = 0.2,  μ = log(15.7),  dist = LogNormal, σ = 0.1),
        IsE = (p = 0.2,  μ = log(2.1),  dist = LogNormal, σ = 0.1),

        EIf = (p = 0.2,  μ = log(10.8), dist = LogNormal, σ = 0),
        IsIf = (p = 0.2, μ = log(1.4),  dist = LogNormal, σ = 0.25),
        IfIf = (p = 0.2, μ = log(16.2), dist = LogNormal, σ = 0.14),

        EdIs = (p = 0.2, μ = log(10.8), dist = LogNormal, σ = 0),
        IfIs = (p = 0.2, μ = log(0.83), dist = LogNormal, σ = 0.),
        IsIs = (p = 0.2, μ = log(0.83), dist = LogNormal, σ = 0.),
    )
)

function ballstick_network(;
            I1_params, 
            I2_params, 
            E_params, 
            connectivity,
            plasticity,
            NE=1000)
    # Number of neurons in the network
    NI = NE ÷ 4
    NI1 = round(Int,NI * 0.35)
    NI2 = round(Int,NI * 0.65)
    # Import models parameters
    # Define interneurons I1 and I2
    @unpack dends, NMDA, param, soma_syn, dend_syn = E_params
    E = SNN.BallAndStickHet(; N = NE, soma_syn = soma_syn, dend_syn = dend_syn, NMDA = NMDA, param = param, name="Exc")
    I1 = SNN.IF(; N = NI1, param = I1_params, name="I1_pv")
    I2 = SNN.IF(; N = NI2, param = I2_params, name="I2_sst")
    # Define synaptic interactions between neurons and interneurons
    E_to_E = SNN.SpikingSynapse(E, E, :he, :d ; connectivity.EdE..., param= plasticity.vstdp)
    E_to_I1 = SNN.SpikingSynapse(E, I1, :ge; connectivity.IfE...)
    E_to_I2 = SNN.SpikingSynapse(E, I2, :ge; connectivity.IsE...)
    I1_to_E = SNN.SpikingSynapse(I1, E, :hi, :s; param = plasticity.iSTDP_rate, connectivity.EIf...)
    I1_to_I1 = SNN.SpikingSynapse(I1, I1, :gi; connectivity.IfIf...)
    I1_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; connectivity.IfIs...)
    I2_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; connectivity.IsIs...)
    I2_to_E = SNN.SpikingSynapse(I2, E, :hi, :d; param = plasticity.iSTDP_potential, connectivity.EdIs...)
    I2_to_I1 = SNN.SpikingSynapse(I2, I1, :gi; connectivity.IsIf...)
    # Define normalization
    norm = SNN.SynapseNormalization(NE, [E_to_E], param = SNN.MultiplicativeNorm(τ = 20ms))
    # background noise
    stimuli = Dict(
        :noise_s   => SNN.PoissonStimulus(E,  :he_s,  param=4.0kHz, cells=:ALL, μ=5.f0, name="noise_s"),
        :noise_i1  => SNN.PoissonStimulus(I1, :ge,   param=1.8kHz, cells=:ALL, μ=1.f0,  name="noise_i1"),
        :noise_i2  => SNN.PoissonStimulus(I2, :ge,   param=2.5kHz, cells=:ALL, μ=1.5f0, name="noise_i2")
    )
    # Store neurons and synapses into a dictionary
    pop = dict2ntuple(@strdict E I1 I2)
    syn = dict2ntuple(@strdict E_to_I1 E_to_I2 I1_to_E I2_to_E I1_to_I1 I2_to_I2 I1_to_I2 I2_to_I1 E_to_E norm)
    # Return the network as a model
    merge_models(pop, syn, stimuli)
end


export quaresima2023, ballstick_network
# quaresima2023 = (
#     plasticity = (
#         iSTDP_rate = SNN.iSTDPParameterRate(η = 0.2Hz, τy = 20ms, r=10Hz, Wmax = 243.0pF, Wmin = 2.78pF), # CHANGED η = 1., τy = 5ms, r=5Hz, Wmax = 273.4pF, Wmin = 0.1pF
#         iSTDP_potential =SNN.iSTDPParameterPotential(η = 0.2Hz, v0 = -70mV, τy = 5ms, Wmax = 243.0pF, Wmin = 2.78pF), # CHANGED η = 0.1, v0 = -70mV, τy = 20ms, Wmax = 273.4pF, Wmin = 0.1pF    
#         vstdp = SNN.vSTDPParameter(
#                 A_LTD = 4.0f-5Hz,  #ltd strength          # made 10 times slower #ADDED units
#                 A_LTP = 14.0f-5Hz, #ltp strength #ADDED units
#                 θ_LTD = -40.0mV,  #ltd voltage threshold # set higher #ADDED units
#                 θ_LTP = -20.0mV,  #ltp voltage threshold #ADDED units
#                 τu = 15.0ms,  #timescale for u variable #ADDED units
#                 τv = 45.0ms,  #timescale for v variable #ADDED units
#                 τx = 20.0ms,  #timescale for x variable #ADDED units
#                 Wmin = 2.78pF,  #minimum ee strength # CHANGED 1.78
#                 Wmax = 41.4pF,   #maximum ee strength #ADDED units
#             )
#     ),
    # connectivity = (
    #     EdE = (p = 0.2,  μ = 10.78, dist = Normal, σ = 0.), # CHANGED μ = 10.
    #     IfE = (p = 0.2,  μ = log(5.27),  dist = LogNormal, σ = 0.),
    #     IsE = (p = 0.2,  μ = log(5.27),  dist = LogNormal, σ = 0.),

    #     EIf = (p = 0.2,  μ = log(15.8), dist = LogNormal, σ = 0.),
    #     IsIf = (p = 0.2, μ = log(0.83),  dist = LogNormal, σ = 0.),
    #     IfIf = (p = 0.2, μ = log(16.2), dist = LogNormal, σ = 0.),

    #     EdIs = (p = 0.2, μ = log(15.8), dist = LogNormal, σ = 0.),
    #     IfIs = (p = 0.2, μ = log(1.47), dist = LogNormal, σ = 0.),
    #     IsIs = (p = 0.2, μ = log(16.2), dist = LogNormal, σ = 0.),
    # )
# )
