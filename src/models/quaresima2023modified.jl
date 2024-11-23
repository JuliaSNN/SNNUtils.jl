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

        EIf = (p = 0.2,  μ = log(10.8), dist = LogNormal, σ = 0),
        IsIf = (p = 0.2, μ = log(5.4),  dist = LogNormal, σ = 0.25),
        IfIf = (p = 0.2, μ = log(15.2), dist = LogNormal, σ = 0.14),

        EdIs = (p = 0.2, μ = log(10.0), dist = LogNormal, σ = 0),
        IfIs = (p = 0.2, μ = log(5.83), dist = LogNormal, σ = 0.1),
        IsIs = (p = 0.2, μ = log(5.83), dist = LogNormal, σ = 0.1),
    )
)

function ballstick_network(;
            NE::Int,
            I1_params, 
            I2_params, 
            E_params, 
            connectivity,
            plasticity,
            )
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
    I1_to_I1 = SNN.SpikingSynapse(I1, I1, :gi; connectivity.IfIf...)
    I1_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; connectivity.IfIs...)
    I2_to_I2 = SNN.SpikingSynapse(I1, I2, :gi; connectivity.IsIs...)
    I2_to_I1 = SNN.SpikingSynapse(I2, I1, :gi; connectivity.IsIf...)
    I1_to_E = SNN.SpikingSynapse(I1, E, :hi, :s; param = plasticity.iSTDP_rate, connectivity.EIf...)
    I2_to_E = SNN.SpikingSynapse(I2, E, :hi, :d; param = plasticity.iSTDP_potential, connectivity.EdIs...)
    # Define normalization
    norm = SNN.SynapseNormalization(NE, [E_to_E], param = SNN.MultiplicativeNorm(τ = 20ms))
    # background noise
    stimuli = Dict(
        :noise_s   => SNN.PoissonStimulus(E,  :he_s,  param=6.0kHz, cells=:ALL, μ=5.f0, name="noise_s",),
        :noise_i1  => SNN.PoissonStimulus(I1, :ge,   param=2.5kHz, cells=:ALL, μ=1.f0,  name="noise_i1"),
        :noise_i2  => SNN.PoissonStimulus(I2, :ge,   param=3.0kHz, cells=:ALL, μ=1.8f0, name="noise_i2")
    )
    # Store neurons and synapses into a dictionary
    pop = dict2ntuple(@strdict E I1 I2)
    syn = dict2ntuple(@strdict E_to_I1 E_to_I2 I1_to_E I2_to_E I1_to_I1 I2_to_I2 I1_to_I2 I2_to_I1 E_to_E norm)
    # Return the network as a model
    merge_models(pop, syn, stimuli, silent=true)
end


export quaresima2023, ballstick_network