# Duarte2019 model parameters without NMDA and GABA_B synapses
duarte2019 = (
    PV = SNN.IFParameterGsyn(
        τm = 104.52pF / 9.75nS,
        El = -64.33mV,
        Vt = -38.97mV,
        Vr = -57.47mV,
        τabs = 0.42ms,
        τre = 0.18ms,
        τde = 0.70ms,
        τri = 0.2ms,
        τdi = 2.50ms,
    ),
    SST = SNN.IFParameterGsyn(
        τm = 102.86pF / 4.61nS,
        El = -61mV,
        Vt = -34.4mV,
        Vr = -47.11mV,
        τabs = 1.34ms,
        τre = 0.18ms,
        τde = 1.80ms,
        τri = 0.19ms,
        τdi = 5.00ms,
        gsyn_e = 0.8,
        gsyn_i = 0.7,
        b = 80.5,       #(pA) 'sra' current increment
        τw = 144,        #(s) adaptation time constant (~Ca-activated K current inactivation)
    ),
    # PV = SNN.IFParameterGsyn(
    #     τm = 104.52pF / 9.75nS,
    #     El = -64.33mV,
    #     Vt = -38.97mV,
    #     Vr = -57.47mV,
    #     τabs = 0.42ms, # CHANGED 0.5ms
    #     τre = 0.18ms,
    #     τde = 0.70ms,
    #     τri = 0.19ms, # CHANGED 0.2ms
    #     τdi = 2.50ms,
    #     gsyn_e = 1.04nS, # ADDED
    #     gsyn_i = 0.84nS, # ADDED
    # ),
    # SST = SNN.IFParameterGsyn(
    #     τm = 102.86pF / 4.61nS,
    #     El = -61mV,
    #     Vt = -34.4mV,
    #     Vr = -47.11mV,
    #     τabs = 1.3ms, #  CHANGED 1.34ms
    #     τre = 0.18ms,
    #     τde = 1.80ms,
    #     τri = 0.19ms,
    #     τdi = 5.00ms,
    #     gsyn_e = 0.7nS, # 0.56nS CHANGED 0.7
    #     gsyn_i = 0.8nS, # 0.59nS CHANGED 0.8
    #     a = 4nS, # ADDED  
    #     b = 80.5pA, #(pA) 'sra' current increment # CHANGED 80.5
    #     τw = 144ms, #(s) adaptation time constant (~Ca-activated K current inactivation) # CHANGED 144ms
    # ),
    AdEx = AdExParameterGsyn(El = -76.43, 
                            τm = 116.5pF/ 4.64nS, 
                        Vt = -44.45, 
                        τabs = 2.05ms,
                        τre = 0.25ms,
                        τde = 2.0ms,
                        gsyn_e = 0.73,
                        τri = 0.5ms, 
                        τdi = 6.0ms,
                        gsyn_i = 0.265)
)
export duarte2019

## Test model
# spiketime = [1000ms]
# neurons = [[1]]
# inputs = SpikeTimeStimulusParameter(neurons=neurons, spiketimes=spiketime)

# IF = SNN.IF(; N = 1, param = duarte2019.PV)

# p1 = SNN.plot()
# SNN.monitor([IF], [:fire, :v])
# stim = SpikeTimeStimulus(1, IF, :ge, p=1.f0, param=inputs)
# sim!([IF],[SNN.EmptySynapse()], [stim], duration=1200ms, dt=0.1ms)
# p1 = SNN.vecplot!(p1, IF, :v, r=800:0.001:1200ms, neurons=[1], dt=0.1ms,    label= IF.param.gsyn_e)

# SNN.monitor([IF], [:fire, :v])
# stim = SpikeTimeStimulus(1, IF, :gi, p=1.f0, param=inputs)
# sim!([IF],[SNN.EmptySynapse()], [stim], duration=1200ms, dt=0.1ms)
# p1 = SNN.vecplot!(p1, IF, :v, r=800:0.001:1200ms, neurons=[1], dt=0.1ms, label= IF.param.gsyn_i)
# SNN.plot!(p1, ylims=:auto)

# #
# IF = SNN.IF(; N = 1, param = duarte2019.SST)

# p2 = SNN.plot()
# SNN.monitor([IF], [:fire, :v])
# stim = SpikeTimeStimulus(1, IF, :ge, p=1.f0, param=inputs)
# sim!([IF],[SNN.EmptySynapse()], [stim], duration=1200ms, dt=0.1ms)
# p2 = SNN.vecplot!(p2, IF, :v, r=800:0.001:1200ms, neurons=[1], dt=0.1ms, label= IF.param.gsyn_e)

# SNN.monitor([IF], [:fire, :v])
# stim = SpikeTimeStimulus(1, IF, :gi, p=1.f0, param=inputs)
# sim!([IF],[SNN.EmptySynapse()], [stim], duration=1200ms, dt=0.1ms)
# p2 = SNN.vecplot!(p2, IF, :v, r=800:0.001:1200ms, neurons=[1], dt=0.1ms, label= IF.param.gsyn_i)
# SNN.plot!(p2, ylims=:auto)

# #
# IF = SNN.AdEx(; N = 1, param = duarte2019.AdEx)

# p3 = SNN.plot()
# SNN.monitor([IF], [:fire, :v])
# stim = SpikeTimeStimulus(1, IF, :ge, p=1.f0, param=inputs, μ=20)
# stim2 = CurrentStimulus(IF, I_base=700pA)
# sim!([IF],[SNN.EmptySynapse()], [stim,stim2], duration=1200ms, dt=0.1ms)
# p3 = SNN.vecplot!(p3, IF, :v, r=800:0.001:1200ms, neurons=[1], dt=0.1ms, label= IF.param.gsyn_e)

# SNN.monitor([IF], [:fire, :v])
# stim = SpikeTimeStimulus(1, IF, :gi, p=1.f0, param=inputs, μ=20)
# sim!([IF],[SNN.EmptySynapse()], [stim,stim2], duration=1200ms, dt=0.1ms)
# p3 = SNN.vecplot!(p3, IF, :v, r=800:0.001:1200ms, neurons=[1], dt=0.1ms, label= IF.param.gsyn_i)
# SNN.plot!(p3, ylims=:auto)

# #
# SNN.plot(p1,p2, p3, legend=true)
# ##

# Duarte2019 model parameters with NMDA and GABA_B synapses
# const DuarteSynapsePV = let
#     E_exc = 0.00       #(mV) Excitatory reversal potential
#     E_gabaB = -90      #(mV) GABA_B reversal potential
#     E_gabaA = -75       #(mV) GABA_A reversal potential

#     gsyn_ampa = 1.6
#     # τr_ampa   = 0.087500
#     τr_ampa = 0.1#80000
#     τd_ampa = 0.7#00000

#     gsyn_nmda = 0.003#2836
#     τr_nmda = 1.0#0.990099
#     τd_nmda = 100.000000

#     gsyn_gabaA = 1.0#0.844049
#     # τr_gabaA   = 0.096154
#     τr_gabaA = 0.1#92308
#     τd_gabaA = 2.5#00000

#     gsyn_gabaB = 0.022#09419
#     τr_gabaB = 25#12.725924
#     τd_gabaB = 400#.866124

#     AMPA = Receptor(E_rev = E_exc, τr = τr_ampa, τd = τd_ampa, g0 = gsyn_ampa)
#     NMDA = ReceptorVoltage(E_rev = E_exc, τr = τr_nmda, τd = τd_nmda, g0 = gsyn_nmda)
#     GABAa = Receptor(E_rev = E_gabaA, τr = τr_gabaA, τd = τd_gabaA, g0 = gsyn_gabaA)
#     GABAb = Receptor(E_rev = E_gabaB, τr = τr_gabaB, τd = τd_gabaB, g0 = gsyn_gabaB)
#     # NMDA  = ReceptorVoltage()
#     # GABAb = Receptor()

#     Synapse(AMPA, NMDA, GABAa, GABAb)
# end

# const DuarteSynapseSST = let
#     E_exc = 0.00       #(mV) Excitatory reversal potential
#     E_gabaA = -75       #(mV) GABA_A reversal potential
#     E_gabaB = -90      #(mV) GABA_B reversal potential

#     τr_ampa = 0.2
#     τd_ampa = 1.8
#     τr_nmda = 1.0
#     τd_nmda = 100.0

#     τr_gabaA = 0.2
#     τd_gabaA = 5.0

#     gsyn_gabaB = 0.025
#     τr_gabaB = 25.0 #198947
#     τd_gabaB = 500.0 #990036

#     AMPA = Receptor(E_rev = E_exc, τr = τr_ampa, τd = τd_ampa, g0 = gsyn_ampa)
#     NMDA = ReceptorVoltage(E_rev = E_exc, τr = τr_nmda, τd = τd_nmda, g0 = gsyn_nmda)
#     GABAa = Receptor(E_rev = E_gabaA, τr = τr_gabaA, τd = τd_gabaA, g0 = gsyn_gabaA)
#     GABAb = Receptor(E_rev = E_gabaB, τr = τr_gabaB, τd = τd_gabaB, g0 = gsyn_gabaB)
#     Synapse(AMPA, NMDA, GABAa, GABAb)
# end