using DrWatson
using Plots
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils

##
spiketime = [1000ms]
neurons = [[1]]
inputs = SpikeTimeStimulusParameter(neurons=neurons, spiketimes=spiketime)

IF_pv = SNN.IF(; N = 1, param = LKD2014.PV)
IF_adex = SNN.AdEx(; N = 1, param = LKD2014.AdEx)

plots = map([IF_pv, IF_adex]) do IF
    p1 = SNN.plot()
    SNN.monitor([IF], [:fire, :v])
    stim = SpikeTimeStimulus(1, IF, :ge, p=1.f0, param=inputs)
    sim!([IF],[SNN.EmptySynapse()], [stim], duration=1200ms, dt=0.1ms)
    label = "AMPA: $(round(IF.param.gsyn_e, digits=2))"
    p1 = SNN.vecplot!(p1, IF, :v, r=800:0.001:1200ms, neurons=[1], dt=0.1ms,    label= label)

    SNN.monitor([IF], [:fire, :v])
    stim = SpikeTimeStimulus(1, IF, :gi, p=1.f0, param=inputs)
    sim!([IF],[SNN.EmptySynapse()], [stim], duration=1200ms, dt=0.1ms)
    label = "GABA: $(round(IF.param.gsyn_i, digits=2))"
    p1 = SNN.vecplot!(p1, IF, :v, r=800:0.001:1200ms, neurons=[1], dt=0.1ms, label= label)
    SNN.plot!(p1, ylims=:auto)
end

#
SNN.plot(plots..., legend=true,)
###