using DrWatson
using Plots
using Revise
using SpikingNeuralNetworks
@load_units;
using SNNUtils

##
spiketime = [1000ms]
neurons = [1]
# inputs = SpikeTimeParameter(neurons=neurons, spiketimes=spiketime)
inputs = SpikeTimeParameter(neurons = neurons, spiketimes = spiketime)

IF_pv = IF(; N = 1, param = LKD2014.PV)
IF_adex = AdEx(; N = 1, param = LKD2014.AdEx)

plots = map([IF_pv, IF_adex]) do IF
    p1 = plot()
    monitor!([IF], [:fire, :v, :ge], sr = 400Hz)
    stim = SpikeTimeStimulus(IF, :ge, p = 1.0f0, param = inputs)
    sim!([IF], [EmptySynapse()], [stim], duration = 1200ms, dt = 0.125ms)
    label = "AMPA: $(round(IF.param.gsyn_e, digits=2))"
    p1 = vecplot!(p1, IF, :ge, r = 800:0.1:1200ms, neurons = [1], dt = 0.1ms, label = label)

    monitor!([IF], [:fire, :v, :gi], sr = 400Hz)
    stim = SpikeTimeStimulus(IF, :gi, p = 1.0f0, param = inputs)
    sim!([IF], [EmptySynapse()], [stim], duration = 1200ms, dt = 0.125ms)
    label = "GABA: $(round(IF.param.gsyn_i, digits=2))"
    p1 = vecplot!(p1, IF, :gi, r = 800:0.1:1200ms, neurons = [1], dt = 0.1ms, label = label)
    plot!(p1, ylims = :auto)
end

plot(plots..., legend = true, layout = (2, 1), size = (500, 600))
