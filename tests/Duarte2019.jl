
## Test model
spiketime = [1000ms]
neurons = [[1]]
inputs = SpikeTimeStimulusParameter(neurons=neurons, spiketimes=spiketime)

IF_pv = SNN.IF(; N = 1, param = duarte2019.PV)
IF_sst = SNN.IF(; N = 1, param = duarte2019.SST)
IF_adex = SNN.AdEx(; N = 1, param = duarte2019.AdEx)
plots = map([IF_pv, IF_sst, IF_adex]) do IF
    p1 = SNN.plot()
    SNN.monitor([IF], [:fire, :v])
    IF.I .= 1000pA
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

SNN.plot(plots..., legend=true, layout=(3,1), size=(500, 600))
