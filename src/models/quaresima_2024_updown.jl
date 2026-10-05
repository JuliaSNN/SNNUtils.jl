# Not loaded by SNNUtils. Requires the definitions of quaresima2022.jl. `EyalGluNAR`,
# `EyalEquivalentNAR` and `quaresima2022_nar` work with SNNModels after 1.8.4
# (`quaresima2022_nar` returns keyword arguments of `Tripod`; up to SNNUtils 0.2.9 it used the
# removed type `AdExSoma`, and the undefined `quaresima2022_nonmda` was exported).
NAR0 = 1.31/0.73
EyalGluNAR(NAR = 1.8, τd = 35ms) = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 0.25, τd = 2.0, g0 = 0.73(1+NAR0-NAR)),
    ReceptorVoltage(E_rev = 0.0, τr = 8, τd = τd, g0 = 0.73*NAR, nmda = 1.0f0),
)

EyalEquivalentNAR(NAR, τd = 35) = Receptors(EyalGluNAR(NAR, τd), MilesGabaDend)

# Keyword arguments of `Tripod`: `Tripod(; N = 100, quaresima2022_nar(1.8)...)`.
quaresima2022_nar(nar, τ = 35ms) = (
    param = TripodParameter(ds = [(150um, 400um), (150um, 400um)]),
    soma_syn = ReceptorSynapse(glu_receptors = [1], gaba_receptors = [2],
        syn = Receptors(DuarteGluSoma, MilesGabaSoma), NMDA = EyalNMDA),
    dend_syn = ReceptorSynapse(glu_receptors = [1, 2], gaba_receptors = [3, 4],
        syn = EyalEquivalentNAR(nar, τ), NMDA = EyalNMDA),
    adex = AdExParameter(Vr = -55mV, Vt = -50mV),
)


export EyalEquivalentNAR, quaresima2022_nar
