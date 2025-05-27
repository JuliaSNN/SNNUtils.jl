
#############################################################
###########       Synapse parameters        #################
#############################################################

## Synapses used in Tripod neuron model

"""
Christof Koch. Biophysics of Computation: Information Processing in Single Neurons, by.Trends in Neurosciences, 22842(7):328–329, July 1999. ISSN 0166-2236, 1878-108X. doi: 10.1016/S0166-2236(99)01403-4.
"""
KochGlu =
    Glutamatergic(Receptor(E_rev = 0.00, τr = 0.2, τd = 25.0, g0 = 0.73), ReceptorVoltage(gsyn=-1))
    


"""
Guy Eyal, Matthijs B. Verhoog, Guilherme Testa-Silva, Yair Deitcher, Ruth Benavides-Piccione, Javier DeFelipe, Chris-832tiaan P. J. de Kock, Huibert D. Mansvelder, and Idan Segev. Human Cortical Pyramidal Neurons: From Spines to833Spikes via Models.Frontiers in Cellular Neuroscience, 12, 2018. ISSN 1662-5102. doi: 10.3389/fncel.2018.00181.
"""

EyalGluDend = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73), 
    ReceptorVoltage(E_rev = 0.0, τr = 8, τd = 35.0, g0 = 1.31, nmda = 1.), 
)

EyalGluDend_nonmda = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 0.25, τd = 2.0, g0 = .0),
    ReceptorVoltage(E_rev = 0.0, τr = 8, τd = 35.0, g0 = 0.000f0, nmda = 0.0f0),
)


"""
Richard Miles, Katalin Tóth, Attila I Gulyás, Norbert Hájos, and Tamas F Freund.  Differences between Somatic923and Dendritic Inhibition in the Hippocampus.Neuron, 16(4):815–823, April 1996. ISSN 0896-6273. doi: 10.1016/924S0896-6273(00)80101-4.
"""
MilesGabaDend = GABAergic(
    Receptor(E_rev = -70.0, τr = 4.8, τd = 29.0, g0 = 0.27),
    Receptor(E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.006), 
)

MilesGabaSoma =
    GABAergic(Receptor(E_rev = -75.0, τr = 0.5, τd = 6.0, g0 = 0.265), Receptor(τr=-1))

DuarteGluSoma = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 0.25, τd = 2.0, g0 = 0.73),
    ReceptorVoltage(τr=-1),

)

# EyalNMDA = NMDAVoltageDependency(mg = Mg_mM, b = nmda_b, k = nmda_k)

quaresima2022 = (
        dends  =  [(150um, 400um), (150um, 400um)],
        soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma), # defines glutamaterbic and gabaergic receptors in the soma
        dend_syn = Synapse(EyalGluDend, MilesGabaDend), # defines glutamaterbic and gabaergic receptors in the dendrites
        NMDA = EyalNMDA, # NMDA synapse
        param = AdExSoma(Vr = -55mV, Vt = -50mV),
)

export quaresima2022, KochGlu, EyalGluDend, EyalNMDA, EyalGluDend_nonmda, MilesGabaDend, MilesGabaSoma, DuarteGluSoma