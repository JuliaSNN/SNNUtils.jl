<div align="center">
    <img src="https://github.com/JuliaSNN/SpikingNeuralNetworks.jl/blob/main/docs/src/assets/SNNLogo.svg" alt="SpikingNeuralNetworks.jl" width="200">
</div>

<h2 align="center"> Input protocols, ML tools, and further utils for Julia SpikingNeuralNetworks.jl 
<p align="center">
    <a href="https://github.com/JuliaSNN/SNNModels.jl/actions">
    <img src="https://github.com/JuliaSNN/SNNModels.jl/workflows/CI/badge.svg"
         alt="Build Status">
  </a>
  <a href="https://juliasnn.github.io/SpikingNeuralNetworks.jl/dev/">
    <img src="https://img.shields.io/badge/docs-stable-blue.svg"
         alt="stable documentation">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yelllow"
       alt="bibtex">
  </a>

</p>
</h2>


# SNNUtils

Support package to generate complex stimuli and run ML pipelines for the JuliaSNN ecosystem.

## Utils

The package contains functions for:

- generating sequences of stimuli with a first-order hierarchical structure (word-phonemes);
- computing the Excitatory-Inhibitory balance and analysing the CompartmentNeuron membrane potential;
- integrating the SpikingNeuralNetworks.jl with the BioSeq framework;
- analysing the network connectivity structure;
 running machine learning analysis on the network activity.


If you do not have specific reasons to install this package separately, we strongly advise you to install the user-facing package at:

https://github.com/JuliaSNN/SpikingNeuralNetworks.jl
