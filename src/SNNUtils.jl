"""
    SNNUtils

Protocols and analysis tools built on SNNModels:

- word/phoneme sequences and the Poisson stimuli that present them
  ([`get_lexicon`](@ref), [`generate_sequence`](@ref), [`word_phonemes_sequence`](@ref),
  [`step_input`](@ref), [`update_stimuli!`](@ref), ...);
- excitation/inhibition balance of dendritic neurons ([`compute_kei`](@ref));
- import of BioSeq tasks ([`import_bioseq_tasks`](@ref), [`seq_bioseq`](@ref));
- short-term-plasticity parameters sampled from experimental fits
  ([`sample_stp_params`](@ref), [`sample_stp_campagnola`](@ref));
- weight and decoding analysis ([`average_weight_dynamics`](@ref), [`SVCtrain`](@ref),
  [`score_spikes`](@ref), [`trial_average`](@ref), ...).

The parameter collections in `src/models/` other than `stp_het.jl` are not loaded by the
package; see the "SNNUtils models" page of the documentation.
"""
module SNNUtils

using SNNModels
@load_units
using DrWatson
using Parameters
using Random
using Distributions
using Printf
using Serialization
using BSON
using JSON
using ThreadTools
using StatsBase
using Statistics

DATA_PATH = joinpath(@__DIR__, "..", "data") 
@assert isdir(DATA_PATH) "Data directory not found at $DATA_PATH. Please ensure the data directory exists and contains the necessary files."

## Functions to generate sequences of words and phonemes
include("stimuli/sequence/stimuli.jl")
include("stimuli/sequence/sequence.jl")
include("stimuli/sequence/sequence_generators.jl")
include("stimuli/sequence/inputs.jl")

## Functions to compute the Excitatory-Inhibitory balance and analyse the CompartmentNeuron activity
include("stimuli/balance_EI/compute_kei.jl")
include("stimuli/balance_EI/bimodal_kernel.jl")

## Integration with BioSeq framework
include("stimuli/bioseq/import_bioseq.jl")

## Collection of parameters, not included in this version. Move it to SpikingNeuralNetworks.jl
include("models/stp_het.jl")

## Functions to analyse the network structure
include("analysis/weights.jl")

## Functions to run machine learning analysis on the network activity
using MLJ
include("analysis/classifiers.jl")


end
