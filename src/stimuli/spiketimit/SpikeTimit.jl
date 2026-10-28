"""
    SpikeTimit

Reader of the Spike TIMIT corpus: real speech inputs (spike trains with word and phone timing)
for SNNModels networks.

Spike TIMIT is the TIMIT corpus encoded into spike trains by auditory models: the BAE threshold
encoding ([`BAE`](@ref), `.mat`, needs `using MAT`) and cochlea models ([`Cochlea`](@ref),
`.npz`). The corpus is licensed (LDC) and not distributed: give its location with
`ENV["SPIKETIMIT_PATH"]` or explicit paths ([`dataset_path`](@ref)).

Workflow: [`create_dataset`](@ref) -> [`find_word`](@ref) / [`select_words`](@ref) ->
[`resample_spikes!`](@ref), [`transform_into_bursts!`](@ref) -> [`get_ordered_spikes`](@ref) ->
[`spiketime_parameter`](@ref) (input of `SNNModels.SpikeTimeStimulus`) and
[`to_sequence`](@ref) / [`label_intervals`](@ref) (word/phone timing).

# Units
- Data functions use `Float64` SECONDS: transcription times are sample indices divided by
  `SpikeTimitConfig().sr` (16 kHz); spike trains are [`SpikeTrains{Float64}`](@ref SpikeTrains).
- Conversion functions ([`to_snn_spiketimes`](@ref), [`spiketime_parameter`](@ref),
  [`ms_intervals`](@ref), [`label_intervals`](@ref), [`to_sequence`](@ref)) return `Float32`
  MILLISECONDS, the SNNModels convention.
- [`transcriptions_dt`](@ref) and [`ft_dt`](@ref) return integer time steps (`dt` in ms).

# Relation to SNNUtils sequences
SNNUtils generates synthetic word/phoneme sequences (`generate_sequence`) and drives stimuli
locked to their intervals (`sign_intervals`, `update_stimuli!`). [`to_sequence`](@ref) builds a
sequence with the same fields from real TIMIT timing, so those functions work unchanged on
speech; [`label_intervals`](@ref) gives the same `Vector{Vector{Float32}}` interval format.

The names are exported by `SpikeTimit` only: use `using SNNUtils.SpikeTimit`.
"""
module SpikeTimit

using DataFrames
using NPZ
using Random
import StatsBase
import SNNModels

export SpikeTimitConfig, dataset_path
export SpikeEncoding, BAE, Cochlea, encoding
export Phone, Word, Transcription, Transcriptions, SpikeTrains
export create_dataset, create_dictionary, find_word, count_words

include("config.jl")
include("types.jl")
include("encodings.jl")
include("dataset.jl")

end
