"""
    SpikeEncoding

Abstract type of the auditory spike encodings of Spike TIMIT: [`BAE`](@ref) and
[`Cochlea`](@ref). Use [`encoding`](@ref) to build one from the names of the old API.
"""
SpikeEncoding

abstract type SpikeEncoding end

"""
    BAE()

BAE threshold encoding (Pan et al. 2020): 620 channels (20 frequency bands x 31 threshold
neurons, band-major), stored in `<sentence>_spike_timing.mat`. Reading it requires MAT.jl
(`using MAT`). Raw times are multiplied by `SpikeTimitConfig().correction` (5) to obtain
seconds.

See also [`Cochlea`](@ref), [`read_spiketimes`](@ref).
"""
BAE

struct BAE <: SpikeEncoding end

"""
    Cochlea(channels::Int)

Cochlea-model encoding with `channels` (35 or 70) channels, stored in
`<sentence>_cochlea<channels>.npz` (array `arr_0`, 2 x K: spike times, 0-based neuron ids).
Times are read as seconds without rescaling. Throws an `ArgumentError` for other channel counts.

See also [`BAE`](@ref), [`read_spiketimes`](@ref).
"""
Cochlea

struct Cochlea <: SpikeEncoding
    channels::Int
    function Cochlea(channels::Integer)
        channels in (35, 70) || throw(ArgumentError("Encoding must be 35 or 70"))
        return new(channels)
    end
end

"""
    encoding(name::AbstractString) -> SpikeEncoding
    encoding(e::SpikeEncoding) -> SpikeEncoding

Return the encoding named `name` (old string API): `"bae"` -> `BAE()`, `"cochlea35"` ->
`Cochlea(35)`, `"cochlea70"` -> `Cochlea(70)`. A `SpikeEncoding` is returned unchanged.
Throws an `ArgumentError` for `"cochlea"` without channels and for unknown names.

# Examples
```jldoctest
julia> using SNNUtils.SpikeTimit

julia> encoding("cochlea70")
Cochlea(70)

julia> encoding("bae")
BAE()
```
"""
function encoding(name::AbstractString)
    name == "bae" && return BAE()
    if startswith(name, "cochlea")
        n = chopprefix(name, "cochlea")
        isempty(n) && throw(ArgumentError("Define cochlea encoding: 35 or 70"))
        channels = tryparse(Int, n)
        channels === nothing && throw(ArgumentError("Encoding must be 'cochlea(35|70)' or 'bae'"))
        return Cochlea(channels)
    end
    throw(ArgumentError("Encoding must be 'cochlea(35|70)' or 'bae'"))
end
encoding(e::SpikeEncoding) = e

"""
    spike_file(e::SpikeEncoding, path::AbstractString) -> String

Return the spike file of the sentence `path` (base path without extension) for encoding `e`.
"""
spike_file(::BAE, path::AbstractString) = path * "_spike_timing.mat"
spike_file(e::Cochlea, path::AbstractString) = path * "_cochlea$(e.channels).npz"

"""
    n_channels(e::SpikeEncoding, config::SpikeTimitConfig) -> Int

Return the number of input channels of encoding `e`.
"""
n_channels(::BAE, config::SpikeTimitConfig) = config.bae_frequencies * config.bae_bins
n_channels(e::Cochlea, ::SpikeTimitConfig) = e.channels
