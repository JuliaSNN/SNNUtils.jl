# Spike post-processing. Every function accepts one sample (spike trains, one vector per channel)
# or a batch (a vector of samples). Times are in seconds.

const _Sample = AbstractVector{<:AbstractVector{<:Real}}
const _Batch = AbstractVector{<:AbstractVector{<:AbstractVector{<:Real}}}

"""
    resample_spikes!(st; n_feat::Int = 8, config = SpikeTimitConfig())
    resample_spikes!(batch; n_feat::Int = 8, config = SpikeTimitConfig()) -> batch

Merge the threshold neurons of each frequency band of a BAE encoding (BAE only).

Within each of the `F = config.bae_frequencies` bands, the `B = config.bae_bins` threshold
channels are grouped by `n_feat`: channel `b` of band `f` goes to output channel
`ceil(Int, b / n_feat) + (f - 1) * nb`, with `nb = ceil(Int, B / n_feat)` channels per band
(620 -> 80 channels for `n_feat = 8`). Spike times (s) of a group are concatenated in channel
order, not sorted.

# Arguments
- `st`: spike trains of one sample with `F * B` (620) channels.
- `batch`: vector of samples; each element is replaced by its resampled version.
- `n_feat::Int = 8`: channels merged per group, in `1:11`.
- `config::SpikeTimitConfig`: BAE band geometry.

# Returns
- One sample: a NEW `SpikeTrains{Float64}` with `F * nb` channels; `st` itself if `n_feat == 1`.
- Batch: the mutated `batch`.

Throws an `ArgumentError` if `n_feat` is not in `1:11` or `length(st) != F * B`.

# Examples
```julia
using SNNUtils.SpikeTimit, MAT
spikes = select_words(df, ["that"]; encoding = BAE()).spikes
resample_spikes!(spikes; n_feat = 8)      # every sample now has 80 channels
```

See also [`transform_into_bursts!`](@ref).
"""
function resample_spikes!(st::_Sample; n_feat::Int = 8,
                          config::SpikeTimitConfig = SpikeTimitConfig())
    n_feat in 1:11 || throw(ArgumentError("Impossible resampling values"))
    n_feat == 1 && return st
    F, B = config.bae_frequencies, config.bae_bins
    length(st) == F * B || throw(ArgumentError(
        "resample_spikes!: expected $(F * B) BAE channels ($(F) bands x $(B) bins), got $(length(st))"))
    nb = ceil(Int, B / n_feat)
    out = [Float64[] for _ in 1:(nb * F)]
    for freq in 1:F, b in 1:B
        append!(out[ceil(Int, b / n_feat) + (freq - 1) * nb], st[b + (freq - 1) * B])
    end
    return out
end

function resample_spikes!(batch::_Batch; n_feat::Int = 8,
                          config::SpikeTimitConfig = SpikeTimitConfig())
    for i in eachindex(batch)
        batch[i] = resample_spikes!(batch[i]; n_feat, config)
    end
    return batch
end

"""
    transform_into_bursts!(st; spikes_per_burst_increase::Int = 0, rng = Random.default_rng()) -> st
    transform_into_bursts!(batch; spikes_per_burst_increase::Int = 0, rng = Random.default_rng()) -> batch

Turn every spike into a burst, in place.

For each original spike at `t` (s) a burst size `v` is drawn from `[2, 3, 4, 5, 6] .+
spikes_per_burst_increase` with probabilities proportional to `[0.8, 0.15, 0.075, 0.035, 0.03]`
(Oswald, Doiron & Maler 2007, Fig. 1B, 0.7 nA), and `v - 1` spikes are added at
`t + 0.004 + 0.005 * randexp(rng)` (s: 4 ms plus an exponential delay of mean 5 ms). Each
non-empty channel is then sorted; empty channels are untouched.

# Arguments
- `st`: spike trains (s) of one sample; `batch`: vector of samples (processed in order with the
  same `rng`).
- `spikes_per_burst_increase::Int = 0`: added to every burst size; must be `>= -1`.
- `rng::AbstractRNG`: random number generator.

Throws an `ArgumentError` if `spikes_per_burst_increase < -1`.

# Examples
```julia
using SNNUtils.SpikeTimit, Random
st = [[0.1, 0.3], Float64[]]
transform_into_bursts!(st; rng = MersenneTwister(1))
```

See also [`resample_spikes!`](@ref).
"""
function transform_into_bursts!(st::_Sample; spikes_per_burst_increase::Int = 0,
                                rng::AbstractRNG = Random.default_rng())
    spikes_per_burst_increase < -1 && throw(ArgumentError(
        "spikes_per_burst_increase must be >= -1, got $(spikes_per_burst_increase)"))
    values = [2, 3, 4, 5, 6] .+ spikes_per_burst_increase
    w = [0.8, 0.15, 0.075, 0.035, 0.03]
    weights = StatsBase.Weights(w ./ sum(w))
    for channel in st
        isempty(channel) && continue
        for t in copy(channel)
            v = StatsBase.sample(rng, values, weights)
            for _ in 1:(v - 1)
                push!(channel, t + 0.005 * randexp(rng) + 0.004)
            end
        end
        sort!(channel)
    end
    return st
end

function transform_into_bursts!(batch::_Batch; spikes_per_burst_increase::Int = 0,
                                rng::AbstractRNG = Random.default_rng())
    for st in batch
        transform_into_bursts!(st; spikes_per_burst_increase, rng)
    end
    return batch
end

function _check_channels(frequencies, n::Int)
    all(f -> 1 <= f <= n, frequencies) || throw(ArgumentError(
        "channel indices must be in 1:$(n), got $(collect(frequencies))"))
    return nothing
end

"""
    remove_frequencies!(st; frequencies) -> st
    remove_frequencies!(batch; frequencies) -> batch

Empty the channels `frequencies` (vector or range of channel indices), in place: each removed
channel is replaced by a new empty vector (the old vector is not modified). For a batch, the
same channels are emptied in every sample. Throws an `ArgumentError` for an index outside the
channels.

# Examples
```julia
using SNNUtils.SpikeTimit
st = [[0.1], [0.2], [0.3]]
remove_frequencies!(st; frequencies = [1, 3])   # [Float64[], [0.2], Float64[]]
```

See also [`select_frequencies!`](@ref).
"""
function remove_frequencies!(st::_Sample;
                             frequencies::Union{AbstractVector{<:Integer},AbstractRange})
    _check_channels(frequencies, length(st))
    for f in frequencies
        st[f] = empty(st[f])
    end
    return st
end

function remove_frequencies!(batch::_Batch;
                             frequencies::Union{AbstractVector{<:Integer},AbstractRange})
    for st in batch
        remove_frequencies!(st; frequencies)
    end
    return batch
end

"""
    select_frequencies!(st; frequencies) -> st
    select_frequencies!(batch; frequencies) -> batch

Keep only the channels `frequencies` and empty all the others, in place (see
[`remove_frequencies!`](@ref)). For a batch the complement is taken over the channels of the
first sample. Throws an `ArgumentError` for an index outside the channels.

# Examples
```julia
using SNNUtils.SpikeTimit
st = [[0.1], [0.2], [0.3]]
select_frequencies!(st; frequencies = 2:2)   # [Float64[], [0.2], Float64[]]
```
"""
function select_frequencies!(st::_Sample;
                             frequencies::Union{AbstractVector{<:Integer},AbstractRange})
    _check_channels(frequencies, length(st))
    return remove_frequencies!(st; frequencies = setdiff(1:length(st), frequencies))
end

function select_frequencies!(batch::_Batch;
                             frequencies::Union{AbstractVector{<:Integer},AbstractRange})
    isempty(batch) && return batch
    n = length(batch[1])
    _check_channels(frequencies, n)
    return remove_frequencies!(batch; frequencies = setdiff(1:n, frequencies))
end
