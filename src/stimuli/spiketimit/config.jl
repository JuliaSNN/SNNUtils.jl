"""
    SpikeTimitConfig(; sr = 16000.0, correction = 5.0, bae_frequencies = 20, bae_bins = 31)

Constants of the Spike TIMIT corpus, passed explicitly to the functions that need them (they
replace the global variables of the old SpikeTimit.jl package). Immutable and `isbits`.

# Fields
- `sr::Float64 = 16000.0`: sample rate (Hz) of the sample indices in the `.wrd`/`.phn` files;
  transcription times (s) are `sample / sr`.
- `correction::Float64 = 5.0`: dimensionless factor applied to the raw spike times of the BAE
  `.mat` files to obtain seconds (`raw * correction`).
- `bae_frequencies::Int = 20`: frequency bands of the BAE encoding.
- `bae_bins::Int = 31`: threshold neurons per band; the BAE encoding has
  `bae_frequencies * bae_bins` (620) channels, band-major.

Throws an `ArgumentError` unless `sr > 0`, `correction > 0`, `bae_frequencies >= 1` and
`bae_bins >= 1`.

# Examples
```jldoctest
julia> using SNNUtils.SpikeTimit

julia> SpikeTimitConfig(sr = 8000).sr
8000.0
```
"""
SpikeTimitConfig

Base.@kwdef struct SpikeTimitConfig
    sr::Float64 = 16000.0
    correction::Float64 = 5.0
    bae_frequencies::Int = 20
    bae_bins::Int = 31
    function SpikeTimitConfig(sr, correction, bae_frequencies, bae_bins)
        sr > 0 || throw(ArgumentError("sr (Hz) must be positive, got $(sr)"))
        correction > 0 || throw(ArgumentError("correction must be positive, got $(correction)"))
        bae_frequencies >= 1 ||
            throw(ArgumentError("bae_frequencies must be >= 1, got $(bae_frequencies)"))
        bae_bins >= 1 || throw(ArgumentError("bae_bins must be >= 1, got $(bae_bins)"))
        return new(sr, correction, bae_frequencies, bae_bins)
    end
end

"""Milliseconds per second: converts SpikeTimit times (s) to SNNModels times (ms)."""
const S_TO_MS = 1000.0

const _SPIKETIMIT_UNKNOWN = "Spike TIMIT location unknown: pass `root`/`dir` explicitly or set " *
                            "ENV[\"SPIKETIMIT_PATH\"] to the directory that contains train/, test/ and DOC/"

"""
    dataset_path(parts::AbstractString...; root = nothing) -> String

Return the absolute, normalised path `joinpath(root, parts...)` inside the Spike TIMIT corpus.

`root` defaults to `ENV["SPIKETIMIT_PATH"]`. Give an absolute path: `~` is not expanded. The root
must be an existing directory; the joined path is not checked. The `parts` must be relative and
must not contain `..`, so that the result stays inside the root.

# Arguments
- `parts::AbstractString...`: path components below the root (e.g. `"train"`, `"DOC"`).
- `root::Union{Nothing,AbstractString} = nothing`: corpus root; `nothing` reads the environment
  variable `SPIKETIMIT_PATH`.

# Returns
- `String`: the normalised absolute path.

Throws an `ArgumentError` if the root is unknown (no `root` and `SPIKETIMIT_PATH` unset or
empty), is not a directory, or if a part is absolute or contains `..`.

# Examples
```julia
using SNNUtils.SpikeTimit
ENV["SPIKETIMIT_PATH"] = "/data/Spike TIMIT"
dataset_path("train")                 # "/data/Spike TIMIT/train"
dataset_path("DOC", "TIMITDIC.TXT")
```

See also [`create_dataset`](@ref), [`create_dictionary`](@ref).
"""
function dataset_path(parts::AbstractString...; root::Union{Nothing,AbstractString} = nothing)
    if root === nothing
        root = get(ENV, "SPIKETIMIT_PATH", "")
        isempty(root) && throw(ArgumentError(_SPIKETIMIT_UNKNOWN))
    end
    isdir(root) ||
        throw(ArgumentError("Spike TIMIT root \"$(root)\" is not an existing directory"))
    for p in parts
        isabspath(p) && throw(ArgumentError("dataset_path: part \"$(p)\" must be a relative path"))
        ".." in splitpath(p) &&
            throw(ArgumentError("dataset_path: part \"$(p)\" must not contain \"..\""))
    end
    return normpath(joinpath(abspath(root), parts...))
end
