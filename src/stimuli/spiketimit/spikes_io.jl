"""
    _matread_variable(file::AbstractString, name::AbstractString)

Return the variable `name` of the MATLAB file `file`. No method is defined here: the method is
added by the package extension `SNNUtilsMATExt`, loaded with `using MAT`.
"""
function _matread_variable end

"""Error message raised when a BAE file is read without MAT.jl."""
const MAT_EXT_MESSAGE = "Reading BAE spike files (*_spike_timing.mat) requires MAT.jl. Run `using MAT` " *
                        "(install it with `import Pkg; Pkg.add(\"MAT\")`) to load the SNNUtilsMATExt " *
                        "extension, then call again."

"""
    _require_mat_ext(ext = Base.get_extension(SNNUtils, :SNNUtilsMATExt)) -> Module

Return the MAT extension module, or throw `ErrorException(MAT_EXT_MESSAGE)` if it is not loaded.
"""
function _require_mat_ext(ext = Base.get_extension(parentmodule(@__MODULE__), :SNNUtilsMATExt))
    ext === nothing && throw(ErrorException(MAT_EXT_MESSAGE))
    return ext
end

"""
    _bae_cells(raw, n::Int, file = "") -> Vector

Check that `raw` (the `spike_output` variable) is a 1 x `n` cell array and return its cells.
"""
function _bae_cells(raw, n::Int, file::AbstractString = "")
    (raw isa AbstractMatrix && size(raw, 1) == 1 && size(raw, 2) == n) ||
        throw(ArgumentError("$(file): expected a 1 x $n cell array in spike_output, got " *
                            (raw isa AbstractArray ? "a $(join(size(raw), " x ")) $(typeof(raw))" :
                             "a $(typeof(raw))")))
    return vec(raw)
end

"""
    _normalize_bae_cells(cells::AbstractVector, correction::Float64) -> Vector{Vector{Float64}}

Convert the cells of a BAE `spike_output` cell array into spike trains (s): a number `x` gives
`[x * correction]`, a numeric array (k x 1, 1 x k or 0 x 0) gives `vec(x) .* correction`. Throws
an `ArgumentError` for any other cell content.
"""
function _normalize_bae_cells(cells::AbstractVector, correction::Float64)
    out = Vector{Vector{Float64}}(undef, length(cells))
    for (i, x) in enumerate(cells)
        if x isa Real
            out[i] = [Float64(x) * correction]
        elseif x isa AbstractArray{<:Real}
            out[i] = vec(Float64.(x)) .* correction
        else
            throw(ArgumentError("cell $i of spike_output is a $(typeof(x))"))
        end
    end
    return out
end

"""
    read_spiketimes(path::AbstractString, enc; config = SpikeTimitConfig()) -> SpikeTrains{Float64}

Read the spike trains (s) of one sentence: `st[c]` are the spike times of channel `c`.

# Arguments
- `path::AbstractString`: sentence base path without suffix (the `path` column of
  [`create_dataset`](@ref)).
- `enc`: a [`SpikeEncoding`](@ref) or its name (`"bae"`, `"cochlea35"`, `"cochlea70"`).
- `config::SpikeTimitConfig`: `correction` and the number of BAE channels.

# Returns
- `SpikeTrains{Float64}`: one vector of spike times (s) per channel; 620 for [`BAE`](@ref),
  35 or 70 for [`Cochlea`](@ref).

[`Cochlea`](@ref): `path * "_cochlea<n>.npz"`, array `arr_0` (2 x K: time, 0-based neuron id).
Times are taken as seconds without rescaling (assumption, see the docs); within a channel the
file order is kept. [`BAE`](@ref): `path * "_spike_timing.mat"`, cell array `spike_output`
(1 x 620); raw times are multiplied by `config.correction`. Reading BAE files requires
`using MAT` (an `ErrorException` explains how to load it otherwise).

Throws an `ArgumentError` naming the file for malformed contents (missing variable, wrong
shape, neuron ids out of `0:(n-1)`); a missing file raises a `SystemError`.

# Examples
```julia
using SNNUtils.SpikeTimit, MAT
df = create_dataset(dataset_path("train"))
st = read_spiketimes(df.path[1], BAE())         # 620 channels, s
st = read_spiketimes(df.path[1], "cochlea70")   # 70 channels, s
```

See also [`get_spiketimes`](@ref), [`to_snn_spiketimes`](@ref).
"""
read_spiketimes(path::AbstractString, enc::AbstractString;
                config::SpikeTimitConfig = SpikeTimitConfig()) =
    read_spiketimes(path, encoding(enc); config)

function read_spiketimes(path::AbstractString, enc::Cochlea;
                         config::SpikeTimitConfig = SpikeTimitConfig())
    file = spike_file(enc, path)
    d = npzread(file)
    (d isa AbstractDict && haskey(d, "arr_0")) ||
        throw(ArgumentError("$(file): expected an .npz archive with the array arr_0"))
    A = d["arr_0"]
    (A isa AbstractMatrix{<:Real} && size(A, 1) == 2) ||
        throw(ArgumentError("$(file): arr_0 must be a 2 x K numeric matrix, got $(summary(A))"))
    n = n_channels(enc, config)
    out = [Float64[] for _ in 1:n]
    for (j, col) in enumerate(eachcol(A))
        t, k = col[1], col[2]
        (isinteger(k) && 0 <= k <= n - 1) || throw(ArgumentError(
            "$(file): column $j has neuron id $k, expected an integer in 0:$(n - 1)"))
        push!(out[Int(k)+1], Float64(t))
    end
    return out
end

function read_spiketimes(path::AbstractString, enc::BAE;
                         config::SpikeTimitConfig = SpikeTimitConfig())
    _require_mat_ext()
    file = spike_file(enc, path)
    raw = _matread_variable(file, "spike_output")
    cells = _bae_cells(raw, n_channels(enc, config), file)
    return _normalize_bae_cells(cells, config.correction)
end

"""
    get_spiketimes(df::AbstractDataFrame, enc; config = SpikeTimitConfig()) -> Vector{SpikeTrains{Float64}}
    get_spiketimes(; df, encoding, config = SpikeTimitConfig())

Read the spike trains (s) of every row of `df` (row order) with [`read_spiketimes`](@ref).
`enc`/`encoding` is a [`SpikeEncoding`](@ref) or its name (`"bae"`, `"cochlea35"`,
`"cochlea70"`).

# Examples
```julia
using SNNUtils.SpikeTimit
df = create_dataset(dataset_path("train"))
spikes = get_spiketimes(find_word(df, "that"), Cochlea(70))
```
"""
function get_spiketimes(df::AbstractDataFrame, enc::Union{SpikeEncoding,AbstractString};
                        config::SpikeTimitConfig = SpikeTimitConfig())
    e = encoding(enc)
    return SpikeTrains{Float64}[read_spiketimes(p, e; config) for p in df.path]
end

get_spiketimes(; df::AbstractDataFrame, encoding, config::SpikeTimitConfig = SpikeTimitConfig()) =
    get_spiketimes(df, encoding; config)
