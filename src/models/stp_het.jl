using DataFrames, DataFramesMeta, CSV

# Table of short-term plasticity fits loaded at package load time from
# `data/recordings/dsxc_mouse_fit_results.csv`. Rows are kept if the fitted facilitation and
# recovery time constants are below 10 s, the fit MSE is below its 75th percentile and
# 0.2 < U < 0.8; time constants are converted from s to ms. The subsets `exc_exc_stp`,
# `exc_inh_stp`, `inh_exc_stp`, `inh_inh_stp` select the pre (`synapse_type`) / post
# (`post_cell.cell_class`) excitatory/inhibitory classes.

stp_data = CSV.read(joinpath(DATA_PATH,"recordings", "dsxc_mouse_fit_results.csv"), DataFrame)
@rtransform! stp_data :post_cell_class= $(Symbol("post_cell.cell_class"))
@rsubset! stp_data :fit_tau_fac .< 10
@rsubset! stp_data :fit_tau_rec .< 10
@rtransform! stp_data :fit_tau_fac = :fit_tau_fac .* 1000
@rtransform! stp_data :fit_tau_rec = :fit_tau_rec .* 1000
@rsubset! stp_data :fit_mse < quantile(stp_data.fit_mse, 0.75)
@rsubset! stp_data :fit_U .> 0.2 && :fit_U .< 0.8


dropmissing!(stp_data, :post_cell_class)
const inh_inh_stp = @rsubset stp_data String(:synapse_type) == "in" && String(:post_cell_class) == "in"
const exc_inh_stp = @rsubset stp_data String(:synapse_type) == "ex" && String(:post_cell_class) == "in"
const exc_exc_stp = @rsubset stp_data String(:synapse_type) == "ex" && String(:post_cell_class) == "ex"
const inh_exc_stp = @rsubset stp_data String(:synapse_type) == "in" && String(:post_cell_class) == "ex"

"""
    sample_stp_params(N; df = stp_data, weights = false)

Draw `N` sets of Tsodyks-Markram short-term plasticity parameters from the table `df` of fitted
synapses (default: the whole filtered table `SNNUtils.stp_data`).

For each sample, every parameter is taken from an independently drawn random row (parameters are
therefore not jointly sampled from the same synapse).

# Returns
`(τF, τD, U)` (and `w` if `weights = true`): vectors of length `N` with the facilitation time
constant (ms, column `fit_tau_fac`), the depression/recovery time constant (ms, `fit_tau_rec`),
the utilisation `U` (`fit_U`) and the fitted amplitude (`fit_w`, units of the source table).
The vectors are `Float64`; `τD`, `τF`, `U` correspond to the (`Float32` vector) fields of
SNNModels' `MarkramSTPParameterHet`.

# Example
```julia
using SNNUtils
p = sample_stp_params(10)
p.τD
```
"""
function sample_stp_params(N; df = stp_data,  weights=false)
    d = map(1:N) do n
        [df[!,x][rand(1:nrow(df))] for x in  [:fit_tau_rec, :fit_tau_fac, :fit_U, :fit_w]]
    end  |> x->reduce(hcat, x)
    if weights
        return (;τF = d[2,:], τD = d[1,:], U = d[3,:], w = d[4,:])
    else
        return (;τF = d[2,:], τD = d[1,:], U = d[3,:])
    end
end

"""
    sample_stp_campagnola(N, type; df = stp_data, weights = true)

Like [`sample_stp_params`](@ref) but restricted to one connection class:
`type` is one of `:exc_exc`, `:exc_inh`, `:inh_exc`, `:inh_inh` (presynaptic class, postsynaptic
class). Returns `(τF, τD, U, w)` by default (`weights = true`). The argument `df` is ignored.

The name refers to Campagnola et al. (2022, Science), the Allen Institute synaptic-physiology
survey of mouse and human neocortex, which is the likely source of the bundled fit table; the
source file does not state the reference.

# Example
```julia
using SNNUtils
p = sample_stp_campagnola(5, :exc_inh)
```
"""
function  sample_stp_campagnola(N, type; df = stp_data, weights=true) 
    if type == :exc_exc
        return sample_stp_params(N; df = exc_exc_stp, weights=weights)
    elseif type == :exc_inh
        return sample_stp_params(N; df = exc_inh_stp, weights=weights)
    elseif type == :inh_exc
        return sample_stp_params(N; df = inh_exc_stp, weights=weights)
    elseif type == :inh_inh
        return sample_stp_params(N; df = inh_inh_stp, weights=weights)
    else
        error("Invalid synapse type. Must be one of :exc_exc, :exc_inh, :inh_exc, :inh_inh")
    end
end

export sample_stp_params, sample_stp_campagnola