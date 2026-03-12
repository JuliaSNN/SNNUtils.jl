using DataFrames, DataFramesMeta, CSV

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