"""
    SNNUtilsMATExt

Package extension loaded with `using MAT`: lets `SNNUtils.SpikeTimit` read the BAE spike files
(`*_spike_timing.mat`) of Spike TIMIT.
"""
module SNNUtilsMATExt

using SNNUtils, MAT

function SNNUtils.SpikeTimit._matread_variable(file::AbstractString, name::AbstractString)
    vars = MAT.matread(file)
    haskey(vars, name) || throw(ArgumentError("$(file) has no variable \"$(name)\""))
    return vars[name]
end

end
