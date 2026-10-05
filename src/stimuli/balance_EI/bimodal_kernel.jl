using Statistics
using StatsBase
# Find bimodal value
# Here I use a simple algorithm that is described in :
# Journal of the Royal Statistical Society. Series B (Methodological)
# Using Kernel Density Estimates to Investigate Multimodality
# https://www.jstor.org/stable/2985156

# It consists in  using Normal kernels to approximate the data and then leverages a theorem on decreasing monotonicity of the number of maxima as function of the window span.

# Kernel Density Estimation
function KDE(t::Real, h::Int64, data)
    ndf(x, h) = exp(-x^2 / h)
    1 / length(data) * 1 / h * sum(ndf.(data .- t, h))
end

# Distribution
function globalKDE(h::Int64, data; v_range = collect(-90:-35))
    kde = zeros(Float64, length(v_range))
    @fastmath @inbounds for n = 1:length(v_range)
        kde[n] = KDE(v_range[n], h, data)
    end
    return kde
end

#Get its maxima
function get_maxima(data)
    arg_maxima = []
    for x = 2:(length(data)-1)
        (data[x] > data[x-1]) && (data[x] > data[x+1]) && (push!(arg_maxima, x))
    end
    @debug "Maxima: $(length(arg_maxima))"
    return arg_maxima
end

#Trash spurious values (below 30% of the true maximum)
function isbimodal(kernel, ratio)
    maxima = get_maxima(kernel)
    z = maximum(kernel[maxima])
    real = []
    @debug length(maxima)
    for n in maxima
        m = kernel[n]
        @debug "Maxima: $m, z: $z"
        if (abs(m / z) > ratio)
            push!(real, m)
        end
    end
    if length(real) > 1
        return true
    else
        return false
    end
end

#Trash spurious values (below 30% of the true maximum)
function count_maxima(kernel, ratio)
    maxima = get_maxima(kernel)
    z = maximum(kernel[maxima])
    real_maxima = []
    for n in maxima
        m = kernel[n]
        if (abs(m / z) > ratio)
            push!(real_maxima, m)
        end
    end
    return length(real_maxima)
end

# Return the critical window (hence the bimodal factor)
"""
    critical_window(data; ratio = 0.1, max_b = 50, v_range = collect(-90:-35))

Bimodality index of the samples `data` (typically membrane potentials in mV), following the
kernel-density test of multimodality of Silverman (1981, J. R. Stat. Soc. B, cited in the source
as "Using Kernel Density Estimates to Investigate Multimodality").

For bandwidths `h = 1, 3, 5, ..., max_b` the density is estimated on `v_range` with the kernel
``exp(-x^2 / h) / h`` (`globalKDE`); a maximum counts if it is larger than `ratio` times the
highest maximum. Returns the first `h` at which the estimate is no longer bimodal (larger values
mean more separated modes), or `max_b`. Not exported (`SNNUtils.critical_window`).
"""
function critical_window(data; ratio = 0.1, max_b = 50, v_range = collect(-90:-35))
    for h = 1:2:max_b
        kernel = globalKDE(h, data, v_range = v_range)
        bimodal = false
        try
            bimodal = isbimodal(kernel, ratio)
        catch
            bimodal = false
            @error "Bimodal failed"
        end
        if !bimodal
            return h
        end
    end
    return max_b
end

"""
    all_windows(data, ratio = 0.3; max_b = 50)

Number of significant maxima (above `ratio` times the highest one) of the kernel density estimate
of `data` on `-90:-35` for every bandwidth `h = 1:max_b`. Returns a vector of length `max_b`.
Not exported.
"""
function all_windows(data, ratio = 0.3; max_b = 50)
    counter = zeros(max_b)
    for h = 1:max_b
        kernel = globalKDE(h, data)
        counter[h] = count_maxima(kernel, ratio)
    end
    return counter
end
