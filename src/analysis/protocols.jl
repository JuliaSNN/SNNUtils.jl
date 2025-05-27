function get_poisson_spikes(; rate::Float32, dt::Float32 = 0.1f0)
    ### rate in KHz
    if rate * dt > rand()
        return 1.0f0
    else
        return 0.0f0
    end
    # return rand(Poisson(1. *abs(rate*dt)))
end


# s=zeros(1000)
# for m in 1:1000
# 	a= zeros(10000)
# 	for x in 1:10000
# 		a[x] = get_poisson_spikes(rate=1.,dt=0.1)
# 	end
# 	s[m] = sum(a)
# end

function get_EPSP(
    v::Array{Float32,2};
    spiketime = -1,
    rest = 0,
    inh = false,
    compartment = 1,
)
    spiketime = spiketime < 0 ? EXCSPIKETIME : spiketime
    if inh == false
        return maximum(v[compartment, spiketime:end]) - rest
    else
        return minimum(v[compartment, spiketime:end]) - rest
    end
end

function _PoissonInput(Hz_rate::Real, interval::Int64, dt::Float32)
    λ = 1000 / Hz_rate
    spikes = falses(round(Int, interval / dt))
    t = 1
    while t < interval / dt
        Δ = rand(Exponential(λ / dt))
        t += Δ
        if t < interval / dt
            spikes[round(Int, t)] = true
        end
    end
    return spikes
end

function PoissonInput(Hz_rate::Real, interval::Int64, dt::Float32; neurons::Int64 = 1)
    spikes = falses(neurons, round(Int, interval / dt))
    for n = 1:neurons
        spikes[n, :] .= _PoissonInput(Hz_rate::Real, interval::Int64, dt::Float32)
    end
    return spikes
end

logrange(x1, x2, n) = [10^y for y in range(log10(x1), log10(x2), length = n)]

export get_poisson_spikes, logrange, PoissonInput