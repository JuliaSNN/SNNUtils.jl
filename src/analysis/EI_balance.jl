function kei_balance(kie_measure, target_rate=-55mV, Nd=nothing)
    if typeof(kie_measure) <: Vector
        return map(n->_get_keibalance(kie_measure[n], target_rate), 1:length(kie_measure)) 
    else
        return _get_keibalance(kie_measure, target_rate)
    end
end

function _get_keibalance(kie_measure, target_rate=-55mV)
    @unpack models, νs, kie_test = kie_measure
    voltage_data = haskey(kie_measure, :voltage_data) ? kie_measure.voltage_data : kie_measure.voltage_soma
    
    mins = zeros(Int,size(voltage_data)[2:end])
    @info "Size of kei: $(size(mins))"
    if ndims(voltage_data) == 2
        for n in 1:length(νs)
            m = argmin(abs.(voltage_data[:,n] .- target_rate))
            if abs(voltage_data[m, n] - target_rate) < 2.3mV
                mins[n] = m
            else
                mins[n] = 1
            end
        end
    elseif ndims(voltage_data) == 3
        for n in 1:length(νs)
            for l in 1:length(models)
                m = argmin(abs.(voltage_data[:,n,l] .- target_rate))
                if abs(voltage_data[m, n, l] - target_rate) < 2.3mV
                    mins[n, l] = m
                else
                    mins[n, l] = 1
                end
            end
        end
    end
    (mins=mins, νs=νs, kie_test=kie_test)
end


export kei_balance