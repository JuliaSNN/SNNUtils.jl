

## Set timestep simulation
# @consts begin
#     # dt = 1.0f-1
#     # HUMAN = Physiology(200Ω * cm, 38907Ω * cm^2, 0.5μF / cm^2) # CHANGED
#     # MOUSE = Physiology(200Ω * cm, 1700Ω * cm^2, 1μF / cm^2) # CHANGED
# end


itr = walkdir(joinpath(@__DIR__))
(_root, _dirs, _files) = first(itr)
for _file in _files
    file = joinpath(_root, _file)
    (file !== @__FILE__) && include(file)
end

@info "Models loaded"
