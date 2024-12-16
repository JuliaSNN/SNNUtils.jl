using JLD2, DrWatson, Interpolations

# file = joinpath(@__DIR__,"optimal_kies_rate_50.jld")
# fid = read(h5open(file,"r"))

tripod_balance = let 
    μmem = -55.0 ## target membrane value
    name = "_highres"
    name = ""

    # file = joinpath(@__DIR__,"optimal_IE_μmem$(μmem)_$name.jld2")
    file = joinpath(@__DIR__, "optimal_IE_μmem$(μmem)$name.jld2")
    if !isfile(file)
        @error "File $file does not exist, run 3_optimal_kies.jl first"
    else
        fid = JLD2.load(file) |> dict2ntuple
        @info "Loaded balance K_EI from $file"
    end
    file = joinpath(@__DIR__, "optimal_IE_μmem$(μmem)_soma_only.jld2")
    if !isfile(file)
        @error "File $file does not exist, run 10_soma_only.jl for soma first"
    else
        fid_soma = JLD2.load(file) |> dict2ntuple
        @info "Loaded balance K_EI from $file"
    end

    νs = fid.νs
    opt_kies = fid.opt_kies
    opt_kies_soma = fid_soma.opt_kies

    # soma_syn_models =
        # (ampa_eq = ampa_equivalent, nmda = nmda_soma, kuhn = ampa_kuhn, ampa = human_synapses)

    balance_kie_soma = (
        ampa_eq = opt_kies_soma.AMPA_EQ,
        nmda = opt_kies_soma.NMDA,
        kuhn = opt_kies_soma.KUHN,
        ampa = opt_kies_soma.AMPA,
    )

    balance_kie_rate = (ampa = opt_kies.dend_AMPA, nmda = opt_kies.dend_NMDA)
    balance_kie_gsyn = (ampa = opt_kies.istdp_AMPA, nmda = opt_kies.istdp_NMDA)
    min_AMPA = fid.min_AMPA
    min_NMDA = fid.min_NMDA

    try
        balance_name = fid.name
    catch
        balance_name = ""
    end

    balance_path = @__DIR__
    @unpack νs, models, min_AMPA, min_NMDA = fid
    dend = (kie=balance_kie_rate, gsyn=balance_kie_gsyn, min_AMPA = min_AMPA, min_NMDA=min_NMDA, models=models .* um, νs=νs .* kHz ) 
    soma = (kie=balance_kie_soma, models = opt_kies_soma) 
    (dend=dend, soma=soma)
end
