using Random, Statistics
# score_spikes `window` keyword. Reference = body of score_spikes before the keyword existed (SNNUtils 1.0.0).
function score_spikes_old(model, seq, target_interval = :offset; delay = nothing, pop = :Exc)
    offsets_ids = findall(seq.sequence[seq.line_id.type, :] .== target_interval)
    words = seq.sequence[seq.line_id.words, offsets_ids]
    offset_times = seq.sequence[seq.line_id.offset, offsets_ids]
    all_populations = subpopulations(filter(x->!occursin("noise", x.name), model.stim))
    word_assemblies = Dict{Symbol, Vector{Int}}()
    for label in keys(all_populations)
        word = string(label)
        startswith(word, "w_") && (word_assemblies[Symbol(word[3:end])] = all_populations[label])
    end
    word_assemblies = word_assemblies |> SNNModels.dict2ntuple
    word_list = keys(word_assemblies) |> collect |> sort
    word_count = [count(x->x==word, words) for word in word_list]
    assemblies = [word_assemblies[word] for word in word_list]
    activity_matrix = zeros(Float32, length(word_assemblies), length(word_assemblies))
    spike_count, r = bin_spiketimes(spiketimes(model.pop[pop]), interval = 0:10ms:(offset_times[end]+100ms))
    predicted = Symbol[]
    target = Symbol[]
    for i in eachindex(offsets_ids)
        target_word = findfirst(word_list .== words[i])
        target_interval = offset_times[i] .+ (0:100) .+ delay
        r_idx = findall(x->(r[x]>target_interval[1] && r[x]<target_interval[end]), eachindex(r))
        _spikes = sum(spike_count[:, r_idx], dims = 2)[:, 1]
        for word in eachindex(word_list)
            activity_matrix[word, target_word] += mean(_spikes[assemblies[word]]) / word_count[target_word]
        end
        push!(predicted, word_list[argmax(mean.([_spikes[assembly] for assembly in assemblies]))])
        push!(target, word_list[target_word])
    end
    cm = SNNUtils.confmat(target, predicted)
    return SNNUtils.kappa(cm), cm, activity_matrix
end

@testset "score_spikes window keyword" begin
    E = IF(N = 60)
    Random.seed!(3)
    E.I .= (200 .+ 600 .* rand(60)) .* 1pA
    monitor!(E, [:fire])
    network = (pop = (Exc = E,), )
    lexicon = get_lexicon([:AB, :BA], 50ms)
    seq = generate_sequence(word_phonemes_sequence; lexicon, presentations = 6, mode = :fixed,
        weights = Dict(:AB => 1.0, :BA => 1.0), seed = 1)
    stim = step_input(; inputs = stimuli_names(lexicon).words, network, sym = :ge, p_post = 0.3,
        peak_rate = 10Hz, proj_strength = 1.0)
    model = (pop = (Exc = E,), stim = stim)
    offset_times = seq.sequence[seq.line_id.offset, seq.sequence[seq.line_id.type, :] .== :offset]
    sim!([E]; duration = maximum(offset_times) + 300ms)
    @test length(vcat(spiketimes(E)...)) > 0

    @testset "default window reproduces the old outputs" begin
        for d in (0, -30)
            new = score_spikes(model, seq; delay = d)
            @test new == score_spikes(model, seq; delay = d, window = 100)
            old = score_spikes_old(model, seq; delay = d)
            @test new[1] === old[1] || (isnan(new[1]) && isnan(old[1]))
            @test new[2] == old[2]
            @test new[3] == old[3]
        end
    end

    @testset "window = 50 counts bins in (0, 50) ms" begin
        d = 0
        a100 = copy(score_spikes(model, seq; delay = d)[3])
        a50 = copy(score_spikes(model, seq; delay = d, window = 50)[3])
        @test a50 != a100
        # hand count: bins with left edge in (t0, t0+50), i.e. spikes in [t0+10, t0+50)
        st = spiketimes(E)
        ids = findall(seq.sequence[seq.line_id.type, :] .== :offset)
        words = seq.sequence[seq.line_id.words, ids]
        pops = subpopulations(filter(x->!occursin("noise", x.name), stim))
        word_list = sort([Symbol(string(k)[3:end]) for k in keys(pops) if startswith(string(k), "w_")])
        ref = zeros(Float32, length(word_list), length(word_list))
        for (i, t0) in enumerate(offset_times)
            tw = findfirst(==(words[i]), word_list)
            for (w, wn) in enumerate(word_list)
                n = sum(count(x -> t0 + 10 <= x < t0 + 50, st[j]) for j in pops[Symbol("w_", wn)])
                ref[w, tw] += n / length(pops[Symbol("w_", wn)]) / count(==(wn), words)
            end
        end
        @test a50 ≈ ref
    end

    @testset "invalid window" begin
        @test_throws ArgumentError score_spikes(model, seq; delay = 0, window = 0)
        @test_throws ArgumentError score_spikes(model, seq; delay = 0, window = -10)
    end
end
