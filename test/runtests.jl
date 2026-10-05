using SNNUtils
using SNNModels
using Test
SNNModels.@load_units

@testset "SNNUtils" begin
    @testset "exports are defined" begin
        @test isempty([n for n in names(SNNUtils) if !isdefined(SNNUtils, n)])
    end

    @testset "E/I balance of dendritic neurons" begin
        m = get_model(200um, 1.8, 2)
        @test m[:gax] > 0 && m[:gm] > 0 && length(m[:dend_syn]) == 4
        k = compute_kei(200um, 1.0)
        @test k > 0
        # the residual current vanishes at the balanced inhibitory/excitatory ratio
        @test abs(residual_current(λ = 1.0, kIE = k, L = 200um, NAR = 1.8, Nd = 2)) < 1e-6
        @test length(optimal_kei(200um, 1.8, 2)) == 100
        @test_throws UndefKeywordError residual_current(λ = 1.0)
    end

    @testset "trial_average along any dimension" begin
        A = reshape(Float32.(1:24), 2, 3, 4)   # trials along dim 2
        code, labels = trial_average(A, [1, 2, 1], 2)
        @test size(code) == (2, 4, 2)
        @test code[:, :, 1] ≈ (A[:, 1, :] .+ A[:, 3, :]) ./ 2
        code2, _ = trial_average(rand(4, 6), [1, 2, 1, 2, 1, 2])
        @test size(code2) == (4, 2)
    end

    @testset "merge_intervals keeps the last interval" begin
        @test merge_intervals([[0f0, 1f0], [3f0, 4f0], [5f0, 6f0]]) == [[0f0, 1f0], [3f0, 4f0], [5f0, 6f0]]
        @test merge_intervals([[0f0, 1f0], [1f0, 2f0]]) == [[0f0, 2f0]]
        @test isempty(merge_intervals(Vector{Float32}[]))
    end

    @testset "word_phonemes_sequence :fixed with uneven weights" begin
        lexicon = get_lexicon([:AB, :BA, :CA], 50ms)
        words, phonemes, L = word_phonemes_sequence(; lexicon, presentations = 10, mode = :fixed,
            weights = Dict(:AB => 1, :BA => 1, :CA => 1), seed = 1)
        # 10 words of two phonemes each (rounded counts 3+3+3 = 9 used to fail)
        @test count(w -> w in (:AB, :BA, :CA), words) == 20
    end

    @testset "MultinomialLogisticRegression runs and does not modify X" begin
        X = randn(3, 60)
        labels = vcat(fill(1, 30), fill(2, 30))
        X[1, 31:end] .+= 3.0
        X0 = copy(X)
        acc, params = MultinomialLogisticRegression(X, labels; test_ratio = 0.3)
        @test 0 <= acc <= 1 && size(params) == (3, 2)
        @test X == X0
    end

    @testset "sym_features reads the record" begin
        P = IF(N = 4)
        P.I .= 300pA
        monitor!(P, [:v])
        sim!([P]; duration = 200ms)
        X = sym_features(:v, P, [[10.0f0, 50.0f0], [60.0f0, 100.0f0]])
        @test size(X) == (4, 2) && all(X .< 0)
    end

    @testset "step_input on a point-neuron population" begin
        E = IF(N = 20)
        network = (pop = (Exc = E,),)
        stim = step_input(; inputs = [:A, :B], network, sym = :ge, p_post = 0.5,
            peak_rate = 10Hz, proj_strength = 1.0)
        @test length(stim) == 2
    end
end
