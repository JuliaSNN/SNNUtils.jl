# Tests of the submodule SNNUtils.SpikeTimit on a synthetic corpus (see fixture.jl).
using MAT, NPZ, DataFrames, Random
using SNNUtils.SpikeTimit
include("fixture.jl")

const ST = SNNUtils.SpikeTimit
const FX = make_fixture(joinpath(mktempdir(), "drive_fixture", "SpikeTIMIT"); mat = true)
const TRAIN = joinpath(FX.root, "train")

# sentences of the train split in the order of create_dataset (sorted by speaker, senID)
const TRAIN_SENTENCES = sort(filter(s -> s.split == "train", FX.sentences),
                             by = s -> (s.speaker, s.senID))

seconds(records) = [(parse(Int, string(a)) / 16000.0, parse(Int, string(b)) / 16000.0)
                    for (a, b, _) in records]

@testset "SpikeTimit" begin
    @testset "namespace" begin
        @test :SpikeTimit in names(SNNUtils)
        @test !(:create_dataset in names(SNNUtils))
        @test !isdefined(SNNUtils, :create_dataset)
        @test isempty([n for n in names(ST) if !isdefined(ST, n)])
        @test !isdefined(ST, :Spiketimes)
        @test isempty([n for n in names(ST) if !Base.Docs.hasdoc(ST, n)])
    end

    @testset "SpikeTimitConfig" begin
        c = SpikeTimitConfig()
        @test (c.sr, c.correction, c.bae_frequencies, c.bae_bins) == (16000.0, 5.0, 20, 31)
        @test isbits(c)
        @test SpikeTimitConfig(sr = 8000).sr === 8000.0
        @test_throws ArgumentError SpikeTimitConfig(sr = 0.0)
        @test_throws ArgumentError SpikeTimitConfig(correction = -1.0)
        @test_throws ArgumentError SpikeTimitConfig(bae_frequencies = 0)
        @test_throws ArgumentError SpikeTimitConfig(bae_bins = 0)
    end

    @testset "S_TO_MS" begin
        @test ST.S_TO_MS == Float64(SNNModels.second)
        @test ST.S_TO_MS isa Float64
    end

    @testset "dataset_path" begin
        @test dataset_path(root = FX.root) == normpath(abspath(FX.root))
        @test dataset_path("train", "dr1"; root = FX.root) == normpath(joinpath(FX.root, "train", "dr1"))
        withenv("SPIKETIMIT_PATH" => FX.root) do
            @test dataset_path("DOC") == normpath(joinpath(FX.root, "DOC"))
        end
        # the joined path is not required to exist, but stays inside the root
        p = dataset_path("nothing_here"; root = FX.root)
        @test startswith(p, normpath(abspath(FX.root)))
        withenv("SPIKETIMIT_PATH" => nothing) do
            err = try
                dataset_path()
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("SPIKETIMIT_PATH", err.msg)
        end
        withenv("SPIKETIMIT_PATH" => "") do
            @test_throws ArgumentError dataset_path()
        end
        missing_dir = joinpath(FX.root, "does_not_exist")
        withenv("SPIKETIMIT_PATH" => missing_dir) do
            err = try
                dataset_path()
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin(missing_dir, err.msg)
        end
        @test_throws ArgumentError dataset_path(".."; root = FX.root)
        @test_throws ArgumentError dataset_path("train", "../../x"; root = FX.root)
        @test_throws ArgumentError dataset_path(abspath("/etc"); root = FX.root)
    end

    @testset "encodings" begin
        @test encoding("bae") === BAE()
        @test encoding("cochlea35") === Cochlea(35)
        @test encoding("cochlea70") === Cochlea(70)
        @test encoding(Cochlea(35)) === Cochlea(35)
        @test BAE() isa SpikeEncoding && Cochlea(70) isa SpikeEncoding
        @test_throws ArgumentError Cochlea(36)
        err(f) = try
            f()
        catch e
            e
        end
        @test err(() -> Cochlea(36)).msg == "Encoding must be 35 or 70"
        @test err(() -> encoding("cochlea")).msg == "Define cochlea encoding: 35 or 70"
        @test err(() -> encoding("mfcc")).msg == "Encoding must be 'cochlea(35|70)' or 'bae'"
        @test err(() -> encoding("cochlea36")) isa ArgumentError
        @test err(() -> encoding("cochleaXX")).msg == "Encoding must be 'cochlea(35|70)' or 'bae'"
        @test ST.spike_file(BAE(), "/a/sa1") == "/a/sa1_spike_timing.mat"
        @test ST.spike_file(Cochlea(35), "/a/sa1") == "/a/sa1_cochlea35.npz"
        @test ST.n_channels(BAE(), SpikeTimitConfig()) == 620
        @test ST.n_channels(BAE(), SpikeTimitConfig(bae_bins = 10)) == 200
        @test ST.n_channels(Cochlea(70), SpikeTimitConfig()) == 70
    end

    @testset "Phone, Word, Transcription" begin
        p = Phone("dh", 0.0, 0.1)
        @test p == Phone("dh", 0.0, 0.1) && hash(p) == hash(Phone("dh", 0.0, 0.1))
        @test p != Phone("ae", 0.0, 0.1)
        w = Word()
        @test (w.word, w.phones, w.duration, w.t0, w.t1) == ("", Phone[], -1.0, -1.0, -1.0)
        w1 = Word("that", [Phone("dh", 0.0, 0.1)], 0.2, 1.0, 1.2)
        w2 = Word(word = "that", phones = [Phone("dh", 0.0, 0.1)], duration = 0.2, t0 = 1.0, t1 = 1.2)
        @test w1 == w2 && hash(w1) == hash(w2)
        @test w1 != Word("that", Phone[], 0.2, 1.0, 1.2)
        @test length(unique([w1, w2])) == 1
        t = Transcription()
        @test t isa Transcription{Float64} && length(t) == 0
        @test Transcription{Int}() isa Transcription{Int}
        t1 = Transcription([(0.0, 0.1), (0.1, 0.3)], ["a", "b"])
        t2 = Transcription{Float64}([(0.0, 0.1), (0.1, 0.3)], ["a", "b"])
        @test t1 == t2 && hash(t1) == hash(t2) && length(t1) == 2
        @test t1 != Transcription([(0.0, 0.1), (0.1, 0.3)], ["a", "c"])
        @test Transcription([(0, 1)], ["a"]) isa Transcription{Int}
        @test_throws ArgumentError Transcription([(0.0, 0.1)], ["a", "b"])
        @test Transcriptions{Float64} == typeof((words = t1, phones = t2))
        @test SpikeTrains{Float64} == Vector{Vector{Float64}}
    end

    df = create_dataset(TRAIN)
    @testset "create_dataset" begin
        @test create_dataset(; dir = TRAIN) == df
        withenv("SPIKETIMIT_PATH" => FX.root) do
            @test create_dataset() == df
        end
        @test nrow(df) == 3
        @test names(df) == ["speaker", "senID", "dialect", "gender", "path", "words", "phones", "sentence"]
        @test eltype.(eachcol(df)) == [String, String, Int, Char, String, Transcription{Float64},
                                       Transcription{Float64}, Vector{String}]
        @test collect(zip(df.speaker, df.senID)) == [("fabc0", "sa1"), ("fabc0", "sx2"), ("mdef0", "sa1")]
        @test df.dialect == [1, 1, 2]           # the decoy "drive_fixture" component is ignored
        @test df.gender == ['f', 'f', 'm']
        @test endswith(df.path[1], joinpath("dr1", "fabc0", "sa1"))
        @test df.path[1] == joinpath(TRAIN, "dr1", "fabc0", "sa1")
        for (row, s) in zip(eachrow(df), TRAIN_SENTENCES)
            @test row.words.intervals == seconds(s.words)
            @test row.words.signs == [w[3] for w in s.words]
            @test row.phones.intervals == seconds(s.phones)
            @test row.phones.signs == [p[3] for p in s.phones]
            @test row.sentence == row.words.signs
            @test row.sentence !== row.words.signs
        end
        @test length(df.words[2]) == 2                         # trailing blank line skipped
        @test !any(occursin("notes", p) for p in df.path)      # non-.wav files ignored
        dft = create_dataset(joinpath(FX.root, "test"))
        @test nrow(dft) == 1 && dft.dialect == [3] && dft.words[1].intervals == [(0.0, 0.2)]
        # empty directory
        dfe = create_dataset(mktempdir())
        @test nrow(dfe) == 0 && eltype.(eachcol(dfe)) == eltype.(eachcol(df))
        # custom sample rate doubles the times
        df8 = create_dataset(TRAIN; config = SpikeTimitConfig(sr = 8000.0))
        @test all(x.intervals == [2 .* b for b in y.intervals] for (x, y) in zip(df8.words, df.words))
        @test_throws ArgumentError create_dataset(joinpath(FX.root, "not_a_dir"))
    end

    @testset "create_dataset failures" begin
        # speaker directory not under dr<k>
        d = mktempdir()
        mkpath(joinpath(d, "train", "xx", "s0"))
        touch(joinpath(d, "train", "xx", "s0", "a.wav"))
        err = try
            create_dataset(joinpath(d, "train"))
        catch e
            e
        end
        @test err isa ArgumentError && occursin(joinpath("xx", "s0"), err.msg)
        # missing .phn: the SystemError of open propagates
        d = mktempdir()
        sdir = joinpath(d, "dr1", "s0")
        mkpath(sdir)
        touch(joinpath(sdir, "a.wav"))
        write(joinpath(sdir, "a.wrd"), "0 10 x\n")
        @test_throws SystemError create_dataset(d)
        # malformed transcription lines: ArgumentError with file:line
        f = joinpath(mktempdir(), "bad.wrd")
        for bad in ["0 10 a\n5 20\n", "0 10 a\n5 x b\n", "0 10 a\n20 5 b\n", "0 10 a\n1.5 20 b\n",
                    "0 10 a\n5 20 b c\n"]
            write(f, bad)
            err = try
                ST._read_transcription(f, 16000.0)
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("$(f):2:", err.msg)
        end
        write(f, "0 16000 a\n\n   \n16000 32000 b\n")
        t = ST._read_transcription(f, 16000.0)
        @test t.intervals == [(0.0, 1.0), (1.0, 2.0)] && t.signs == ["a", "b"]
        @test ST._parse_dialect(joinpath("x", "dr7", "s0")) == 7
        @test ST._parse_dialect(joinpath("x", "dr12", "s0") * "/") == 12
        @test_throws ArgumentError ST._parse_dialect(joinpath("drive", "x", "s0"))
        @test_throws ArgumentError ST._parse_dialect(joinpath("x", "dr", "s0"))
    end

    @testset "create_dictionary" begin
        dfile = joinpath(FX.root, "DOC", "TIMITDIC.TXT")
        dict = create_dictionary(dfile)
        @test dict == FX.dictionary
        @test dict isa Dict{String,String}
        @test create_dictionary(; file = dfile) == dict
        withenv("SPIKETIMIT_PATH" => FX.root) do
            @test create_dictionary() == dict
        end
        f = joinpath(mktempdir(), "dic.txt")
        write(f, "; c\nab  /a b/\nab  /x/\n")
        @test create_dictionary(f) == Dict("ab" => "x")       # later duplicates overwrite
        write(f, "; c\nab  /a b/\nno slash here\n")
        err = try
            create_dictionary(f)
        catch e
            e
        end
        @test err isa ArgumentError && occursin("$(f):3", err.msg)
    end

    @testset "find_word" begin
        had = find_word(df, "had")
        @test collect(zip(had.speaker, had.senID)) == [("fabc0", "sa1"), ("mdef0", "sa1")]
        @test nrow(find_word(df, ["had", "she"])) == 3
        @test find_word(; df, words = "had") == had
        @test find_word(; df, words = ["had"]) == had
        @test nrow(find_word(df, "absent")) == 0
        @test names(find_word(df, "absent")) == names(df)
        @test nrow(find_word(df, "That")) == 0                  # case-sensitive
        had.speaker[1] = "changed"
        @test df.speaker[1] == "fabc0"                          # result is a copy
    end

    @testset "count_words" begin
        @test count_words(df, ["that", "had", "she"]) == Dict("that" => 3, "had" => 2, "she" => 2)
        @test count_words(df, ["absent"]) == Dict("absent" => 0)
        ws = [Word(word = "a"), Word(word = "b"), Word(word = "a")]
        @test count_words(ws) == Dict("a" => 2, "b" => 1)
    end

    @testset "read_spiketimes / get_spiketimes" begin
        for (row, s) in zip(eachrow(df), TRAIN_SENTENCES)
            T = fixture_duration(s)
            for n in (35, 70)
                st = read_spiketimes(row.path, Cochlea(n))
                @test st isa SpikeTrains{Float64}
                @test st == fixture_spikes(n, T)
                @test read_spiketimes(row.path, "cochlea$(n)") == st
            end
            raw = FX.raw_bae[row.path]
            bae = read_spiketimes(row.path, BAE())
            @test bae isa SpikeTrains{Float64} && length(bae) == 620
            @test bae == [x .* 5.0 for x in raw]
            @test bae[3] == [0.01 * 5.0]                  # scalar cell
            @test bae[2] == raw[2] .* 5.0                 # 1 x k row cell
            @test bae[11] == Float64[]                    # 0 x 0 cell
            @test read_spiketimes(row.path, "bae") == bae
            @test read_spiketimes(row.path, BAE(); config = SpikeTimitConfig(correction = 1.0)) == raw
        end
        sp = get_spiketimes(df, Cochlea(35))
        @test sp isa Vector{SpikeTrains{Float64}} && length(sp) == 3
        @test sp == [read_spiketimes(p, Cochlea(35)) for p in df.path]
        @test get_spiketimes(; df, encoding = "cochlea35") == sp
        @test get_spiketimes(df[[3, 1], :], "bae") == [read_spiketimes(df.path[i], BAE()) for i in (3, 1)]
        @test get_spiketimes(df[1:0, :], Cochlea(35)) == SpikeTrains{Float64}[]
        @test_throws UndefKeywordError get_spiketimes(; df)
        # malformed files: ArgumentError naming the file
        for f in ("x", "y", "u", "v")
            err = try
                read_spiketimes(joinpath(FX.bad, f), Cochlea(35))
            catch e
                e
            end
            @test err isa ArgumentError && occursin(joinpath(FX.bad, f), err.msg)
        end
        @test_throws SystemError read_spiketimes(joinpath(FX.bad, "missing"), Cochlea(35))
        for f in ("z", "w", "s")
            @test_throws ArgumentError read_spiketimes(joinpath(FX.bad, f), BAE())
        end
        # MAT extension guard
        @test ST._require_mat_ext() === Base.get_extension(SNNUtils, :SNNUtilsMATExt)
        @test ST._require_mat_ext() isa Module
        err = try
            ST._require_mat_ext(nothing)
        catch e
            e
        end
        @test err isa ErrorException && err.msg == ST.MAT_EXT_MESSAGE
        @test occursin("using MAT", err.msg)
        # BAE cells, without MAT
        @test ST._normalize_bae_cells(Any[0.5, reshape([0.1, 0.2], :, 1), [0.3 0.4], zeros(0, 0), 2], 5.0) ==
              [[2.5], [0.5, 1.0], [1.5, 2.0], Float64[], [10.0]]
        @test ST._normalize_bae_cells(Any[], 5.0) == Vector{Float64}[]
        @test_throws ArgumentError ST._normalize_bae_cells(Any["x"], 5.0)
        @test_throws ArgumentError ST._normalize_bae_cells(Any[Any[1.0]], 5.0)
        @test ST._bae_cells(Any[1.0 2.0], 2) == Any[1.0, 2.0]
        @test_throws ArgumentError ST._bae_cells(Any[1.0 2.0], 3)
        @test_throws ArgumentError ST._bae_cells(reshape(Any[1.0, 2.0], 2, 1), 2)
        @test_throws ArgumentError ST._bae_cells(1.0, 1)
    end

    bae1 = read_spiketimes(df.path[1], BAE())
    coch1 = read_spiketimes(df.path[1], Cochlea(35))

    @testset "resample_spikes!" begin
        st = deepcopy(bae1)
        @test resample_spikes!(st; n_feat = 1) === st
        # reference: the old mapping written as a plain loop
        function ref_resample(st, n_feat; F = 20, B = 31)
            nb = ceil(Int, B / n_feat)
            out = [Float64[] for _ in 1:(nb * F)]
            for f in 1:F, b in 1:B
                append!(out[ceil(Int, b / n_feat) + (f - 1) * nb], st[b + (f - 1) * B])
            end
            return out
        end
        for n_feat in (2, 3, 8, 11)
            r = resample_spikes!(deepcopy(bae1); n_feat)
            @test r isa SpikeTrains{Float64}
            @test length(r) == 20 * ceil(Int, 31 / n_feat)
            @test r == ref_resample(bae1, n_feat)
        end
        # hand example: channel c holds the single spike c
        lab = [[Float64(c)] for c in 1:620]
        r = resample_spikes!(lab; n_feat = 8)
        @test length(r) == 80
        @test r[1] == 1.0:8.0 && r[4] == 25.0:31.0 && r[5] == 32.0:39.0 && r[80] == 614.0:620.0
        @test lab == [[Float64(c)] for c in 1:620]          # input not modified
        # batch
        batch = [deepcopy(bae1), deepcopy(bae1)]
        @test resample_spikes!(batch; n_feat = 8) === batch
        @test length.(batch) == [80, 80] && batch[1] == ref_resample(bae1, 8)
        # other BAE geometry through the config
        small = [[Float64(c)] for c in 1:8]
        @test resample_spikes!(small; n_feat = 2, config = SpikeTimitConfig(bae_frequencies = 2, bae_bins = 4)) ==
              [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        @test_throws ArgumentError resample_spikes!(deepcopy(bae1); n_feat = 0)
        @test_throws ArgumentError resample_spikes!(deepcopy(bae1); n_feat = 12)
        @test_throws ArgumentError resample_spikes!(deepcopy(coch1); n_feat = 8)
    end

    @testset "transform_into_bursts!" begin
        st = deepcopy(bae1)
        @test transform_into_bursts!(st; rng = MersenneTwister(1)) === st
        @test all(issorted, st)
        @test all(all(t in st[c] for t in bae1[c]) for c in eachindex(st))
        @test all(2 * length(bae1[c]) <= length(st[c]) <= 6 * length(bae1[c]) for c in eachindex(st))
        @test all(isempty(st[c]) for c in eachindex(st) if isempty(bae1[c]))
        st2 = transform_into_bursts!(deepcopy(bae1); rng = MersenneTwister(1))
        @test st2 == st                                         # same seed, same result
        # one spike per channel: count of added spikes and their minimal delay (4 ms)
        for inc in (0, 2)
            xs = [transform_into_bursts!([[1.0], Float64[], [2.0]];
                                         spikes_per_burst_increase = inc, rng = MersenneTwister(seed))
                  for seed in 1:50]
            @test all(x[2] == Float64[] for x in xs)
            @test all(x[1][1] == 1.0 && all(x[1][2:end] .>= 1.004) for x in xs)
            @test all(x[3][1] == 2.0 && all(x[3][2:end] .>= 2.004) for x in xs)
            counts = [length(x[c]) - 1 for x in xs for c in (1, 3)]
            @test all(1 + inc .<= counts .<= 5 + inc)
            @test minimum(counts) == 1 + inc                    # most bursts have 2 + inc spikes
        end
        # batch: the rng is used sample after sample
        batch = [deepcopy(bae1), deepcopy(coch1)]
        @test transform_into_bursts!(batch; rng = MersenneTwister(7)) === batch
        rng = MersenneTwister(7)
        @test batch == [transform_into_bursts!(deepcopy(bae1); rng), transform_into_bursts!(deepcopy(coch1); rng)]
        @test_throws ArgumentError transform_into_bursts!(deepcopy(bae1); spikes_per_burst_increase = -2)
    end

    @testset "remove_frequencies! / select_frequencies!" begin
        st = deepcopy(coch1)
        ref = st[1]
        @test remove_frequencies!(st; frequencies = [1, 2]) === st
        @test st[1] == Float64[] && st[2] == Float64[] && st[3:end] == coch1[3:end]
        @test ref == coch1[1] && st[1] !== ref                  # replaced, old vector untouched
        st = deepcopy(coch1)
        @test select_frequencies!(st; frequencies = 3:5) === st
        @test st[3:5] == coch1[3:5] && all(isempty, st[[1, 2, 6:35...]])
        batch = [deepcopy(coch1), deepcopy(coch1)]
        @test remove_frequencies!(batch; frequencies = 1:10) === batch
        @test all(all(isempty, b[1:10]) && b[11:end] == coch1[11:end] for b in batch)
        batch = [deepcopy(coch1), deepcopy(coch1)]
        @test select_frequencies!(batch; frequencies = [2]) === batch
        @test all(b[2] == coch1[2] && all(isempty, b[[1, 3:35...]]) for b in batch)
        @test_throws ArgumentError remove_frequencies!(deepcopy(coch1); frequencies = [36])
        @test_throws ArgumentError remove_frequencies!(deepcopy(coch1); frequencies = 0:1)
        @test_throws ArgumentError select_frequencies!(deepcopy(coch1); frequencies = [40])
        @test_throws ArgumentError remove_frequencies!([deepcopy(coch1)]; frequencies = [36])
        @test_throws ArgumentError select_frequencies!([deepcopy(coch1)]; frequencies = [36])
    end

    sec(x) = x / 16000.0

    @testset "word_intervals" begin
        wi = word_intervals(df, "that")
        @test wi isa Vector{Vector{NTuple{2,Float64}}}
        @test wi == [[(sec(9600), sec(12800))], [(sec(800), sec(4000))],
                     [(sec(1600), sec(4800)), (sec(8000), sec(11200))]]
        @test word_intervals(df, "absent") == [NTuple{2,Float64}[] for _ in 1:3]
    end

    @testset "get_spikes_in_interval" begin
        st = [[0.25, 0.5, 0.75], [0.5], Float64[]]
        sp, d = get_spikes_in_interval(st, (0.25, 0.75))   # strict bounds: 0.25 and 0.75 excluded
        @test sp == [[0.25], [0.25], Float64[]] && d === 0.5
        @test sp isa SpikeTrains{Float64}
        sp32, _ = get_spikes_in_interval([Float32[0.5]], (0.0, 1.0))
        @test sp32 isa SpikeTrains{Float64}
        sps, ds = get_spikes_in_interval(st, [(0.0, 0.5), (0.25, 1.0)])
        @test sps == [[[0.25], Float64[], Float64[]], [[0.25, 0.5], [0.25], Float64[]]] && ds == [0.5, 0.75]
        spb, db = get_spikes_in_interval([st, st], [[(0.25, 0.75)], [(0.0, 1.0), (0.5, 1.0)]])
        @test spb == [sp, [[0.25, 0.5, 0.75], [0.5], Float64[]], [[0.25], Float64[], Float64[]]]
        @test db == [0.5, 1.0, 0.5]
        @test spb isa Vector{SpikeTrains{Float64}}
        @test get_spikes_in_interval([st], [NTuple{2,Float64}[]]) == (SpikeTrains{Float64}[], Float64[])
        @test_throws DimensionMismatch get_spikes_in_interval([st, st], [[(0.0, 1.0)]])
    end

    @testset "get_word_spikes" begin
        sp, d = get_word_spikes(df, "that"; encoding = Cochlea(35))
        @test (sp, d) == get_word_spikes(df, ["that"]; encoding = "cochlea35")
        @test length(sp) == 4 && all(≈(0.2), d)
        sp_all = get_spiketimes(df, Cochlea(35))
        @test sp[1] == get_spikes_in_interval(sp_all[1], (sec(9600), sec(12800)))[1]
        @test sp[4] == get_spikes_in_interval(sp_all[3], (sec(8000), sec(11200)))[1]
        # several words: word, then row, then occurrence
        sw, dw = get_word_spikes(df, ["had", "that"]; encoding = Cochlea(35))
        sh, dh = get_word_spikes(df, "had"; encoding = Cochlea(35))
        @test sw == vcat(sh, sp) && dw == vcat(dh, d)
        @test_throws UndefKeywordError get_word_spikes(df, "that")
    end

    @testset "get_word_labels" begin
        labels = get_word_labels(df, "that")
        @test length.(labels) == [1, 1, 2]
        w = labels[3]
        rel(a, b, t0) = (sec(a) - sec(t0), sec(b) - sec(t0))
        @test w[1] == Word("that", [Phone("dh", rel(1600, 3200, 1600)...), Phone("ae", rel(3200, 4000, 1600)...),
                                    Phone("t", rel(4000, 4800, 1600)...)],
                           sec(4800) - sec(1600), sec(1600), sec(4800))
        # inclusive bounds: dh starts at the word onset, t ends at the offset; d (7200-8000) ends at
        # the onset of the second "that" but starts before it, so it is excluded
        @test [p.ph for p in w[2].phones] == ["dh", "ae", "t"]
        @test w[2].t0 == sec(8000) && w[2].t1 == sec(11200) && w[2].duration == sec(11200) - sec(8000)
        @test get_word_labels(df, "absent") == [Word[], Word[], Word[]]
    end

    @testset "select_words" begin
        sw = select_words(df, ["that", "had", "she"]; encoding = Cochlea(35))
        @test keys(sw) == (:durations, :spikes, :labels)
        @test [nrow(find_word(df, w)) for w in ["that", "had", "she"]] == [3, 2, 2]
        @test [l.word for l in sw.labels] == vcat(fill("that", 4), fill("had", 2), fill("she", 2))
        @test length(sw.spikes) == length(sw.durations) == 8
        @test sw.durations == [l.duration for l in sw.labels]
        @test count_words(sw.labels) == Dict("that" => 4, "had" => 2, "she" => 2)
        @test sw.spikes[1:4] == get_word_spikes(df, "that"; encoding = Cochlea(35))[1]
        # reproducible with an rng
        a = select_words(df, ["that"]; encoding = Cochlea(35), samples = 1, rng = MersenneTwister(3))
        b = select_words(df, ["that"]; encoding = Cochlea(35), samples = 1, rng = MersenneTwister(3))
        @test a == b
        # without replacement: 3 samples out of 3 sentences always give all 4 occurrences
        all_t0 = sort([l.t0 for l in sw.labels[1:4]])
        @test all(sort([l.t0 for l in select_words(df, ["that"]; encoding = Cochlea(35), samples = 3,
                                                    rng = MersenneTwister(seed)).labels]) == all_t0
                  for seed in 1:20)
        # replace = true: the old rule rand(rows, samples)
        dfw = find_word(df, "that")
        for seed in 1:5
            inds = rand(MersenneTwister(seed), 1:3, 2)
            r = select_words(df, ["that"]; encoding = Cochlea(35), samples = 2, replace = true,
                             rng = MersenneTwister(seed))
            @test r.labels == reduce(vcat, get_word_labels(dfw[inds, :], "that"))
            @test r.spikes == get_word_spikes(dfw[inds, :], "that"; encoding = Cochlea(35))[1]
        end
        # more samples than sentences: all of them
        @test select_words(df, ["had"]; encoding = Cochlea(35), samples = 10).labels ==
              select_words(df, ["had"]; encoding = Cochlea(35)).labels
        @test select_words(df, ["absent", "had"]; encoding = Cochlea(35)) ==
              select_words(df, ["had"]; encoding = Cochlea(35))
        @test_throws ArgumentError select_words(df, ["had"]; encoding = Cochlea(35), samples = 0)
        @test_throws UndefKeywordError select_words(df, ["had"])
    end

    @testset "get_encoding_dimension" begin
        @test get_encoding_dimension(get_spiketimes(df, Cochlea(35))) == 35
        @test get_encoding_dimension(get_spiketimes(df, Cochlea(70))) == 70
        @test get_encoding_dimension(get_spiketimes(df, BAE())) == 620
        @test_throws ArgumentError get_encoding_dimension(SpikeTrains{Float64}[])
    end

    @testset "internal helpers" begin
        ft, nn = ST._times_x_neurons([[0.5, 0.25], [0.25], Float64[], [0.5]])
        @test ft == [0.25, 0.5] && nn == [[1, 2], [1, 4]]
        @test ST._times_x_neurons(SpikeTrains{Float64}()) == (Float64[], Vector{Int}[])
        ft = [0.3, 0.1, 0.3, 0.2]
        nn = [[1], [2], [3], [4]]
        ST._sort_spikes!(ft, nn)
        @test ft == [0.1, 0.2, 0.3, 0.3] && nn == [[2], [4], [1], [3]]      # stable
        ft = [1, 2, 2, 2, 3, 3]
        nn = [[1], [2], [3], [4], [5], [6]]
        ST._remove_duplicate_ft!(ft, nn)
        @test ft == [1, 2, 3] && nn == [[1], [2, 3, 4], [5, 6]]              # run of 3 keeps all
        ft, nn = Int[], Vector{Int}[]
        ST._remove_duplicate_ft!(ft, nn)
        @test isempty(ft) && isempty(nn)
    end

    @testset "stack_spiketimes" begin
        s1 = [[0.01, 0.03], [0.03]]
        s2 = [Float64[], [0.02]]
        dd, silence, shift = [0.1, 0.07], 0.05, 0.3
        out = stack_spiketimes([s1, s2], dd, silence, shift)
        g = shift
        e1 = [0.01, 0.03] .+ g
        g += dd[1]
        g += silence
        e2 = [0.02] .+ g
        @test out.ft == vcat(e1, e2)
        @test out.neurons == [[1], [1, 2], [2]]
        @test out isa NamedTuple{(:ft, :neurons),Tuple{Vector{Float64},Vector{Vector{Int}}}}
        # equal times from different samples are not merged
        out = stack_spiketimes([[[0.01]], [[0.01]]], [0.0, 0.0], 0.0, 0.0)
        @test out.ft == [0.01, 0.01] && out.neurons == [[1], [1]]
        @test stack_spiketimes(SpikeTrains{Float64}[], Float64[], 0.1, 0.0).ft == Float64[]
        @test_throws DimensionMismatch stack_spiketimes([s1, s2], [0.1], 0.0, 0.0)
        @test_throws ArgumentError stack_spiketimes([s1], [0.1], -0.1, 0.0)
        @test_throws ArgumentError stack_spiketimes([s1], [0.1], 0.1, -1.0)
    end

    @testset "stack_labels" begin
        w1 = Word("a", [Phone("x", 0.0, 0.05), Phone("y", 0.05, 0.1)], 0.1, 1.0, 1.1)
        w2 = Word("b", [Phone("z", 0.0, 0.07)], 0.07, 2.0, 2.07)
        dd, silence, shift = [0.1, 0.07], 0.05, 0.3
        t = stack_labels([w1, w2], dd, silence, shift)
        @test t isa Transcriptions{Float64}
        g1 = shift
        g2 = g1 + (dd[1] + silence)
        @test t.words.intervals == [(g1, g1 + 0.1), (g2, g2 + 0.07)]
        @test t.words.signs == ["a", "b"]
        @test t.phones.intervals == [(g1 + 0.0, g1 + 0.05), (g1 + 0.05, g1 + 0.1), (g2 + 0.0, g2 + 0.07)]
        @test t.phones.signs == ["x", "y", "z"]
        @test_throws ArgumentError stack_labels([w1, w2], [0.1, 0.08], silence, shift)
        @test_throws DimensionMismatch stack_labels([w1, w2], [0.1], silence, shift)
    end

    @testset "mix_inputs" begin
        m = mix_inputs(MersenneTwister(1), 3, 2)
        @test sort(m) == [1, 1, 2, 2, 3, 3]
        @test m == mix_inputs(MersenneTwister(1), 3, 2)
        @test sort(mix_inputs(4, 3)) == sort(repeat(1:4, 3))
        @test mix_inputs(3, 0) == Int[]
        @test_throws ArgumentError mix_inputs(-1, 2)
        @test_throws ArgumentError mix_inputs(MersenneTwister(1), 2, -2)
    end

    wi = select_words(df, ["that", "had"]; encoding = Cochlea(35))
    @testset "get_ordered_spikes" begin
        ids = [2, 1, 5]
        o, t = get_ordered_spikes(wi; silence_time = 0.1, shift = 0.2, ids)
        @test o == stack_spiketimes(wi.spikes[ids], wi.durations[ids], 0.1, 0.2)
        @test t == stack_labels(wi.labels[ids], wi.durations[ids], 0.1, 0.2)
        o2, t2 = get_ordered_spikes(wi; silence_time = 0.1, shift = 0.2, repetitions = 2,
                                    rng = MersenneTwister(4))
        ids2 = mix_inputs(MersenneTwister(4), length(wi.durations), 2)
        @test o2 == stack_spiketimes(wi.spikes[ids2], wi.durations[ids2], 0.1, 0.2)
        @test t2 == stack_labels(wi.labels[ids2], wi.durations[ids2], 0.1, 0.2)
        @test length(t2.words) == 2 * length(wi.durations)
        @test_throws ArgumentError get_ordered_spikes(wi; silence_time = 0.1, shift = 0.2)
    end

    @testset "transcriptions_dt" begin
        _, t = get_ordered_spikes(wi; silence_time = 0.1, shift = 0.2, ids = [1, 5])
        for dt in (0.1, 1, 0.25)
            ti = dt == 0.1 ? transcriptions_dt(t) : transcriptions_dt(t; dt)
            @test ti isa Transcriptions{Int}
            @test ti.words.intervals == [(round(Int, a * 1000 / dt), round(Int, b * 1000 / dt))
                                         for (a, b) in t.words.intervals]
            @test ti.phones.intervals == [(round(Int, a * 1000 / dt), round(Int, b * 1000 / dt))
                                          for (a, b) in t.phones.intervals]
            @test ti.words.signs == t.words.signs && ti.phones.signs == t.phones.signs
        end
    end

    @testset "ft_dt" begin
        data = (ft = [0.001, 0.00102, 0.002], neurons = [[1], [2], [3]])
        r = ft_dt(data)
        @test r.ft == [10, 20] && r.neurons == [[1, 2], [3]]
        @test r isa NamedTuple{(:ft, :neurons),Tuple{Vector{Int},Vector{Vector{Int}}}}
        @test data.neurons == [[1], [2], [3]] && data.ft == [0.001, 0.00102, 0.002]   # input unchanged
        @test ft_dt(data; unit = "s") == r
        @test ft_dt((ft = [1.0, 1.02, 2.0], neurons = [[1], [2], [3]]); unit = :ms) == r
        @test ft_dt(data; dt = 1.0).ft == [1, 2]
        r3 = ft_dt((ft = [0.001, 0.00101, 0.00102], neurons = [[1], [2], [3]]))
        @test r3.ft == [10] && r3.neurons == [[1, 2, 3]]                              # run of 3
        @test_throws DomainError ft_dt(data; unit = :us)
    end

    @testset "get_savepoints" begin
        trans = (words = Transcription([(0, 100)], ["a"]),
                 phones = Transcription([(0, 50), (50, 100)], ["x", "y"]))
        sp = get_savepoints(trans; per_word = 4, per_phone = 2)
        @test sp.words == [25 50 75 100]
        @test sp.phones == [25 50; 75 100]
        @test sp.words isa Matrix{Int}
        @test get_savepoints(trans; per_word = 3, per_phone = 1).words == [33 66 99]
        tf = (words = Transcription([(0.0, 0.1)], ["a"]), phones = Transcription([(0.0, 0.1)], ["x"]))
        @test_throws MethodError get_savepoints(tf; per_word = 4, per_phone = 2)
        @test_throws ArgumentError get_savepoints(trans; per_word = 0, per_phone = 2)
        @test_throws ArgumentError get_savepoints(trans; per_word = 1, per_phone = 0)
    end

    @testset "matrix2spiketimes / spiketimes2sequence" begin
        st, T = matrix2spiketimes(BitMatrix([1 0 1; 0 1 0]), 0.5)
        @test st == [[0.5, 1.5], [1.0]] && T === 1.5
        @test st isa SpikeTrains{Float64}
        @test matrix2spiketimes(Bool[1 1; 0 0], 1) == (spiketimes = [[1.0, 2.0], Float64[]], duration = 2.0)
        x = [[0.5, 0.25], [0.25], Float64[]]
        @test spiketimes2sequence(x) == stack_spiketimes([x], [0.0], 0.0, 0.0)
        @test spiketimes2sequence(x) == (ft = [0.25, 0.5], neurons = [[1, 2], [1]])
    end

    @testset "to_snn_spiketimes" begin
        st = deepcopy(bae1)
        r = to_snn_spiketimes(st)
        @test r isa SNNModels.Spiketimes && r isa Vector{Vector{Float32}}
        @test r == [Float32[Float32(t * 1000.0) for t in ch] for ch in bae1]
        @test st == bae1
        @test to_snn_spiketimes([[0.5, 0.25]]) == [Float32[500, 250]]          # order kept
    end

    pairs_of(p) = sort(collect(zip(p.spiketimes, p.neurons)))
    @testset "spiketime_parameter" begin
        p = spiketime_parameter(coch1)
        @test p isa SNNModels.SpikeTimeStimulusParameter{Vector{Float32},Vector{Int}}
        @test issorted(p.spiketimes)
        @test pairs_of(p) == sort([(Float32(t * 1000), c) for (c, ch) in enumerate(coch1) for t in ch])
        o, _ = get_ordered_spikes(wi; silence_time = 0.1, shift = 0.2, ids = [1, 2, 5])
        p2 = spiketime_parameter(o)
        @test p2 isa SNNModels.SpikeTimeStimulusParameter{Vector{Float32},Vector{Int}}
        @test issorted(p2.spiketimes)
        @test pairs_of(p2) == sort([(Float32(t * 1000), n) for (t, ns) in zip(o.ft, o.neurons) for n in ns])
        @test pairs_of(spiketime_parameter(spiketimes2sequence(coch1))) == pairs_of(p)
        @test spiketime_parameter(coch1; delay = 10.0).spiketimes == p.spiketimes .+ 10.0f0
        @test spiketime_parameter(o; delay = 5).spiketimes == p2.spiketimes .+ 5.0f0
        @test_throws ArgumentError spiketime_parameter((ft = [0.1], neurons = [[-1]]))
        @test_throws ArgumentError spiketime_parameter((ft = [0.1], neurons = [[0]]))
        # smoke test: all spikes are delivered by a SpikeTimeStimulus
        E = SNNModels.IF(N = 10)
        stim = SNNModels.SpikeTimeStimulus(E, :ge; param = spiketime_parameter(coch1),
                                           conn = (p = 1.0, μ = 1.0), N = 35)
        model = SNNModels.compose(; E, stim, silent = true)
        SNNModels.sim!(; model, duration = 1000.0f0)
        @test stim.next_index[1] == -1
    end

    @testset "ms_intervals / label_intervals" begin
        t = df.words[3]                                # that, had, that
        @test ms_intervals(t) == [Float32[Float32(a * 1000), Float32(b * 1000)] for (a, b) in t.intervals]
        @test ms_intervals(t) isa Vector{Vector{Float32}}
        that = label_intervals(t, "that")
        @test that == [Float32[100, 300], Float32[500, 700]]
        @test that == [Float32[Float32(a * 1000), Float32(b * 1000)] for (a, b) in t.intervals[[1, 3]]]
        @test label_intervals(t, :that) == that
        @test label_intervals(t, "absent") == Vector{Float32}[]
        @test label_intervals(t, "absent") isa Vector{Vector{Float32}}
    end

    @testset "to_sequence" begin
        _, trans = get_ordered_spikes(wi; silence_time = 0.1, shift = 0.2, repetitions = 2,
                                      rng = MersenneTwister(1))
        seq = to_sequence(trans; tail = 0.3)
        @test keys(seq) == (:dict, :symbols, :ph_duration, :silence, :sequence, :line_id)
        @test seq.line_id == (words = 1, phonemes = 2, duration = 3, type = 4, onset = 5, offset = 6)
        @test seq.silence == :_ && size(seq.sequence, 1) == 6
        @test seq.symbols.words == [:had, :that]
        @test seq.symbols.phonemes == [:ae, :d, :dh, :hh, :t]
        @test seq.dict[:that] == [:dh, :ae, :t]
        # "had" has 2 phones in fabc0/sa1 and 3 in mdef0/sa1: the first occurrence decides
        first_had = findfirst(==("had"), trans.words.signs)
        (h0, h1) = trans.words.intervals[first_had]
        @test seq.dict[:had] == [Symbol(p) for ((a, b), p) in zip(trans.phones.intervals, trans.phones.signs)
                                 if a >= h0 && b <= h1]
        @test seq.dict isa Dict{Symbol,Vector{Symbol}} && seq.ph_duration isa Dict{Symbol,Float32}
        @test haskey(seq.ph_duration, :_) && all(>(0), values(seq.ph_duration))
        approx_intervals(a, b) = length(a) == length(b) && all(all(isapprox.(x, y; atol = 1e-3)) for (x, y) in zip(a, b))
        for w in ("that", "had")
            @test approx_intervals(SNNUtils.merge_intervals(SNNUtils.sign_intervals(Symbol(w), seq)),
                                   label_intervals(trans.words, w))
        end
        for ph in ("dh", "hh", "t")
            @test approx_intervals(SNNUtils.merge_intervals(SNNUtils.sign_intervals(Symbol(ph), seq)),
                                   label_intervals(trans.phones, ph))
        end
        @test isapprox(SNNUtils.sequence_end(seq), 1000 * (trans.words.intervals[end][2] + 0.3); atol = 1e-3)
        @test :w_that in SNNUtils.stimuli_names(seq).words && :p_dh in SNNUtils.stimuli_names(seq).phonemes
        S = seq.sequence
        @test S[5, 1] == 0.0f0 && S[6, end] ≈ sum(S[3, :]) && all(S[5, 2:end] .== S[6, 1:(end - 1)])
        @test all(x -> x isa Float32, S[3, :]) && all(x -> x isa Float32, S[5, :])
        # type row: one run per presentation, :onset first, :offset last
        runs = Tuple{Int,Int}[]
        i = 1
        while i <= size(S, 2)
            if S[1, i] != :_
                j = i
                while j < size(S, 2) && S[1, j + 1] == S[1, i]
                    j += 1
                end
                push!(runs, (i, j))
                i = j + 1
            else
                i += 1
            end
        end
        @test length(runs) == length(trans.words)
        @test all(S[4, a] == :onset && S[4, b] == :offset for (a, b) in runs)
        @test all(S[4, b - 1] == :offset_2 for (a, b) in runs if b - a == 2)   # 3-phone words
        @test any(b - a == 2 for (a, b) in runs) && any(b - a == 1 for (a, b) in runs)
        @test all(S[4, i] == :silence for i in axes(S, 2) if S[2, i] == :_)
        # same word twice with no silence: two presentations
        _, t2 = get_ordered_spikes(wi; silence_time = 0.0, shift = 0.0, ids = [1, 1])
        seq2 = to_sequence(t2)
        @test count(==(:onset), seq2.sequence[4, :]) == 2
        @test seq2.sequence[1, :] == [:_, :that, :that, :that, :_, :that, :that, :that, :_]
        @test seq2.sequence[3, 5] == 0.0f0 && seq2.sequence[3, end] == 0.0f0
        # update_stimuli! accepts the sequence
        E = SNNModels.IF(N = 10)
        stim = SNNUtils.step_input(; inputs = SNNUtils.stimuli_names(seq).all, network = (pop = (Exc = E,),),
                                   sym = :ge, p_post = 0.5, peak_rate = 0.01, proj_strength = 1.0)
        @test SNNUtils.update_stimuli!(; seq, model = (; stim)) == (; stim)
        # words with silent gaps between phones
        tg = (words = Transcription([(0.1, 0.4)], ["w"]),
              phones = Transcription([(0.15, 0.2), (0.25, 0.3)], ["a", "b"]))
        sg = to_sequence(tg; tail = 0.05).sequence
        @test sg[1, :] == [:_, :w, :w, :w, :w, :w, :_]
        @test sg[2, :] == [:_, :_, :a, :_, :b, :_, :_]
        @test isapprox(collect(sg[3, :]), Float32[100, 50, 50, 50, 50, 100, 50]; atol = 1e-3)
        # errors
        bad(w, p) = (words = Transcription(w, fill("x", length(w))), phones = Transcription(p, fill("p", length(p))))
        @test_throws ArgumentError to_sequence(bad([(0.0, 0.1)], [(0.05, 0.15)]))              # phone crosses the offset
        @test_throws ArgumentError to_sequence(bad([(0.0, 0.2), (0.1, 0.3)], NTuple{2,Float64}[])) # overlapping words
        @test_throws ArgumentError to_sequence(bad([(0.2, 0.3), (0.0, 0.1)], NTuple{2,Float64}[])) # unsorted words
        @test_throws ArgumentError to_sequence(bad([(0.0, 0.1)], [(0.2, 0.3)]))                # phone after every word
        @test_throws ArgumentError to_sequence(bad([(0.2, 0.3)], [(0.0, 0.1)]))                # phone before the word
        @test_throws ArgumentError to_sequence(bad([(0.0, 0.2)], [(0.1, 0.2), (0.0, 0.1)]))    # unsorted phones
        @test_throws ArgumentError to_sequence(bad([(0.0, 0.1)], NTuple{2,Float64}[]); tail = -1.0)
        # a word spelled like a phone label: warning
        tc = (words = Transcription([(0.0, 0.1)], ["ah"]), phones = Transcription([(0.0, 0.1)], ["ah"]))
        @test_logs (:warn, r"ah") to_sequence(tc)
        @test to_sequence(Transcriptions{Float64}((Transcription(), Transcription()))).sequence[1, :] == [:_]
    end

    @testset "real data (optional)" begin
        if haskey(ENV, "SPIKETIMIT_PATH") && isdir(ENV["SPIKETIMIT_PATH"]) &&
           ENV["SPIKETIMIT_PATH"] != FX.root
            dfr = create_dataset(; dir = dataset_path("train"))
            @test nrow(dfr) > 0
            @test length(read_spiketimes(dfr.path[1], BAE())) == 620
        else
            @info "SpikeTimit: SPIKETIMIT_PATH not set, real-data tests skipped"
        end
    end
end
