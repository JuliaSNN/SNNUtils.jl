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
