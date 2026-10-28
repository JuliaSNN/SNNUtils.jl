# Synthetic Spike TIMIT corpus used by the SpikeTimit tests.
#
# Everything here is invented: no TIMIT/LDC text, audio or spikes. The files follow the on-disk
# formats of the real corpus (lowercase layout, `.wrd`/`.phn` records "t0 t1 label" in samples at
# 16 kHz, `DOC/TIMITDIC.TXT`). `make_fixture` writes ONLY under the directory it is given.

"""Sample rate (Hz) of the sample indices written in the fixture transcriptions."""
const FIXTURE_SR = 16000.0

"""
Sentences of the fixture: path below the root (without extension), split, speaker, sentence id,
dialect, gender, word records and phone records `(t0, t1, label)` in samples.
"""
const FIXTURE_SENTENCES = [
    (path = joinpath("train", "dr1", "fabc0", "sa1"), split = "train", speaker = "fabc0",
     senID = "sa1", dialect = 1, gender = 'f',
     words = [(1600, 4800, "she"), (4800, 8000, "had"), (9600, 12800, "that")],
     phones = [(0, 1600, "h#"), (1600, 3200, "sh"), (3200, 4800, "iy"), (4800, 6400, "hh"),
               (6400, 8000, "ae"), (8000, 9600, "pau"), (9600, 11200, "dh"),
               (11200, 12000, "ae"), (12000, 12800, "t"), (12800, 14400, "h#")]),
    (path = joinpath("train", "dr1", "fabc0", "sx2"), split = "train", speaker = "fabc0",
     senID = "sx2", dialect = 1, gender = 'f',
     words = [(800, 4000, "that"), (4800, 8000, "she")],
     phones = [(0, 800, "h#"), (800, 2400, "dh"), (2400, 3200, "ae"), (3200, 4000, "t"),
               (4000, 4800, "pau"), (4800, 6400, "sh"), (6400, 8000, "iy"), (8000, 8800, "h#")]),
    (path = joinpath("train", "dr2", "mdef0", "sa1"), split = "train", speaker = "mdef0",
     senID = "sa1", dialect = 2, gender = 'm',
     words = [(1600, 4800, "that"), (4800, 8000, "had"), (8000, 11200, "that")],
     phones = [(0, 1600, "h#"), (1600, 3200, "dh"), (3200, 4000, "ae"), (4000, 4800, "t"),
               (4800, 6400, "hh"), (6400, 7200, "ae"), (7200, 8000, "d"), (8000, 9600, "dh"),
               (9600, 10400, "ae"), (10400, 11200, "t"), (11200, 12800, "h#")]),
    (path = joinpath("test", "dr3", "fghi0", "si3"), split = "test", speaker = "fghi0",
     senID = "si3", dialect = 3, gender = 'f',
     words = [(0, 3200, "she")],
     phones = [(0, 1600, "sh"), (1600, 3200, "iy")]),
]

"""Content of the fixture `DOC/TIMITDIC.TXT`."""
const FIXTURE_DICTIONARY_TEXT = """
; synthetic dictionary for tests
;
a  /ax/
had  /hh ae1 d/
she  /sh iy1/
that  /dh ae1 t/

"""

"""Dictionary expected from `create_dictionary` on the fixture."""
const FIXTURE_DICTIONARY = Dict("a" => "ax", "had" => "hh ae1 d", "she" => "sh iy1",
                                "that" => "dh ae1 t")

"""
    fixture_spikes(nch, T)

Deterministic spike trains (s): channels 1-10 fire every 50 ms from `12.5 ms * c` up to `T` s,
the other channels are silent.
"""
fixture_spikes(nch::Int, T::Real) = [c <= 10 ? collect(0.0125 * c:0.05:T) : Float64[] for c in 1:nch]

"""Duration (s) of a fixture sentence: last phone offset / sample rate."""
fixture_duration(s) = s.phones[end][2] / FIXTURE_SR

function _write_records(file, records; trailing_blank::Bool = false)
    open(file, "w") do io
        for (t0, t1, label) in records
            print(io, "$t0 $t1 $label\n")
        end
        trailing_blank && print(io, "\n")
    end
end

"""
    cochlea_array(spikes) -> Matrix{Float64}

2 x K array of the cochlea `.npz` files: columns `(t, c - 1)` ordered by channel, then time.
"""
cochlea_array(spikes) =
    reduce(hcat, [[t, Float64(c - 1)] for (c, ch) in enumerate(spikes) for t in ch];
           init = zeros(2, 0))

"""
    bae_raw(T) -> Vector{Vector{Float64}}

Raw BAE spike times (time / 5) of a sentence of duration `T` s; channel 3 has one spike.
"""
function bae_raw(T::Real)
    raw = [x ./ 5 for x in fixture_spikes(620, T)]
    raw[3] = [0.01]
    return raw
end

"""
    bae_cell(raw) -> Matrix{Any}

1 x n MATLAB cell array: `zeros(0, 0)` for empty channels, a scalar for one spike, a k x 1
column otherwise, except channel 2 stored as a 1 x k row.
"""
function bae_cell(raw)
    cell = Matrix{Any}(undef, 1, length(raw))
    for (c, x) in enumerate(raw)
        cell[1, c] = isempty(x) ? zeros(0, 0) :
                     length(x) == 1 ? x[1] :
                     c == 2 ? reshape(x, 1, :) : reshape(x, :, 1)
    end
    return cell
end

"""
    make_fixture(root; mat::Bool) -> NamedTuple

Write the synthetic corpus under `root` and return the expected values used by the tests.
Malformed spike files go to `joinpath(dirname(root), "bad")`, outside the corpus. BAE `.mat`
files are written only with `mat = true` (requires `using MAT`).
"""
function make_fixture(root::AbstractString; mat::Bool)
    mkpath(root)
    for s in FIXTURE_SENTENCES
        base = joinpath(root, s.path)
        mkpath(dirname(base))
        touch(base * ".wav")
        _write_records(base * ".wrd", s.words; trailing_blank = s.senID == "sx2")
        _write_records(base * ".phn", s.phones)
    end
    write(joinpath(root, "train", "dr1", "fabc0", "notes.txt"), "not a sentence\n")
    mkpath(joinpath(root, "DOC"))
    write(joinpath(root, "DOC", "TIMITDIC.TXT"), FIXTURE_DICTIONARY_TEXT)
    # spike files
    raw_bae = Dict{String,Vector{Vector{Float64}}}()
    for s in FIXTURE_SENTENCES
        base = joinpath(root, s.path)
        T = fixture_duration(s)
        for n in (35, 70)
            NPZ.npzwrite(base * "_cochlea$(n).npz", Dict("arr_0" => cochlea_array(fixture_spikes(n, T))))
        end
        if mat
            raw_bae[base] = bae_raw(T)
            MAT.matwrite(base * "_spike_timing.mat", Dict("spike_output" => bae_cell(raw_bae[base])))
        end
    end
    bad = joinpath(dirname(root), "bad")
    mkpath(bad)
    NPZ.npzwrite(joinpath(bad, "x_cochlea35.npz"), Dict("arr_0" => zeros(3, 4)))
    NPZ.npzwrite(joinpath(bad, "y_cochlea35.npz"), Dict("arr_0" => [0.1 0.2; 0.0 35.0]))
    NPZ.npzwrite(joinpath(bad, "u_cochlea35.npz"), Dict("arr_0" => [0.1 0.2; 0.0 1.5]))
    NPZ.npzwrite(joinpath(bad, "v_cochlea35.npz"), Dict("other" => [0.1 0.2; 0.0 1.0]))
    if mat
        MAT.matwrite(joinpath(bad, "z_spike_timing.mat"), Dict("spike_output" => bae_cell(bae_raw(0.5)[1:10])))
        MAT.matwrite(joinpath(bad, "w_spike_timing.mat"), Dict("other" => bae_cell(bae_raw(0.5))))
        cell = bae_cell(bae_raw(0.5))
        cell[1, 5] = "x"
        MAT.matwrite(joinpath(bad, "s_spike_timing.mat"), Dict("spike_output" => cell))
    end
    return (root = root, bad = bad, sentences = FIXTURE_SENTENCES, dictionary = FIXTURE_DICTIONARY,
            fixture_spikes = fixture_spikes, sr = FIXTURE_SR, raw_bae = raw_bae)
end
