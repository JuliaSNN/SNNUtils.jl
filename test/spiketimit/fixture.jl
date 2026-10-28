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
    make_fixture(root; mat::Bool) -> NamedTuple

Write the synthetic corpus under `root` and return the expected values used by the tests.
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
    return (root = root, sentences = FIXTURE_SENTENCES, dictionary = FIXTURE_DICTIONARY,
            fixture_spikes = fixture_spikes, sr = FIXTURE_SR)
end
