# timit_loader

Python loaders for the TIMIT / Spike TIMIT corpus, moved here from the SpikeTimit.jl repository
(`TIMIT_loader/`). They build a pandas DataFrame of the sentences and compute CQT spectra per
word and phone with librosa; these spectra are the input of the spectra example of
`SNNUtils.SpikeTimit` (loaded through PythonCall). Julia never imports or runs these files.

## Requirements

Python >= 3.8 with pandas, numpy, librosa, scikit-learn and nltk. `timit_torch.py` additionally
imports torch and torchaudio.

## Usage

Set the corpus location (the directory that contains `train/`, `test/` and `DOC/`) and import the
package from the parent directory:

```bash
export SPIKETIMIT_PATH=/path/to/Spike_TIMIT
cd tools && python -c "import os, timit_loader as T; df = T.create_dataset(os.path.join(os.environ['SPIKETIMIT_PATH'], 'train'))"
```

The notebooks read `os.environ["SPIKETIMIT_PATH"]`; their outputs were removed.

## Status of the files

| File | Status |
|---|---|
| `timit_parser.py` | working: dataset DataFrame, speaker data, word/phone times, `get_spectra` |
| `load_wav.py` | working: wav reading and normalisation |
| `__init__.py` | re-exports `timit_parser` |
| `timit_torch.py` | exploratory: executes code at import and references undefined names; does not run |
| `cochlea_spikes.py` | incomplete copy of the dataset walker; NOT the generator of the `*_cochlea*.npz` files |
| `setup.py` | template metadata (author, description, URL): to be confirmed |
| `TIMIT_usage.ipynb`, `audio_spectra.ipynb` | exploration notebooks, outputs stripped |

## Licence

- Code: the original repository has no licence statement and `setup.py` is an unfilled template
  (its MIT classifier comes from the template). These files are covered by the SNNUtils MIT
  licence only after the author(s) confirm it; metadata to be confirmed.
- Data: TIMIT is distributed by the LDC (LDC93S1) under its licence, and Spike TIMIT follows the
  same terms. Never commit corpus files, derived figures or notebook outputs to this repository.
