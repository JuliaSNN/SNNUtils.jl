# SNNUtils changelog

## 1.1.0

New submodule `SNNUtils.SpikeTimit` (`using SNNUtils.SpikeTimit`): reader of the Spike TIMIT corpus,
migrated from the unregistered SpikeTimit.jl 0.1. No change to the 1.0 API; `SNNUtils` exports only
the new name `SpikeTimit`.

- Reads TIMIT transcriptions, the dictionary and the BAE (`.mat`) and cochlea (`.npz`) spike files;
  extracts, post-processes and stacks word inputs; converts them to SNNModels stimuli
  (`spiketime_parameter`, Float32 ms) and SNNUtils sequences (`to_sequence`, `label_intervals`).
  Corpus location from arguments or `ENV["SPIKETIMIT_PATH"]`; constants in `SpikeTimitConfig`.
- MAT is a weak dependency: the extension `SNNUtilsMATExt` (loaded by `using MAT`) reads the BAE files.
- The Python TIMIT tools of SpikeTimit.jl (`TIMIT_loader`) are not part of the package (unclear licence); they live with the examples in SNNExamples.
- Fixes with respect to SpikeTimit.jl: dialect parsed from the speaker's parent directory, not from any
  `dr*` component of the absolute path; `ft_dt` keeps all neurons of 3 or more equal time steps and no
  longer mutates its input; `get_word_spikes` requires `encoding` and works with a vector of words;
  `get_spikes_in_interval` dispatches on its argument types (single sample + intervals no longer returns
  garbage); `get_ordered_spikes` without `ids` no longer throws `UndefVarError`.
- Behaviour changes: `create_dataset` stores `words`/`phones` as `Transcription{Float64}` (`signs`,
  `intervals`) instead of n x 3 matrices; `select_words` samples sentences without replacement by default
  (`replace = true` restores the old rule) with an explicit `rng`; `get_ordered_spikes` takes
  `silence_time` and `shift` in seconds (the `unit` keyword is removed).
- Remove the old SpikeTimit.jl package from environments that load SNNUtils (name clash).

## 1.0.1

- `score_spikes` gets a `window = 100` keyword (ms): the counting window is `offset_time .+ (0:window) .+ delay`;
  `window <= 0` throws `ArgumentError`. The default reproduces the previous outputs.
- `score_spikes` records the spikes over `0:10ms:(maximum(offset_times) + max_delay + window)` (previously
  `0:10ms:(offset_times[end] + 100ms)`, which is the recording range, not the window). Old outputs change only when
  the counted bins extend past `offset_times[end] + 100`: delay > 10 ms (the bin with left edge `offset + 100` is the
  last one recorded before), or `delay = nothing`, which tests delays up to 100. The affected presentations are the
  last one(s) (those whose window end exceeds `offset_times[end] + 100`), whose window was truncated, i.e. the missing
  bins counted zero. Delays <= 10 ms, and any window ending before `offset_times[end] + 100`, are bit-identical.

## 1.0.0

Requires SNNModels 2 (compat `SNNModels = "2"`); first 1.x release (previous registered version 0.2.10).

- Weight analysis reads plastic weights through `plasticity_value`, and builds the synapse transpose arrays
  (`ensure_transpose!`) before reading `J`/`index`, which are lazy in SNNModels 2.
- Removed code that no module included (stale parameter sets and analysis files: `analysis/{EI_balance, base,
  import_data, membrane, performance, populations, protocols, rates}`, `models/{connections, dendrite_STM,
  duarte2019, lkd2014, models, mongillo_WM2008, quaresima2022, quaresima2023, quaresima_2024_updown,
  learning/structs}`, `stimuli/balance_EI/load`). They were not loaded in 0.2.10 either; restore from git
  history if needed.
- Dependencies removed: Dates, Interpolations, JLD2, Requires, Printf, Serialization, BSON, ThreadTools,
  Distributions, LinearAlgebra, Tables, ScientificTypes (unused).
