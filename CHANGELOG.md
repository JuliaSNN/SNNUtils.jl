# SNNUtils changelog

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
