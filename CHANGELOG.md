# SNNUtils changelog

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
