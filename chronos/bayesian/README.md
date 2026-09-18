# Bayesian workspace

## Layout

```text
notebooks/
  models/          Bayesian inference, variants and diagnostics
  appendixes/      Appendix figures and representation probes
  experiments/     Solar telemetry benchmarks and complex-signal sweeps
support_scripts/   Shared Python modules and command-line runners
tests/             Unit tests, synthetic validation and A3 diagnostic builders
_run/              Existing runs, checkpoints and figures (unchanged location)
results/           Existing collected data (unchanged location)
coordination/      Historical coordination material
```

The reorganisation preserves notebook outputs and local edits. It does not select a
scientifically validated model or rerun inference. Version names describe variants,
not successful validation. Validation and experiment notes remain in this directory.

## Bayesian notebooks (`notebooks/models`)

| Notebook | Purpose |
|---|---|
| [bayesian_analysis](notebooks/models/bayesian_analysis.ipynb) | Integrated structural-aliasing Bayesian workflow: collection, models, diagnostics and decisions. |
| [new_analysis](notebooks/models/new_analysis.ipynb) | Streaming analysis of saved posterior checkpoints; does not produce missing fits. |
| [localisation_analysis](notebooks/models/localisation_analysis.ipynb) | Integrated workflow using frequency hit/miss localisation. |
| [localisation_analysis_gpu](notebooks/models/localisation_analysis_gpu.ipynb) | GPU-oriented variant of the localisation workflow. |
| [h1_fixed_offset_analysis](notebooks/models/h1_fixed_offset_analysis.ipynb) | H1 paired lock/control analysis with fixed ±1 Hz offsets and adaptive-control comparison; synthetic smoke by default. |
| [model_A_convergence_pilot_colab](notebooks/models/model_A_convergence_pilot_colab.ipynb) | Isolated convergence pilot for Model A. |
| [model_A_complete_colab](notebooks/models/model_A_complete_colab.ipynb) | Completion and validation of Model A using existing full-run checkpoints. |
| [model_A_freq_standalone_colab](notebooks/models/model_A_freq_standalone_colab.ipynb) | Frequency-localisation quality analysis replacing amplitude recovery. |
| [model_A_freq_standalone_colab_v3](notebooks/models/model_A_freq_standalone_colab_v3.ipynb) | Student-t model of the lock/control quality difference with non-centred effects. |
| [model_A_localisation_standalone_colab](notebooks/models/model_A_localisation_standalone_colab.ipynb) | Standalone hierarchical Bernoulli hit/miss model (A1). |
| [model_A2_localisation_standalone_colab](notebooks/models/model_A2_localisation_standalone_colab.ipynb) | Geometry-specific lock effects, separate lower/upper controls and background-only calibration. |
| [model_A2_localisation_standalone_colab_improved](notebooks/models/model_A2_localisation_standalone_colab_improved.ipynb) | A2 variant with revised validation. |
| [model_A3_localisation_standalone_colab](notebooks/models/model_A3_localisation_standalone_colab.ipynb) | Local effects by geometry and frequency, with per-chain checkpoints. |
| [model_A3_reduced_localisation_standalone_colab](notebooks/models/model_A3_reduced_localisation_standalone_colab.ipynb) | Reduced A3 without the local baseline residual; retained for diagnosis. |
| [model_A3_reduced_ncp_localisation_standalone_colab](notebooks/models/model_A3_reduced_ncp_localisation_standalone_colab.ipynb) | Same reduced model with non-centred geometry coordinates for sampler diagnosis. |
| [model_D2_diagnostics_standalone_colab](notebooks/models/model_D2_diagnostics_standalone_colab.ipynb) | Diagnoses detected collapse sites and branch-identification gates from existing Parquet tables. |
| [models_D1_D2_full_profiles_standalone_colab](notebooks/models/models_D1_D2_full_profiles_standalone_colab.ipynb) | Original D1 and D2 inference from continuous collapse profiles. |
| [models_D1_D2_full_profiles_v2_standalone_colab](notebooks/models/models_D1_D2_full_profiles_v2_standalone_colab.ipynb) | Restores D1's original PPC and introduces profile-specific levels, coefficients and noise for D2. |

## Appendix notebooks (`notebooks/appendixes`)

| Notebook | Purpose |
|---|---|
| [appendix_decomposition](notebooks/appendixes/appendix_decomposition.ipynb) | Signal/component and spectral decomposition figures. |
| [appendix_decomposition_comparison](notebooks/appendixes/appendix_decomposition_comparison.ipynb) | Decomposition comparisons between observed signals, retrained and official models. |
| [appendix_reconstruction](notebooks/appendixes/appendix_reconstruction.ipynb) | Appendix reconstruction figures, including patch/stride frequency axes. |
| [reconstruction_figures](notebooks/appendixes/reconstruction_figures.ipynb) | Pure-sinusoid frequency reconstruction and spectral recovery sweeps. |
| [probing_space_saving](notebooks/appendixes/probing_space_saving.ipynb) | Representation probing and MDL space-saving figures, including Appendix F data. |

## Experiment notebooks (`notebooks/experiments`)

| Notebook | Purpose |
|---|---|
| [solar_telemetry](notebooks/experiments/solar_telemetry.ipynb) | Daily measured SolarTechLab telemetry viewer. |
| [solar_telemetry_benchmark_colab](notebooks/experiments/solar_telemetry_benchmark_colab.ipynb) | Standalone Colab benchmark of retrained geometries and the published baseline. |
| [solar_telemetry_comparison](notebooks/experiments/solar_telemetry_comparison.ipynb) | Daily views plus shared-window, resumable solar benchmark. |
| [solar_chronos2_comparison](notebooks/experiments/solar_chronos2_comparison.ipynb) | PV forecasts from multivariate Chronos-2 versus official and retrained Bolt 16×16. |
| [complex_signal_sweep](notebooks/experiments/complex_signal_sweep.ipynb) | Frequency probes and rollouts on multi-component synthetic signals. |

## Why the support scripts exist

All modules below live in `support_scripts/`. Keeping computation separate lets
notebooks focus on analysis and allows the same implementation to be tested and run
from a CLI.

| Script | Responsibility |
|---|---|
| `probe_lib.py` | Shared batched forecasts, frequency recovery, representation probing, MDL and collapse measurements. Avoids executing other notebooks to obtain measurements. |
| `collect.py` | Produces the observation tables consumed by Bayesian inference; separates expensive Chronos collection from statistical analysis. |
| `model_loader.py` | Resolves checkpoint identity and loads local or Hugging Face weights consistently. |
| `checkpointing.py` | Atomic writes, hashes and fingerprints for traceable, resumable runs. |
| `bayesian_checks.py` | Centralises convergence, identification and reportability gates. |
| `migrate_shards.py` | Verifies whether old collection shards can be reused in a new design before migrating them. |
| `h1_offset_lib.py` | Implements fixed-offset H1 design, collection, model fitting and validation independently of presentation. |
| `comparison_lib.py` | Common checkpoint registry, resumable inference and metrics for official/retrained comparisons. |
| `comparison_figures.py` | Component and spectrum plotting shared by comparison notebooks. |
| `solar_benchmark_lib.py` | Shared-window solar benchmark, quantile diagnostics and CPU worker orchestration. |
| `complex_sweep_lib.py` | Multi-component signal generation, binary probes and dominant-frequency sweeps. |
| `solar_chronos2_lib.py` | Past-covariate preparation and Chronos-2 versus Bolt comparison. |
| `run_comparison.py` | CLI runner for the original solar/recovery comparison tasks. |
| `run_extended_comparisons.py` | CLI runner for extended solar, complex-signal and Chronos-2 tasks, including timing benchmarks. |

`tests/` retains `test_*.py` plus the A3 development tools: `build_A3_reduced.py`
and `build_A3_ncp.py` generate diagnostic notebook variants;
`validate_A3_recovery.py` and `validate_A3_reduced.py` exercise synthetic recovery;
`diagnose_A3_sparse.py` compares sampler coordinates; `benchmark_A3_reduced.py`
measures validation cost. These belong to validation tooling, not runtime imports.

## Running

From the repository root:

```powershell
.venv/Scripts/python.exe chronos/bayesian/support_scripts/collect.py --out chronos/bayesian/results --smoke
.venv/Scripts/python.exe chronos/bayesian/support_scripts/run_comparison.py --task all --smoke
.venv/Scripts/python.exe chronos/bayesian/support_scripts/run_extended_comparisons.py solar --smoke
.venv/Scripts/python.exe -m unittest discover -s chronos/bayesian/tests -p "test_*comparison*.py" -v
```

Notebook setup locates the repository and adds `support_scripts` to the import path.
Colab workflows that clone the repository need a revision containing this layout.
Existing `_run` and `results` locations are retained. Source fingerprints change
when source paths or code change; existing compatibility gates remain active and
may reject an old checkpoint rather than silently resume it.

Full notebook execution is not part of this directory-only reorganisation: it can
download models, collect forecasts and run lengthy MCMC. Run the chosen notebook
top-to-bottom in its documented PILOT/SMOKE environment before a new full run.
