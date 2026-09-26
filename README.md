# Structural Aliasing in Patch-Based Time-Series Forecasting

Coursework project for the *Computer Science and Digital Technologies* programme (University of Brescia, 2025-26). This repository formalises, empirically validates and proposes mitigations for **structural aliasing** in patch-based time-series foundation models, using Amazon Chronos-Bolt as the target architecture.

## What is structural aliasing?

Patch-based tokenisation splits a time series into overlapping windows of size P with stride S. When a signal's frequency satisfies the phase-locking condition f = c·fs/S or f = k·fs/P, consecutive patches become identical and the model's representation degenerates. The set of such frequencies is F_lock. Structural aliasing is the empirical phenomenon in which this degeneracy propagates to the learned latent space, making distinct inputs indistinguishable.

## Repository layout

```
patchAliasing/
├── notebooks/                    # code, notebooks and data behind Deliverable 3
│   ├── chronos_architecture/     # architecture notebook, training sweep (train_sweep.py), model upload
│   ├── bayesian/                 # Bayesian notebook, its README and the implementation report
│   ├── signal_analysis/          # sweep (App. F), decomposition (App. G) and solar (App. H) notebooks
│   ├── support_scripts/          # probe_lib, collect, comparison and model-loading modules
│   └── data/
│       ├── synthetic/            # Light TSMixup and KernelSynth generators
│       └── Dataset-SolarTechLab.csv
├── coursework/
│   ├── deliverable1_v0/, deliverable1_v1/, deliverable2_v0/   # submitted deliverables (frozen)
│   └── deliverable3/             # current report: main.tex, sections/, tables/, figures/
├── pyproject.toml
└── LICENSE                       # MIT
```

Which notebook produces which part of the report:

| Report part | Notebook |
|---|---|
| Figure 1 and Appendix F sweep | `signal_analysis/single_reconstruction_figures.ipynb`, `signal_analysis/appendix_reconstruction.ipynb` |
| Table 5 and Appendix G | `signal_analysis/appendix_decomposition_comparison.ipynb` |
| Table 6 and Appendix H | `signal_analysis/solar_telemetry_comparison.ipynb` |
| Section 5.4, Appendix E, Table 7 | `bayesian/deliverable2_bayesian_models.ipynb` |
| Appendix B | `chronos_architecture/train_sweep.py` |

## Hypotheses

| ID | Claim | Test |
|----|-------|------|
| **H1** | Locked frequencies suffer localised information loss | Paired log-contrast d < 0 (behavioural) + higher MDL codelength (representational) |
| **H2** | The deficit is phase-invariant | Per-phase offset spread sigma_phi near zero |
| **H3a** | Stride-branch sites track fs/S | Detected collapse sites move with 1/S at fixed P |
| **H3b** | Patch-branch sites track fs/P | Detected collapse sites move with 1/P at fixed S |

## Models

All models are Chronos-Bolt-Tiny retrained from scratch (random init) on the official Chronos pre-training data (TSMixup 10M + KernelSynth 1M at 9:1 ratio), with a uniform 100k-step budget and seed 42. Only (P, S) varies. Checkpoints are available at [`federicosabbadini/chronos-bolt-patch-sweep`](https://huggingface.co/federicosabbadini/chronos-bolt-patch-sweep).

## Setup

Requires Python 3.11-3.12.

```bash
uv sync
```

### Running the experiments

**Bayesian data collection**:
```bash
cd notebooks
python support_scripts/collect.py --out ./results --smoke   # pipeline check
python support_scripts/collect.py --out ./results           # full design
```

Then open `notebooks/bayesian/deliverable2_bayesian_models.ipynb` for the PyMC inference. See the [Bayesian guide](notebooks/bayesian/README.md) for its setup and execution modes.

## Status

The theory, the design and the descriptive readings are in the report. The hypotheses are decided only by
the Bayesian notebook, under the rules of Deliverable 2 fixed before any posterior is read; until its full
run is complete, no hypothesis is reported as supported or refuted.

## Licensing

The written report is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All source code, probing scripts and software artifacts are released under the [MIT License](LICENSE).

## Authors

- **Matteo Boniotti** — University of Brescia
- **Gianluca Brignoli** — University of Brescia
- **Federico Sabbadini** — University of Brescia

## Acknowledgements

During the preparation of this work, the authors used AI-based tools (GPT-5.5, Gemini 3.6 Flash, Sonnet 5, Opus 4.6) to support language refinement and improve clarity. All generated content was reviewed, revised and validated by the authors.
