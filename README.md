# Structural Aliasing in Patch-Based Time-Series Forecasting

Coursework project for the *Computer Science and Digital Technologies* programme (University of Brescia, 2025-26). This repository formalises, empirically validates and proposes mitigations for **structural aliasing** in patch-based time-series foundation models, using Amazon Chronos-Bolt as the target architecture.

## What is structural aliasing?

Patch-based tokenisation splits a time series into overlapping windows of size P with stride S. When a signal's frequency satisfies the phase-locking condition f = c·fs/S or f = k·fs/P, consecutive patches become identical and the model's representation degenerates. The set of such frequencies is F_lock. Structural aliasing is the empirical phenomenon in which this degeneracy propagates to the learned latent space, making distinct inputs indistinguishable.

## Repository layout

```
patchAliasing/
├── chronos/                    # code, experiments and data
│   ├── models/                 # training sweep (22 geometries, 100k steps each)
│   │   ├── train_sweep.py      # from-scratch retraining driver
│   │   ├── upload_models.py    # push checkpoints to HuggingFace
│   │   └── weights/            # local checkpoints (gitignored)
│   ├── testing/                # hypothesis testing (Deliverable 1 workflow)
│   │   ├── testing_lib.py      # model loader and shared helpers
│   │   ├── hypotheses.py       # H1/H2/H3 tests (pure + background signals)
│   │   ├── run_all.py          # single-command reproducible run
│   │   └── chronosBolt_layer_probing.ipynb
│   ├── bayesian/               # Bayesian analysis (Deliverable 2 workflow)
│   │   ├── probe_lib.py        # batched probing core (22 geometries)
│   │   ├── collect.py          # data collection for the Bayesian models
│   │   ├── bayesian_analysis.ipynb
│   │   └── reconstruction_figures.ipynb
│   ├── data/
│   │   ├── synthetic/          # TSMixup and KernelSynth signal generators
│   │   └── dataset/            # real-world PV telemetry (SolarTechLab)
│   └── ontology/               # OWL2 DL semantic layer for the PV domain
│       ├── ontology.ttl        # OWL ontology (Turtle)
│       ├── pv_data.py          # Python adapter (concept risk, state classification)
│       └── pv_full_analysis.ipynb
├── coursework/                 # LaTeX deliverables and report
│   ├── _coursework_/           # current working report
│   │   ├── main.tex
│   │   └── sections/
│   ├── delivered/              # submitted deliverables (frozen)
│   │   ├── deliverable_1/      # formalisation, 5 models, probing, security
│   │   └── deliverable_2/      # extended analysis, Bayesian models, ontology
│   └── bayesian/               # standalone Bayesian coursework notebooks
├── pyproject.toml
└── LICENSE                     # MIT
```

Current report work should happen in `coursework/_coursework_`. The directories under `coursework/delivered` are submitted snapshots and should be treated as frozen references unless a comparison or packaging task explicitly targets them.

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

**Hypothesis tests** (Deliverable 1 workflow):
```bash
cd chronos/testing
python run_all.py              # full run: 5 models x {probing, pure, tsm, ks}
python run_all.py --smoke      # fast sanity check
```

**Bayesian data collection** (Deliverable 2 workflow):
```bash
cd chronos/bayesian
python -m collect --out ./results --smoke   # pipeline check
python -m collect --out ./results           # full design
```

Then open `bayesian_analysis.ipynb` for the PyMC inference.

## Key findings (provisional)

- **Representation penalised, forecast intact**: locked frequencies show higher MDL codelength (representational loss confirmed), but forecast amplitude recovery is not systematically attenuated.
- **Phase-invariant deficit** (H2 supported): the deficit does not depend on where in its cycle the signal starts.
- **Stride sites track fs/S** (H3a supported): collapse dips shift proportionally when the stride changes.
- **Patch branch not yet identified** (H3b): the scaling law requires more geometries with varying P at fixed S.
- **Overlap as mitigation**: shorter stride at fixed P raises the lowest stride-lock frequency and thins the in-band family, but cannot affect the patch branch.

## Licensing

The written report is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All source code, probing scripts and software artifacts are released under the [MIT License](LICENSE).

## Authors

- **Matteo Boniotti** — University of Brescia
- **Gianluca Brignoli** — University of Brescia
- **Federico Sabbadini** — University of Brescia

## Acknowledgements

During the preparation of this work, the authors used AI-based tools (GPT-5.5, Gemini 3.6 Flash, Sonnet 5, Opus 4.6) to support language refinement and improve clarity. All generated content was reviewed, revised and validated by the authors.
