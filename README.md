# Structural Aliasing in Patch-Based Time-Series Forecasting

Coursework project for the *Computer Science and Digital Technologies* programme (University of Brescia, 2025-26). This repository formalises, empirically validates and proposes mitigations for **structural aliasing** in patch-based time-series foundation models, using Amazon Chronos-Bolt as the target architecture.

## What is structural aliasing?

Patch-based tokenisation splits a time series into windows of size P cut every S samples. When a signal completes a whole number of cycles in one stride, f = c·fs/S, consecutive patches are identical; when it completes a whole number in one patch, f = k·fs/P, the token sequence repeats every P/gcd(P,S) tokens. The union of the two combs is the closed-form candidate set F_lock. For S ≤ P the raw token sequence still determines the signal, so a candidate frequency is a place where a loss could sit, not a proof of one. Structural aliasing is the representation-level proposition that a model trained on such sequences nevertheless fails to make those frequencies readable from its internal state.

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
│       ├── Dataset-SolarTechLab.csv   # third-party dataset (CC BY), see notebooks/data/README.md
│       └── README.md                  # source, licence and attribution of the dataset
├── coursework/
│   ├── deliverable1_v0/, deliverable1_v1/, deliverable2_v0/   # submitted deliverables (frozen)
│   └── deliverable3/             # current report: main.tex, sections/, tables/, figures/
├── pyproject.toml
└── LICENSE                       # MIT (code); other material: see Licensing
```

Which notebook produces which part of the report:

| Report part | Notebook |
|---|---|
| Figure 1 and Appendix F sweep | `signal_analysis/single_reconstruction_figures.ipynb`, `signal_analysis/appendix_reconstruction.ipynb` |
| Table 5 and Appendix G | `signal_analysis/appendix_decomposition_comparison.ipynb` |
| Table 6 and Appendix H | `signal_analysis/solar_telemetry_comparison.ipynb` |
| Section 5.4, Appendix E, Tables 7-8 and Figure 2 | `bayesian/deliverable3_bayesian_models.ipynb` |
| Figure 4 (Appendix E) | `bayesian/figure_ppc_dispersion.py`, from the `tables/05_ppc.csv` the notebook writes |
| Appendix B | `chronos_architecture/train_sweep.py` |

## Hypotheses

| ID | Claim | Test |
|----|-------|------|
| **H1** | Candidate frequencies suffer a localised information loss | Model A: recovery at a candidate relative to its controls, exp(beta_bar) < 0.8 (behavioural); Model B: probe codelength expansion exp(theta_lock) > 1.2 (representational) |
| **H2** | The deficit is phase-invariant | Model C: per-phase offset scale sigma_phi < log 1.1 |
| **H3** | Dips sit on both predicted grids | Model D1: theta_S < 0 and theta_P < 0 jointly, and the two-label fit wins leave-one-out |
| **H3a** | Stride-branch sites track fs/S | Model D2: \|kappa_S - 1\| < 0.1 |
| **H3b** | Patch-branch sites track fs/P | Model D2: \|kappa_P - 1\| < 0.1, once at least ten unambiguous sites exist |
| **M1** | More overlap mitigates the loss | Model A': overlap slope delta_O > 0 and the overlap term wins leave-one-out |

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

Then open `notebooks/bayesian/deliverable3_bayesian_models.ipynb` for the PyMC inference. See the [Bayesian guide](notebooks/bayesian/README.md) for its setup and execution modes.

## Status

The full Bayesian run is complete and reported in Section 5.4 and Appendix E of Deliverable 3. Every fit
converged, but none passes the pre-registered posterior predictive check and Models A to C also fail
parameter recovery, so under the rules fixed before any posterior was read every claim is not reportable
and H3b is not identified. Before the gates the readings point away from a blind spot: a tone at a
candidate frequency is recovered better than at its controls, and the stride sites move as fs/S predicts.

## Licensing

| Material | Licence |
|---|---|
| Written report (LaTeX source and PDF, `coursework/deliverable3/`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Ontology (`coursework/deliverable3/ontology.ttl`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Result tables (the LaTeX tables in `coursework/deliverable3/tables/` and the CSV result tables the notebooks write) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Source code, probing scripts, notebooks and software artefacts | [MIT License](LICENSE) |

These licences apply only to material produced by the authors. Third-party material keeps its own licence or terms of use and is attributed below.

### Third-party material

- **Photovoltaic dataset.** `notebooks/data/Dataset-SolarTechLab.csv` is a copy of *Photovoltaic Power and Weather Parameters*, measured at the SolarTech Lab, Politecnico di Milano, by S. Leva, A. Nespoli, S. Pretto, M. Mussetta and E. Ogliari (IEEE DataPort, 23 September 2020, [doi:10.21227/42V0-JZ14](https://doi.org/10.21227/42V0-JZ14)). It is an open-access dataset released under a Creative Commons Attribution (CC BY) licence and is redistributed here only so that the experiments can be reproduced; any reuse must credit the original authors and cite the DOI. Details in [`notebooks/data/README.md`](notebooks/data/README.md).
- **LaTeX template.** The report layout is the *Journal Article* template by Vel ([LaTeXTemplates.com](https://www.LaTeXTemplates.com)), licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); the licence header is kept in `main.tex` and `LTJournalArticle.cls`.
- **Chronos models and training corpora.** The Chronos-Bolt-Tiny architecture and the Chronos training datasets are used under their own licences and terms of use, cited in the report.

## Authors

- **Matteo Boniotti** — University of Brescia
- **Gianluca Brignoli** — University of Brescia
- **Federico Sabbadini** — University of Brescia

## Acknowledgements

The use of AI-based tools in this work, which tools, for which parts and to what extent, is declared in the title footnote of the Deliverable 3 report. All generated content was reviewed, revised and validated by the authors.
