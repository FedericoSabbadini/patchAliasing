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

The SV probing entry points are the [pure-tone sweep](notebooks/signal_analysis/probing_pure_tones.ipynb)
and the [KS/TSM sweep](notebooks/signal_analysis/probing_complex_signals.ipynb).
Both read ten stages (four encoder blocks, four decoder blocks, the input of the output projection and the
output quantile head), with a shuffled-label and an untrained-network control, and add a lock/control reading
at the candidate sites. The pure-tone sweep produces the retrained 16–16 SV figure and the tables of Appendix F;
the KS/TSM sweep exports the median and quartiles of SV for each checkpoint, background family and task.
The notebooks write run results under `notebooks/_run/probing/` and, in full mode, copy the values quoted in the
report to `notebooks/signal_analysis/results/probing/`. The probing values of the report come from these files,
not from `02_mdl_bandtasks.parquet` of Model B, whose blocks are not grouped by frequency.

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

**SV probing** (run from the repository root):

```bash
uv pip install -r notebooks/support_scripts/requirements-probing.txt
python notebooks/support_scripts/run_probing.py --kind both --mode smoke --device cpu --root notebooks/_run/probing
python notebooks/support_scripts/run_probing.py --kind both --mode full --device cpu --root notebooks/_run/probing
```

To save executed notebook outputs as well, use
`python notebooks/support_scripts/execute_probing_notebooks.py --mode full`.
It uses the current Python environment, checkpoints completed cells, and saves
the canonical notebooks after execution. With `--mode smoke`, it saves labelled
validation copies and HTML previews under `notebooks/_run/probing/validation/`.
The default executor captures rich outputs through IPython without local TCP
sockets. Use `--engine nbclient` for a separate Jupyter kernel where available.

The full run compares the published checkpoint and all fifteen retrained
geometries at decoder blocks 0–3 and the output head. The pure-tone sweep uses
2–250 Hz at 1-Hz steps and ten distinct phases. For KS and Light TSMixup,
each of 100 fixed backgrounds per family receives the same labelled sweep,
giving one SV per background, model, stage and task. The main table counts
`SV > 0` at the output head for the seven hierarchical tasks, with an exact
denominator of 100 in every cell.

Five prequential blocks keep all phases of one frequency together. Scaler,
PCA and logistic regression are fitted on preceding blocks only; the first
block costs one bit per label. Thus `SV = 1 - L_bits/n`, negative SV values
are retained, and the initial uniform block limits the attainable SV to about
0.8. A shuffled-label control accompanies the true task. This is the
repository's grouped prequential protocol, with a figure layout inspired by
[Pagani et al., Figure 3](https://arxiv.org/html/2605.06361v1).

Both notebooks default to the full design. To execute a notebook as a small
pipeline check, set `PATCHALIASING_PROBING_MODE=smoke` in its kernel environment
before running it from top to bottom. Smoke mode uses the retrained 16–16,
two backgrounds per family, two phases and a 4-Hz step; its outputs are marked
as smoke results and cannot supply the report's 100-background table. The CLI
also accepts `--models` in smoke mode to select checkpoint tags for a bounded diagnostic.
The CPU execution processes one checkpoint at a time; atomic, fingerprinted
shards allow an interrupted run to resume with the same command. Configuration,
input, code and checkpoint changes select a different run identity.

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
