# bayesian/ — Bayesian analysis of structural aliasing

Implements the Bayesian section of Deliverable 1
(`coursework/delivered/deliverable1_v2/sections/deliverable1.tex`, §Bayesian Analysis), including
the H2 and H3 extension the lecturers asked for.

| file | what it is |
|---|---|
| `bayesian_analysis.ipynb` | **the** notebook — priors, observation, likelihood, posterior, checks |
| `probe_lib.py` | light, notebook-free probing core (forecast recovery, token collapse, `[REG]` capture, prequential MDL, lock geometry) |
| `collect.py` | runs the five geometries and writes five tidy tables; shards per model so a run resumes |

Nothing under `chronos/testing/` is modified. The model registry is *reused* from
`chronos/testing/testing_lib.py`, so there is still one place in the repo that maps `(P, S)` to a
checkpoint.

## The five models

| model | hypothesis | estimand | deliverable |
|---|---|---|---|
| **A** hierarchical Student-$t_4$ on the paired log contrast $d$ | H1 behavioural | $e^{\bar\beta}$ — recovery at a lock relative to its controls | Eq. (8)–(9) |
| **B** Gamma regression, log link, on the frequency-local MDL codelength | H1 representational | $e^{\theta_{lock}}$ — codelength expansion at a lock | Eq. (10) |
| **C** Model A + a first-harmonic circular term in the signal phase | H2 | $A_\phi$ — phase-modulation amplitude, with a ROPE and a LOO test | Eq. (11)–(12) |
| **D1** comb likelihood on the token-collapse profile | H3 location | LOO among $\Delta=f_s/S$, $\Delta=f_s/P$, no comb | Eq. (13) |
| **D2** regression of the detected comb spacing on $\log S^{-1}$, $\log P^{-1}$ | H3 movement | $\gamma_S$ — H3 predicts 1 | Eq. (14) |

## Geometries

`p16-s16` (official `amazon/chronos-bolt-tiny`), `p16-s12`, `p16-s8`, `p8-s8`, `p24-s24`.
`p16-s4` is excluded, matching `deliverable1.tex`: its stride-lock class has a single in-band
member, so it carries no local H1 contrast.

## Running it

The notebook is written for Colab. Open it, run Part 0, and go. Part 0 clones this repository,
mounts Google Drive for checkpoints (falling back to `/content` if you decline), and prints the
design.

**Every part checkpoints.** After a disconnect, re-run from the top: completed parts reload instead
of recomputing, so Part 2's Chronos work is paid for once. Because the parts communicate only
through files, they can run in different runtimes:

* **Part 2** wants a **GPU** runtime (it is the only part that loads a model);
* **Parts 1, 3, 4, 5** are **CPU**-only PyMC and detect the absence of `torch` automatically,
  reading the tables a previous GPU run wrote.

Set `SMOKE = True` in Part 0.3 for a few-minute pipeline check. Smoke output goes to its own
directory and every part prints a banner, so it cannot be confused with a reportable run.

## Collecting without the notebook

```bash
cd chronos/bayesian
python -m collect --out ./bayes_data                    # all five geometries, full design
python -m collect --out ./bayes_data --smoke            # tiny grids
python -m collect --out ./bayes_data --models p16-s16 p8-s8
python -m collect --out ./bayes_data --merge-only       # re-merge existing shards
```

## The five tables

| table | one row per | key columns |
|---|---|---|
| `contrasts` | (geometry, generator, background, lock, phase) | `R_lock`, `R_lo`, `R_hi`, `d`, `y_deficit`, `phase`, `live` |
| `mdl_cells` | (geometry, probe stage, candidate frequency) | `L_bits`, `SV`, `is_locked`, `dist_to_lock` |
| `mdl_bandtasks` | (geometry, stage, band task) | `SV`, `SV_shuffled`, `SV_random_init` |
| `collapse` | (geometry, signal mode, frequency, replicate) | `z`, `z_norm` |
| `sites` | (geometry, signal mode, replicate) | `sites`, `f1`, `delta_hat`, `n_sites` |

Shards live in `<out>/raw/{table}__p{P}-s{S}.parquet`; merged tables in `<out>/02_{table}.parquet`.

## Design decisions worth knowing

**Matched triplets.** Each lock $f_k$ is paired with controls $f_k \pm 0.25 f_s/S$ sharing the
*same* background realisation and the *same* phase. This differs deliberately from
`hypotheses.py`, which averages over phases sampled independently per frequency: the contrast has
to be paired for Eq. (8) to mean what it says, and the phase index has to survive for H2 to be
testable at all.

**Dropped lock sites.** A lock whose control would itself land on another member of
$\mathcal F_{lock}$ is dropped rather than silently compared against a lock. At `p16-s12` this is
most of them — the patch grid ($32$ Hz) and the stride grid ($42.\overline{6}$ Hz) interleave too
densely — leaving 64, 128 and 192 Hz.

**Frequency-local codelength.** The seven band tasks span the whole band, so one $L(D)$ summarises
hundreds of frequencies and the `IsLocked` indicator of Eq. (10) has nothing to attach to. Each MDL
cell instead asks a sharply local question: from the 256-d `[REG]` vector, can a probe separate a
tone at $f_c-1$ Hz from one at $f_c+1$ Hz? Same prequential protocol, same probe pipeline, narrower
scope. The band tasks are still collected, as a descriptive cross-check.

**The union sweep grid.** The collapse sweep runs on a uniform 1 Hz grid *unioned with the exact
predicted sites of every geometry*. Scoring a model only where its own geometry predicts a dip
would make every model trivially pass; and the union puts the non-integer sites on the grid
($42.\overline{6}$ Hz for $S=12$, $21.\overline{3}$ Hz for $S=24$), which a plain 1 Hz sweep misses.

**Signals.** Per the deliverable's Data section, the tone rides on a unit-variance **TSMixup** or
**KernelSynth** background at SNR 4. The collapse sweep also records the **pure sinusoid** mode,
where the degeneracy is exact ($z = 0$ when consecutive patches coincide); the background modes
turn that zero into a deep dip on the same grid and are the realistic-signal cross-check.

**One seed.** A single training checkpoint exists per geometry, so the seed level of the hierarchy
is omitted and every statement is conditional on that checkpoint. The configuration-level spread
$\tau$ absorbs training noise a multi-seed design would separate out.
