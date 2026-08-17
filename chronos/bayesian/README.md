# bayesian/ — Bayesian analysis of structural aliasing

Implements the Bayesian section of Deliverable 2
(`coursework/delivered/deliverable_2/deliverable2_v0/sections/deliverable2.tex`, §Bayesian Analysis
Design). Every model, prior and decision rule here is the one written there; if the two disagree,
the deliverable is the specification.

| file | what it is |
|---|---|
| `bayesian_analysis.ipynb` | **the** notebook — priors, observation, likelihood, posterior, checks |
| `probe_lib.py` | light, notebook-free probing core (forecast recovery, token collapse, `[REG]` capture, prequential MDL, lock geometry) |
| `collect.py` | runs every geometry in `probe_lib.MODELS` and writes the tidy tables; shards per model so a run resumes |

Nothing under `chronos/testing/` is modified. Local checkpoint discovery is reused from
`chronos/testing/testing_lib.py`; which checkpoint a geometry means is decided here, in
`probe_lib.load_checkpoint` (see Geometries below).

## The seven estimands

| model | claim | estimand | predicts |
|---|---|---|---|
| **A** hierarchical Student-$t_4$ on the paired log contrast $d$ | H1 behavioural | $e^{\bar\beta}$ — recovery at a lock relative to its controls | $<1$ |
| **B** Gamma regression, log link, on the frequency-local MDL codelength | H1 representational | $e^{\theta_{lock}}$ — codelength expansion at a lock | $>1$ |
| **C** Model A on the per-phase deficit, eight phase bins | H2 | $\sigma_\phi$ — spread of the deficit across phase | $\approx 0$ |
| **D1** two grid labels on $\log z_g(f)$, compared by LOO | H3 location | $\theta_S,\theta_P$ | both $<0$ |
| **D2** scaling law, stride branch | H3a | $\kappa_S$ | $=1$ |
| **D2** scaling law, patch branch | H3b | $\kappa_P$ | $=1$ |
| **A$'$** the configuration level of A | M1 mitigation | $\delta_O$ | $<0$ |

M1 is decided by the sign of $\delta_O$ **and** by a LOO comparison among four configuration-level
parameterisations (`config_level` in `model_A_contrast`): overlap, patch size, both, neither.

## Geometries

The nineteen runs of `tab:patchStride`, all retrained from scratch under the same 100k-step budget
and the same seed: `p8-s8`; `p16-s8`, `p16-s12`, `p16-s15`, `p16-s16`; `p24-s8`, `p24-s12`,
`p24-s15`, `p24-s16`, `p24-s20`, `p24-s24`; `p32-s8`, `p32-s12`, `p32-s15`, `p32-s16`, `p32-s20`,
`p32-s24`, `p32-s28`, `p32-s32`.

The published `amazon/chronos-bolt-tiny` checkpoint is **not** used: `p16-s16` is the
budget-matched retrained baseline. The shortest strides on the sweep repository are excluded,
matching `deliverable1.tex`: their stride-lock class has too few in-band members to carry a local
H1 contrast.

Checkpoints are resolved by `probe_lib.load_checkpoint`, not by `testing_lib.load_pipeline`: the
latter belongs to the Deliverable 1 workflow, where `(16,16)` means the published checkpoint and
only the first five geometries exist. Here every geometry is `p{P}-s{S}-seed42`, taken from the
local weights directory if there is one and otherwise from the Hugging Face sweep repo. If a run is
missing from the hub, `load_checkpoint` raises rather than silently substituting another geometry.

**`MODELS` and `BAYES_MODELS` are not the same list.** `MODELS` is the full design and is what the
sweeps and the descriptive figures run on. `BAYES_MODELS` is the subset the Bayesian models are
fitted on, and it drops any geometry whose stride does not divide the probing context `CTX = 480`,
because the last patch would then be internally padded and the padding, not the geometry, would
produce the collapse. Today that excludes `p32-s28` alone: 480 = 17*28 + 4, and the smallest
context divisible by 28 as well is 1680, far outside the probing convention. `p32-s28` is therefore
collected and plotted but not fitted, and `deliverable2.tex` records the exclusion.

`probe_lib.design_gaps()` prints what the active set cannot identify. An estimand named there must
be reported as **not identified** rather than as a null result, and the notebook's verdict table
does exactly that for a branch with no usable rows.

**Why nineteen and not five.** When $S$ divides $P$, every stride lock $c f_s/S$ is also a patch
null $k f_s/P$ with $k = cP/S$, so the two branches coincide over the whole grid and no frequency in
that geometry can be attributed to one of them. Only $S \nmid P$ yields **stride-only** sites. Ten
of the nineteen satisfy it and between them they carry **68** such sites: `p16-s12` (4),
`p16-s15` (7), `p24-s15` (6), `p24-s16` (4), `p24-s20` (8), `p32-s12` (4), `p32-s15` (7),
`p32-s20` (8), `p32-s24` (8), `p32-s28` (12). The rest make $P$ vary at fixed $S$ — six strides
appear at more than one patch size — which is what identifies H3b and separates $\delta_O$ from
$\delta_P$; three runs share $O = 0.5$ at three different patch sizes, which is what separates an
effect of the overlap ratio from an effect of the absolute stride.

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
python -m collect --out ./bayes_data                    # every geometry in MODELS, full design
python -m collect --out ./bayes_data --smoke            # tiny grids
python -m collect --out ./bayes_data --models p24-s16 p24-s20
python -m collect --out ./bayes_data --merge-only       # re-merge existing shards
```

## The five tables

| table | one row per | key columns |
|---|---|---|
| `contrasts` | (geometry, generator, background, lock, phase) | `R_lock`, `R_lo`, `R_hi`, `d`, `y_deficit`, `phase`, `delta`, `family`, `live` |
| `mdl_cells` | (geometry, probe stage, candidate frequency) | `L_bits`, `SV`, `is_locked`, `dist_to_lock` |
| `mdl_bandtasks` | (geometry, stage, band task) | `SV`, `SV_shuffled`, `SV_random_init` |
| `collapse` | (geometry, signal mode, frequency, replicate) | `z`, `z_norm` |
| `sites` | (geometry, signal mode, replicate, **branch**) | `sites`, `f1`, `delta_hat`, `n_sites`, `predicted_spacing`, `n_ambiguous` |

Shards live in `<out>/raw/{table}__p{P}-s{S}.parquet`; merged tables in `<out>/02_{table}.parquet`.

## Design decisions worth knowing

**Matched triplets.** Each lock $f_k$ is paired with two controls sharing the *same* background
realisation and the *same* phase. This differs deliberately from `hypotheses.py`, which averages
over phases sampled independently per frequency: the contrast has to be paired for the deliverable's
Eq. (1) to mean what it says, and the phase index has to survive for H2 to be testable at all.

**The control offset is chosen per site, not per geometry.** Deliverable 1 fixed it at
$0.25 f_s/S$. That rule is defined from the stride alone and cannot see the patch grid, and at
`p16-s12` it is exactly one third of the patch-null spacing $f_s/P$: every stride-only site there
gets one control on a patch null, and all four are discarded — the whole of that geometry's
stride-only evidence. `control_offset(P, S, f)` instead returns the largest offset not exceeding
$0.25 f_s/S$ that keeps both controls clear of **both** grids, and drops a site only if none
exists. It changes nothing at any other geometry.

**The near-zero filter conditions on the response.** `live` drops a contrast whose recovery is near
zero at the lock and at both controls. That removes the ties and keeps the asymmetric cases, so it
can only push $\bar\beta$ away from zero, towards H1. The deliverable therefore requires the fit to
be reported **with and without** it, and the removed fraction per geometry to be reported as a
result rather than as a design parameter.

**Frequency-local codelength.** The seven band tasks span the whole band, so one $L(D)$ summarises
hundreds of frequencies and the `IsLocked` indicator has nothing to attach to. Each MDL cell
instead asks a sharply local question: from the captured internal state, can a probe separate a
tone at $f_c-1$ Hz from one at $f_c+1$ Hz? Same prequential protocol, narrower scope. The band
tasks are still collected, as a descriptive cross-check.

**`[REG]` is an encoder token.** Ten states are probed: the encoder `[REG]` token after each
encoder block, the decoder state after each decoder block, the pre-projection vector and the
quantile head. Chronos-Bolt carries a `[REG]` token in the encoder only; its decoder is driven by a
single decoder-start token, so the decoder-side vectors are decoder states and not `[REG]`.

**The union sweep grid.** The collapse sweep runs on a uniform 1 Hz grid *unioned with the exact
predicted sites of every geometry*. Scoring a model only where its own geometry predicts a dip
would make every model trivially pass; and the union puts the non-integer sites on the grid
($42.\overline{6}$ Hz for $S=12$, $25.6$ Hz for $S=20$), which a plain 1 Hz sweep misses.

**One scaling law per branch.** $\mathcal F_{lock}$ is a union, so the detected set is the union of
two combs and has no single spacing. `derive_sites` assigns each detected dip to the branch that
predicts it, sets aside the ambiguous ones, and reports one row per branch; `model_D2_movement`
then fits one no-intercept law per branch, with $\kappa_S = 1$ and $\kappa_P = 1$ as the two
predictions. The earlier formulation regressed one pooled spacing on both predictors and predicted
$\kappa_P = 0$, which contradicts H3 as approved in Deliverable 1.

**Signals.** Per the deliverable's Data section, the tone rides on a unit-variance **TSMixup** or
**KernelSynth** background at SNR 4. The collapse sweep also records the **pure sinusoid** mode,
where the degeneracy is exact ($z = 0$ when consecutive patches coincide); the background modes
turn that zero into a deep dip on the same grid and are the realistic-signal cross-check.

**One seed.** A single training checkpoint exists per geometry, so the seed level of the hierarchy
is omitted and every statement is conditional on that checkpoint. The configuration-level spread
$\tau$ absorbs training noise a multi-seed design would separate out.

**Geometry and sequence length are not separable.** At a fixed context, changing $P$ or $S$ also
changes the token count. A geometry effect and a sequence-length effect cannot be told apart in
this design, and the deliverable records it as a limitation.
