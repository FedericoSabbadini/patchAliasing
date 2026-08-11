# testing/ — structural-aliasing testing workflow

Select a model by geometry `(P, S)`, load the signals, and run the analyses — without editing any
notebook by hand. Trained checkpoints live under `../models/weights/` (gitignored, also on
Hugging Face).

## The single selection point — `testing_lib.py`

`load_pipeline(P, S)` picks the model by geometry: `(16,16)` → official `amazon/chronos-bolt-tiny`;
any other `(P, S)` → the retrained variant, **local checkpoint** (`../models/weights/p{P}-s{S}-seed42`,
latest) if present, **else Hugging Face**. `ALL_MODELS` = `(16,16) (16,12) (16,8) (16,4) (8,8) (24,24)`.

## The three analyses

| file | question it answers | scope |
|---|---|---|
| `chronosBolt_layer_probing.ipynb` | Is the frequency **decodable** as the signal passes through encoder→decoder→output? (reproduces Pagani *et al.*) | all models |
| `contamination.ipynb` | Given a patch lock, does the **forecast** rebuild the tone, and does a co-present lock tone hurt the **other** frequencies? | official `16-16` |
| `hypotheses.py` | Direct PASS/FAIL tests of the deliverable's **H1/H2/H3** | all models |

## Run everything

```bash
cd chronos/testing
python run_all_models.py                 # 6 models: probing + (16-16) contamination + H1/H2/H3 + cross
python run_all_models.py --smoke         # fast coarse pass
python run_all_models.py 16-16 8-8       # only these models
python run_all_models.py --no-hypotheses # notebooks only
```

Interactive: open a notebook, set `PATCH, STRIDE` in the CONFIG cell, run. Or run one hypothesis pass:

```bash
python hypotheses.py --P 16 --S 8        # H1/H2/H3 for one model
python hypotheses.py --cross             # H3 collapse-site table + figure, all models
```

Outputs:

```
outputs/
  per_model/p{P}-s{S}/     FIG3_space_saving, FIG5_reconstruction, FIG5b_generated_signals
  contamination/p16-s16/   recovery_at_lock, impact_on_others
  hypotheses/p{P}-s{S}/    H1_local_contrast, H2_phase_invariance, H3_collapse_sites
  hypotheses/              H3_collapse_sites_all_models.png, collapse_sites_all_models.csv
  executed/p{P}-s{S}/      the fully-executed copy of each notebook
```

---

## The figures — what each shows and how to read it

**`H3_collapse_sites`** — across-patch std of the input-patch-embedding tokens vs frequency.
*Read it:* the curve drops to **0** exactly where consecutive patches become identical
(`t_k = t_{k+1}`). Those zeros are the structural-aliasing sites; they sit on the stride grid `c·fs/S`.
This is the deliverable's definition of the phenomenon, measured directly.

**`H3_collapse_sites_all_models`** — the same curve stacked for the six geometries.
*Read it:* the collapse sites **move with the stride** (dense for `S=16`, sparse for `S=4`) and the
shape is the **same at every training budget** — so the collapse is geometric, not learned.

**`H1_local_contrast`** — forecast recovery at each lock `f_k` vs its controls `f_k ± δ` (left), and
the log-contrast `d` (right). *Read it:* green bars (`d<0`) are the localized loss H1 predicts; red
bars mean the lock recovers **as well or better** than its neighbours; grey = dead zone (nothing to
recover on either side). Only `cpp=3` (96 Hz) is a true localized loss.

**`H2_phase_invariance`** — the lock deficit `R(control) − R(lock)` measured across the signal's
phase offsets. *Read it:* points clustered tightly around their mean (small `CV`) mean the deficit is
set by the **geometry**, not the phase — the effect is structural.

**`FIG5b_generated_signals`** — left: injected-tone recovery vs cpp on the generated set; right: input
vs output dominant frequency. *Read it:* points **off the diagonal** on the right (96→128, 112→8) are
aliasing caught in the act — a tone reconstructed at the wrong frequency.

**`FIG5_reconstruction`** — reconstructed vs input frequency for a pure-tone sweep.
*Read it:* faithful up to the band edge `fs/P = 32 Hz`, then chaotic with **revivals** exactly at the
`cpp=2` and `cpp=4` locks (64, 128 Hz) but **not** at `cpp=3` (96 Hz), and dead above 128 Hz.

**`recovery_at_lock`** — context + ground truth + forecast at 32 Hz and 64 Hz.
*Read it:* the forecast overlays the ground truth with **~1–2° phase error** — the tone is rebuilt,
not nulled (a nulled/invented tone would show ~90° error).

**`impact_on_others`** — recovery of four carrier frequencies with vs without a co-present tone.
*Read it:* green = no added tone; grey = a non-critical control tone added; red = a **critical** tone
added. Red below grey = the critical tone costs the other frequencies more than an equal ordinary tone.

**`FIG3_space_saving`** — MDL compression (SV) per band task at every stage (paper Fig. 3).
*Read it:* bars are high and flat across encoder/decoder, then the **output head** drops on the
fully-trained model — decodability is largely architectural; the learned loss is at the projection.
(Dense, paper-parity figure; the annotations `(…s/…r)` are the shuffled-label and random-init controls.)

> `FIG4` (the task-stratified accuracy heatmap) was **removed**: its dips sat on the band-classification
> decision boundaries, not on the lock frequencies, so it did not isolate the phenomenon. `H1` now does
> that job cleanly.

---

## What the results show

*Verified on the executed notebooks and the saved figures in `outputs/`.*

**1. The lock tone is recovered by the forecast — it is not nulled.**
A pure patch-lock tone is rebuilt with the correct phase (32 Hz recovery 0.88, phase 2°; 64 Hz 0.83,
phase 1°; `contamination/p16-s16/recovery_at_lock.png`). Across the generated set the integer-cpp tones
recover far better than the half-integer ones on **every** model:

| model | recovery critical (int cpp) | recovery non-critical (half-int cpp) |
|---|---|---|
| p16-s16 | 0.690 | 0.178 |
| p16-s12 | 0.274 | 0.015 |
| p16-s8  | 0.322 | 0.044 |
| p16-s4  | 0.230 | 0.059 |
| p8-s8   | 0.678 | 0.030 |
| p24-s24 | 0.438 | 0.029 |

The advantage is an **average**, not a rule: the survivors are the cpp whose period divides the patch
(`cpp 1,2,4` → period 16,8,4 samples); `cpp=3` (96 Hz, period 5.33) collapses to ~0 and is aliased
onto 128 Hz (`FIG5b`, `FIG5`). So "critical recovers better" really means "octave-aligned cpp recover;
the others, including the critical `cpp=3`, do not."

**2. The token collapse is real, geometric, and lives on the stride grid `c·fs/S` (H3).**
Measuring the across-patch std of the `input_patch_embedding` tokens on a 1 Hz sweep (0 = consecutive
patches identical), the collapse sites match the integer members of `c·fs/S` **exactly** on every model:

| model | S vs P | measured collapse sites (integer-Hz) |
|---|---|---|
| p16-s16 | S = P | 32, 64, 96, 128, 160, 192, 224 |
| p16-s12 | S < P | 128  (42.7/85.3/170.7/213.3 Hz are off the 1 Hz grid) |
| p16-s8  | S < P | 64, 128, 192 |
| p16-s4  | S ≪ P | 128 |
| p8-s8   | S = P | 64, 128, 192 |
| p24-s24 | S = P | 64, 128, 192 |

The collapse is a **stride** effect: consecutive patches coincide iff `x[n+S]=x[n]` ⇒ `f=c·fs/S`. For
`S<P` the patch-null frequencies `k·fs/P` do **not** collapse (e.g. p16-s8: 32 Hz shows std ≈ 1.85 while
64 Hz shows 0.000). A smaller stride raises the lowest lock `fs/S` and thins the family below Nyquist —
this is the deliverable's mitigation, confirmed live. The result is **training-independent** (identical
in the under-trained variants), which is exactly why those variants are informative here even though
their absolute forecast quality is lower.

**3. A co-present critical tone costs the other frequencies — official model only.**
On `16-16`, adding a 32 Hz critical tone to four carriers lowers their mean recovery from **0.555** to
**0.398**, whereas an equal non-critical (48 Hz) control only lowers it to **0.480** — a critical-specific
extra cost of **+0.08** (`contamination/p16-s16/impact_on_others.png`). This is forecast-based, so it is
only meaningful where the model forecasts the carriers at all; the retrained variants recover them at ~0
(nothing to contaminate), so contamination is reported for the official model only.

**4. Frequency information stays decodable across encoder/decoder; the learned loss is at the output head.**
On the official model, band-classification compression (`FIG3`) is high and flat (SV ≈ 0.73–0.75) and
essentially equal to a random-init control internally; the only learned drop is at the output head
(0.470 vs random-init 0.761, gap **−0.29**). On the under-trained variants the output head shows a
learned *gain* instead of a drop (p16-s12 +0.14, p24-s24 +0.40) — the annotation is now sign-aware and
no longer mislabels a gain as a drop.

## The three hypotheses (from the deliverable), tested directly by `hypotheses.py`

- **H1** (localized loss at a lock: lower recovery **and** higher codelength than at `f_k ± δ`) —
  **refuted in its strong form on every model.** The token collapse is present at every lock, but the
  forecast does **not** uniformly lose information: among the live locks, recovery at the lock is on
  average *higher* than at the controls (mean `d>0`). The single clean localized loss is `cpp=3` (96 Hz)
  on the official model. So the loss is **frequency-selective, not present at every lock**.
- **H2** (the deficit is a property of the geometry, not the signal's phase) — **supported on every
  model**: across the sampled phases the lock-vs-control deficit is near-constant (`CV` 0.01–0.15).
- **H3** (lock sites move with the geometry) — **supported on every model**: measured collapse sites
  equal the integer stride grid `c·fs/S` and shift as `S` changes (see the table above and
  `H3_collapse_sites_all_models.png`).

## No data leakage

- **Probing MDL** is prequential: for each block it trains on the earlier blocks and scores the next,
  which is never in its training set; `StandardScaler+PCA+LogReg` are fit **inside** the fold on train
  only, so the feature transform never sees the test rows. Two controls guard it: shuffled-label
  (`SV ≤ 0`) and random-init.
- **`hypotheses.py`** and **`contamination.ipynb`** are controlled *measurements* (amplitude fits at a
  known frequency, token statistics), not trained classifiers, so train/test leakage does not apply; the
  H1 controls `f_k ± δ` are checked not to coincide with another lock, and `CTX=480` is divisible by every
  stride so no internal padding can fake a collapse.

### One-line summary

The patch **token collapse is real, geometric, and sits exactly on `c·fs/S`** (H3), moving with the
stride and independent of training; but it does **not** null the forecast (H1 refuted) — locked tones are
*rebuilt*, octave-aligned cpp even better than their neighbours, with `cpp=3` the one genuinely aliased
frequency; the deficit is **phase-invariant** (H2); a co-present lock tone **contaminates** the others on
the trained model; and frequency information stays **decodable** through the stack, with the only learned
drop at the output head.
