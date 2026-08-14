# testing/ — structural-aliasing testing workflow

Select a model by geometry `(P, S)`, load the signals, and run the analyses — without editing any
notebook by hand. Trained checkpoints live under `../models/weights/` (gitignored, also on
Hugging Face).

## The single selection point — `testing_lib.py`

`load_pipeline(P, S)` picks the model by geometry: `(16,16)` → official `amazon/chronos-bolt-tiny`;
any other `(P, S)` → the retrained variant, **local checkpoint** (`../models/weights/p{P}-s{S}-seed42`,
latest) if present, **else Hugging Face**. `ALL_MODELS` = `(16,16) (16,12) (16,8) (8,8) (24,24)`. (`S=4` is excluded — see below.)

### Which `(16,16)` — official 200k vs retrained 100k

Two roles are kept distinct, and this is why `(16,16)` can mean two different checkpoints:

- **Forecast-quality measurements** (the reconstruction figures FIG5/FIG5b; any recovery/amplitude
  claim) → the **official 200k** `chronos-bolt-tiny`. The budget-matched
  retrains are deliberately under-trained and would misrepresent *absolute* forecast accuracy, so
  the official model is used wherever the number is about how well the model forecasts.
- **Comparative, cross-configuration analyses** (MDL probing SV; the `H1/H2/H3` hypotheses) → a
  budget-matched **retrained 100k** `p16-s16` on the same footing as the other four geometries, so
  the comparison is not confounded by training budget. (The `H3` token collapse is
  training-independent, so it is unaffected either way.)

Until the retrained `p16-s16-seed42` checkpoint exists under `../models/weights/`, `(16,16)` resolves
to the official model everywhere; the comparative analyses adopt the retrained one once it is trained.

## The two analyses

| file | question it answers | scope |
|---|---|---|
| `chronosBolt_layer_probing.ipynb` | Is the frequency **decodable** as the signal passes through encoder→decoder→output? (reproduces Pagani *et al.*) | all models |
| `hypotheses.py` | Direct PASS/FAIL tests of the deliverable's **H1/H2/H3** | all models |

## Run everything

```bash
cd chronos/testing
python run_all.py                 # 5 models: probing + H1/H2/H3 (pure+tsm+ks) + cross
python run_all.py --smoke         # fast coarse pass
python run_all.py 16-16 8-8       # only these models
python run_all.py --no-probing    # hypotheses only  (--modes pure to skip tsm/ks)
```

Interactive: open the notebook, set `PATCH, STRIDE` in the CONFIG cell, run. Or run one hypothesis pass:

```bash
python hypotheses.py --P 16 --S 8        # H1/H2/H3 for one model
python hypotheses.py --cross             # H3 collapse-site table + figure, all models
```

### Signal mode: pure sinusoid (default) vs realistic background

By default `hypotheses.py` uses **pure sinusoids** — this isolates the geometry, so the phenomenon
shows its clean signature (the token collapse is *exactly* 0 at a lock). A **realistic-signal
cross-check** is available that rides the tone on a **unit-variance TSMixup background at SNR = 4** —
the *same* construction the probing notebook uses. It writes to parallel `*_tsm` paths and never
overwrites the pure-sinusoid figures:

```bash
python hypotheses.py --P 16 --S 8 --background-tsm    # -> outputs/hypotheses/p16-s8_tsm/ (TSMixup)
python hypotheses.py --P 16 --S 8 --background-ks     # -> outputs/hypotheses/p16-s8_ks/  (KernelSynth)
python hypotheses.py --cross --background-tsm          # -> outputs/hypotheses/..._tsm.{png,csv}
python hypotheses.py --cross --background-ks           # -> outputs/hypotheses/..._ks.{png,csv}
```

What the cross-check shows (see the dedicated section below): **H1/H2/H3 verdicts are unchanged**, the
collapse dips still land exactly on `c·fs/S` (and the background even exposes the *non-integer*
stride locks). So pure sinusoids remain the primary analysis; the background is a supplementary
robustness check, not a replacement.

Outputs:

```
outputs/
  probing/p{P}-s{S}/       FIG3_space_saving, FIG4_accuracy_heatmap, FIG5_reconstruction, FIG5b_generated_signals
  hypotheses/p{P}-s{S}/    H1_local_contrast, H2_phase_invariance, H3_collapse_sites
  hypotheses/p{P}-s{S}_tsm/ same three figures, tone on a TSMixup background
  hypotheses/p{P}-s{S}_ks/ same three figures, tone on a KernelSynth background
  hypotheses/              H3_collapse_sites_all_models{,_tsm}.png, collapse_sites_all_models{,_tsm}.csv
                           H1_local_contrast_all_models{,_tsm}.png
```

---

## The figures — what each shows and how to read it

**`H3_collapse_sites`** — token std vs frequency for one model. Drops to **0** at the stride grid
`c·fs/S` (green); secondary dip at `(c+½)·fs/S` (orange, anti-periodic). Same format as each row
of `H3_collapse_sites_all_models`.

**`H3_collapse_sites_all_models`** — same curve stacked for all five geometries. Collapse sites
**move with S** and are **training-independent**.

**`H1_local_contrast`** — forecast recovery vs frequency for one model, with stride locks (green)
and patch nulls (blue dashed) overlaid. Same format as each row of `H1_local_contrast_all_models`.

**`H1_local_contrast_all_models`** — recovery spectrum stacked for all five geometries. Peaks sit
**on the stride locks** (revivals, not nulls); 128 Hz is a universal revival.

**`H2_phase_invariance`** — lock (red) vs control (grey) recovery across phase offsets, one panel
per lock. Two flat bands + constant gap = deficit is geometric, not phase-dependent. `gap = μ ± σ`
annotation and overall CV in the suptitle.

**`FIG5b_generated_signals`** — injected-tone recovery vs cpp on the TSMixup/KernelSynth set.
Integer-cpp tones recover better; `cpp=3` (96 Hz) collapses to ~0.

**`FIG5_reconstruction`** — reconstructed vs input frequency (512-sample rollout, Pagani parity).
Faithful up to `fs/P`, then revivals at `cpp=2,4` (64, 128 Hz), dead at `cpp=3` (96 Hz).

**`FIG3_space_saving`** — MDL compression (SV) per band task at every stage (paper Fig. 3). The
only learned drop is at the output head; internal decodability is architectural.

**`FIG4_accuracy_heatmap`** — per-frequency band-classification accuracy (paper Fig. 4) with lock
frequencies overlaid. Dips near `k·fs/P` / `c·fs/S` are the aliasing signature. Folds are
**grouped by frequency** (`StratifiedGroupKFold` on frequency), so every phase of a held-out
frequency is excluded from training: the probe must generalise the band boundary, not memorise the
frequency. Accuracy is high where the band is genuinely decodable and **drops where the
representation degenerates** (the aliasing sites).

---

## What the results show

*Verified on the executed notebook and the saved figures in `outputs/`.*

**1. The lock tone is recovered by the forecast — it is not nulled.**
A pure patch-lock tone is rebuilt with the correct phase (the `hypotheses.py` recovery test returns
near-zero phase error at the locks; a nulled/invented tone would show ~90°). Across the generated set
the integer-cpp tones recover far better than the half-integer ones on **every** model (`FIG5b`):

| model | recovery critical (int cpp) | recovery non-critical (half-int cpp) |
|---|---|---|
| p16-s16 | 0.690 | 0.178 |
| p16-s12 | 0.274 | 0.015 |
| p16-s8  | 0.322 | 0.044 |
| p8-s8   | 0.678 | 0.030 |
| p24-s24 | 0.438 | 0.029 |

The advantage is an **average**, not a rule: the survivors are the cpp whose period divides the patch
(`cpp 1,2,4` → period 16,8,4 samples); `cpp=3` (96 Hz, period 5.33) collapses to ~0 and is aliased
onto 128 Hz (`FIG5b`, `FIG5`). So "critical recovers better" really means "octave-aligned cpp recover;
the others, including the critical `cpp=3`, do not."

**2. The token collapse is real, geometric, and lives on the stride grid `c·fs/S` (H3).**
Measuring the across-patch std of the `input_patch_embedding` tokens on a 0.1 Hz sweep (0 = consecutive
patches identical), the collapse sites match `c·fs/S` **exactly** on every model — including the
non-integer locks:

| model | S vs P | measured collapse sites |
|---|---|---|
| p16-s16 | S = P | 32, 64, 96, 128, 160, 192, 224 |
| p16-s12 | S < P | 42.7, 85.3, 128, 170.7, 213.3 |
| p16-s8  | S < P | 64, 128, 192 |
| p8-s8   | S = P | 64, 128, 192 |
| p24-s24 | S = P | 21.3, 42.7, 64, 85.3, 106.7, 128, 149.3, 170.7, 192, 213.3, 234.7 |

The collapse is a **stride** effect: consecutive patches coincide iff `x[n+S]=x[n]` ⇒ `f=c·fs/S`. For
`S<P` the patch-null frequencies `k·fs/P` do **not** collapse (e.g. p16-s8: 32 Hz shows std ≈ 1.85 while
64 Hz shows 0.000). A secondary, weaker dip sits at the half-stride points `(c+½)·fs/S`, where the
per-stride phase advance is an odd multiple of π so `x[n+S]=−x[n]` (patches sign-flipped, not identical).
A smaller stride raises the lowest lock `fs/S` and thins the family below Nyquist — this is the
deliverable's mitigation, confirmed live. The result is **training-independent** (identical in the
under-trained variants), which is exactly why those variants are informative here even though their
absolute forecast quality is lower.

**3. Frequency information stays decodable across encoder/decoder; the learned loss is at the output head.**
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
  - *Observation — does restricting H1 to the stride locks rescue it?* No. Evaluating H1 on the
    **stride-lock frequencies `c·fs/S` only** (excluding the patch nulls, relevant only for `S<P`:
    s12/s8) refutes it *more* strongly (mean log-contrast `d` = +3.5 / +3.4; `d<0` at 0 %
    of live locks). Reason: the in-band stride locks are exactly the octave-aligned survivors the
    forecast rebuilds best (`R_lock` 0.42–0.97), while their off-lock controls `f_k ± δ` sit above the
    `fs/P` reconstruction ceiling and recover ≈ 0 (`R_ctrl` 0.00–0.05). At a stride lock the forecast
    thus shows a localized *gain*; the genuine blind spots are the off-lock high frequencies (a
    band-limit/ceiling effect), not the locks. This is a check of an alternative lock set, **not** a
    redefinition of H1, which by design spans the full `c·fs/S ∪ k·fs/P` set.
- **H2** (the deficit is a property of the geometry, not the signal's phase) — **supported on every
  model**: across the sampled phases the lock-vs-control deficit is near-constant (`CV` 0.09–0.18).
- **H3** (lock sites move with the geometry) — **supported on every model**: measured collapse sites
  equal the integer stride grid `c·fs/S` and shift as `S` changes (see the table above and
  `H3_collapse_sites_all_models.png`).

### Per-model verdict summary

| model | H3 | H1 verdict | H1 d<0 % | H1 mean d | H1 live locks | H2 verdict | H2 CV |
|---|---|---|---|---|---|---|---|
| p16-s16 | PASS | REFUTED | 50% | +1.01 | 4 | SUPPORTED | 0.09 |
| p16-s12 | PASS | REFUTED |  0% | +2.55 | 2 | SUPPORTED | 0.13 |
| p16-s8  | PASS | REFUTED | 33% | +2.11 | 3 | SUPPORTED | 0.18 |
| p8-s8   | PASS | REFUTED |  0% | +3.58 | 2 | SUPPORTED | 0.13 |
| p24-s24 | PASS | REFUTED |  0% | +2.37 | 5 | SUPPORTED | 0.09 |

**H1 per-lock detail** (live locks only, grey = dead zone):

| model | lock Hz | cpp | R_lock | R_ctrl | d | tag |
|---|---|---|---|---|---|---|
| p16-s16 | 32 | 1.00 | 0.93 | 0.94 | −0.01 | loss |
| p16-s16 | 64 | 2.00 | 0.83 | 0.47 | +1.15 | revival |
| p16-s16 | 96 | 3.00 | 0.02 | 0.12 | −1.33 | loss |
| p16-s16 | 128 | 4.00 | 0.97 | 0.00 | +4.24 | revival |
| p16-s12 | 64 | 2.00 | 0.10 | 0.01 | +1.97 | revival |
| p16-s12 | 128 | 4.00 | 0.96 | 0.03 | +3.13 | revival |
| p16-s8 | 32 | 1.00 | 0.12 | 0.34 | −0.55 | loss |
| p16-s8 | 64 | 2.00 | 0.42 | 0.04 | +2.59 | revival |
| p16-s8 | 128 | 4.00 | 0.88 | 0.00 | +4.30 | revival |
| p8-s8 | 64 | 1.00 | 0.33 | 0.01 | +2.85 | revival |
| p8-s8 | 128 | 2.00 | 0.95 | 0.00 | +4.31 | revival |
| p24-s24 | 21.3 | 1.00 | 0.98 | 0.51 | +1.18 | revival |
| p24-s24 | 42.7 | 2.00 | 0.97 | 0.04 | +3.08 | revival |
| p24-s24 | 64 | 3.00 | 0.26 | 0.01 | +2.70 | revival |
| p24-s24 | 85.3 | 4.00 | 0.15 | 0.00 | +2.43 | revival |
| p24-s24 | 128 | 6.00 | 0.91 | 0.07 | +2.47 | revival |

**H2 per-lock gap** (from the two-bands plots):

| model | lock Hz | gap (ctrl − lock) | gap σ | interpretation |
|---|---|---|---|---|
| p16-s16 | 32 | +0.01 | 0.04 | near-parity |
| p16-s16 | 64 | −0.35 | 0.05 | revival |
| p16-s16 | 96 | +0.11 | 0.03 | genuine loss |
| p16-s16 | 128 | −0.97 | 0.01 | strong revival |
| p16-s12 | 64 | −0.09 | 0.07 | near-parity (stride lock) |
| p16-s12 | 128 | −0.93 | 0.07 | strong revival (stride lock) |
| p16-s8 | 32 | +0.26 | 0.09 | moderate loss (patch null) |
| p16-s8 | 64 | −0.39 | 0.16 | revival |
| p16-s8 | 128 | −0.88 | 0.02 | strong revival |
| p8-s8 | 64 | −0.31 | 0.12 | revival |
| p8-s8 | 128 | −0.95 | 0.05 | strong revival |
| p24-s24 | 21.3 | −0.47 | 0.03 | revival |
| p24-s24 | 42.7 | −0.93 | 0.02 | strong revival |
| p24-s24 | 64 | −0.24 | 0.06 | revival |
| p24-s24 | 85.3 | −0.14 | 0.07 | revival |
| p24-s24 | 128 | −0.84 | 0.06 | strong revival |

The pattern is consistent: **stride locks are revivals** (negative gap = lock recovers better).
The one **patch-only null** still testable here — 32 Hz on p16-s8 — is a **genuine loss**
(positive gap). On p16-s12 the patch nulls 32/96 Hz are not H1/H2-testable at δ = 0.25·fs/S
(their `f_k ± δ` controls fall on stride locks), so the clean-control filter drops them and only
the stride locks {64, 128, 192} survive. On S=P models every lock is a stride lock, so there are
no genuine losses.

## Realistic-signal cross-check — pure sinusoid vs TSMixup / KernelSynth background

The deliverable's Data section prescribes **two** signal sets: pure sine waves (2–250 Hz) and
TSMixup mixtures. The hypothesis tests use the **pure** set (to isolate the geometry); the probing
notebook uses the **TSMixup** set (to test decodability under a realistic background). To confirm the
conclusions are not an artefact of the clean input, every hypothesis test can be re-run with the tone
riding on a background at SNR = 4: **TSMixup** (`--background-tsm`, outputs to `*_tsm` paths) or
**KernelSynth** (`--background-ks`, outputs to `*_ks` paths). Result: **the conclusions hold; only
the sharpness changes.**

| test | pure sinusoid (primary) | TSMixup background + tone | changed? |
|---|---|---|---|
| **H3** collapse sites | exact zeros, clean PASS on the full `c·fs/S` grid including non-integer locks (all 5 models) | dips stay on `c·fs/S`; std no longer hits 0 (see note below) | mechanism identical, sharpness reduced |
| **H1** localized loss | REFUTED (5/5; the sole clean loss is `cpp=3`) | REFUTED (5/5) | **no** |
| **H2** phase-invariance | SUPPORTED (5/5), CV 0.09–0.18 | SUPPORTED (5/5), CV 0.08–0.17 | **no** |

*Reading the difference.* H3's exact-zero degeneracy is a property of the *noise-free* signal:
the zero requires `x[n+S] = x[n]` for **every** frequency component of the input. A pure sinusoid at
`f = c·fs/S` satisfies this by construction. A TSMixup signal is technically periodic (it is a sum of
sinusoids), but its period is the LCM of all component periods — not `S`. For the collapse to reach
zero, every component would need a frequency that is a multiple of `fs/S`; in practice the kernel
frequencies are drawn from a continuous distribution, so generically none of them land on that grid.
The tone component still satisfies the identity, pulling the std toward zero, but the background
components break it — giving a deep **dip** (2–3× below the off-lock baseline) instead of a zero.
The dip sits on exactly the same stride grid, which is why the background version is a confirmation,
not a contradiction. H1/H2 are
geometry-driven and therefore background-independent. **Conclusion: pure sinusoids are the correct
primary tool; the background cross-check corroborates H1/H2/H3.**

## No data leakage

- **Probing MDL** is prequential: for each block it trains on the earlier blocks and scores the next,
  which is never in its training set; `StandardScaler+PCA+LogReg` are fit **inside** the fold on train
  only, so the feature transform never sees the test rows. Two controls guard it: shuffled-label
  (`SV ≤ 0`) and random-init.
- **`hypotheses.py`** is a controlled *measurement* (amplitude fits at a known frequency, token
  statistics), not a trained classifier, so train/test leakage does not apply; the H1 controls
  `f_k ± δ` are checked not to coincide with another lock, and `CTX=480` is divisible by every stride
  so no internal padding can fake a collapse.
- **`FIG4` accuracy heatmap** uses **frequency-grouped** cross-validation (`StratifiedGroupKFold` on
  frequency): all phases of a test frequency are held out of training, so the probe cannot memorise a
  frequency's identity through its other phases — the accuracy reflects genuine band generalisation.

### One-line summary

The patch **token collapse is real, geometric, and sits exactly on `c·fs/S`** (H3), moving with the
stride and independent of training; but it does **not** null the forecast (H1 refuted) — locked tones are
*rebuilt*, octave-aligned cpp even better than their neighbours, with `cpp=3` the one genuinely aliased
frequency; the deficit is **phase-invariant** (H2); and frequency information stays **decodable** through
the stack, with the only learned drop at the output head.
