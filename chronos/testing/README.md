# testing/ — structural-aliasing testing workflow

Clean, reproducible workflow to **select a model**, **generate/load the signals**, and **run the
two notebooks** for every model — without editing any notebook or script by hand.

Trained checkpoints live under `../models/weights/` (gitignored, ~16 GB, also on Hugging Face).

## `testing_lib.py` — the single selection point

Shared by both notebooks: `load_pipeline(P, S)` selects the model by geometry —
`(16,16)` → official `amazon/chronos-bolt-tiny`; any other `(P, S)` → the retrained variant,
**local checkpoint** (`../models/weights/p{P}-s{S}-seed42`, latest) if present, **else Hugging
Face**. `ALL_MODELS` lists the six geometries: `(16,16) (16,12) (16,8) (16,4) (8,8) (24,24)`.

## Step 1 — generate the signals

Compact, explicit dataset (fs = 512 Hz, P = 16 reference) mixing **critical** cpp (integer,
patch-locked: 32/64/96/128 Hz = cpp 1/2/3/4) and **non-critical** cpp (48/80/112 Hz = cpp
1.5/2.5/3.5), balanced across **TSMixup** and **KernelSynth**, plus two multi-tone signals.
Edit `../data/synthetic/signals.json`, then:

```bash
cd chronos/data/synthetic
python signalGenerator.py            # writes .npy + .png into ./signals/
```

## Step 2 — run the notebooks on the models

```bash
cd chronos/testing
python run_all_models.py                       # probing on all 6 models; contamination on 16-16
python run_all_models.py --smoke               # fast coarse pass (PROBE_SMOKE=1)
python run_all_models.py 16-16 8-8             # only these models (p{P}-s{S} tags)
```

`run_all_models.py` executes the **notebooks themselves** (via `jupyter nbconvert --execute`),
passing the geometry through `PROBE_PATCH`/`PROBE_STRIDE`. **Contamination is run only on the
official `16-16`** (see below for why). Outputs:

```
outputs/
  per_model/p{P}-s{S}/       FIG3_space_saving, FIG4_accuracy_heatmap, FIG5_reconstruction, FIG5b_generated_signals
  contamination/p16-s16/     recovery_at_lock, impact_on_others
  executed/p{P}-s{S}/        the fully-executed copy of each notebook
```

Interactive use: open either notebook, set `PATCH, STRIDE` in the first cell (or the env vars),
run. Every lock frequency is re-derived from the loaded geometry.

---

## What the current results show

*Every statement below is verified on the executed notebooks and the saved figures in `outputs/`.*

**1. The critical frequency is recovered by the forecast — it is not nulled.**
On the official model, a pure patch-lock tone is rebuilt with the correct phase: 32 Hz recovery
0.88 (phase error 2°), 64 Hz recovery 0.83 (phase error 1°) (`contamination/p16-s16/recovery_at_lock.png`).
Across the generated dataset the critical (integer-cpp) tones recover far better than the
non-critical ones on **every** model:

| model | recovery critical | recovery non-critical |
|---|---|---|
| p16-s16 | 0.690 | 0.178 |
| p16-s12 | 0.274 | 0.015 |
| p16-s8  | 0.322 | 0.044 |
| p16-s4  | 0.230 | 0.059 |
| p8-s8   | 0.678 | 0.030 |
| p24-s24 | 0.438 | 0.029 |

The advantage is on **average**, not uniform: in `FIG5b`/`FIG5` the cpp≈3 harmonic (96 Hz) dips
to ~0, and reconstruction collapses above ~95 Hz with a revival exactly at 128 Hz.

**2. The structural collapse is geometric and shrinks as the stride shrinks (S < P mitigates).**
Measuring the across-patch standard deviation of the `input_patch_embedding` tokens for a pure
tone sweep (0 = consecutive patches identical), the collapse frequencies are — on a 1 Hz grid:

| model | S vs P | collapse frequencies | count |
|---|---|---|---|
| p16-s16 | S = P | 32, 64, 96, 128 | 4 |
| p16-s8  | S < P | 64, 128 | 2 |
| p16-s4  | S ≪ P | 128 | 1 |
| p8-s8   | S = P | 64, 128 | 2 |

`S = P` collapses at **every** integer cpp (patch-null and stride-lock coincide); `S < P`
collapses **only** at the stride-locks `c·fs/S` (the overlap breaks the patch-null degeneracy),
so fewer frequencies collapse (s16→4, s8→2, s4→1). This is **training-independent** (it is a
property of the embedding geometry), which is why the under-trained variants are still
informative here. (Non-integer stride-locks — e.g. 42.7/85.3 Hz for S=12 — are not on the 1 Hz
grid and are undercounted; the s16→s8→s4 trend is unaffected.)

**3. A co-present critical tone costs the other frequencies — measured on the official model only.**
On `16-16`, adding a 32 Hz critical tone to four non-critical carriers lowers their mean recovery
from **0.555** (without) to **0.398** (with), whereas an equal-amplitude non-critical (48 Hz)
control only lowers it to **0.480** — a critical-specific extra cost of ~0.08
(`contamination/p16-s16/impact_on_others.png`). This test is **forecast-based**, so it is only
meaningful where the model forecasts well: the retrained variants recover the carriers at ~0
(nothing to contaminate), so contamination is reported for the official model only.

**4. Frequency information is linearly decodable at every encoder/decoder stage.**
The band-classification Space-Saving (`FIG3`) is high and flat across all encoder and decoder
stages (SV ≈ 0.73–0.75 on the official model) and is **essentially equal to an untrained
random-init model** (learned gap ≈ 0 internally). The only stage with a learned drop is the
output head (trained 0.470 vs random-init 0.761, gap −0.291). So internal decodability is largely
an architectural property; the learned degradation is concentrated at the output projection.

**Caveat on the per-frequency heatmap (`FIG4`).** The accuracy dips there sit at the band-task
**decision boundaries** (33/64/95/126… Hz) and do **not** move with `P`: p16-s16 (critical at
32/64/96) and p8-s8 (critical at 64/128) show the dips in the *same* places. So `FIG4` on its own
does **not** isolate a critical-frequency MDL drop — it is confounded by the classification task.
A clean per-frequency MDL test at `f_k` vs `f_k ± δ` (deliverable hypothesis H1) is **not yet
built**; it is the natural next step.

### One-line summary

Critical (patch-locked) frequencies are **recovered** in the forecast, not nulled; the patch
**token collapse is real and geometric, and is reduced by a smaller stride** (`S < P`); a
co-present critical tone **degrades the other frequencies** on the fully-trained model; the
frequency information stays **decodable across encoder/decoder**, with the learned loss at the
output head. The direct per-frequency MDL test of H1 remains to be added.
