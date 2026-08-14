# Chronos-Bolt Tiny — from-scratch P/S retraining sweep (`train_sweep.py`)

## Purpose and scope

`train_sweep.py` does **training only** — no evaluation, no inference. It retrains
Chronos-Bolt Tiny **from scratch** (random weights, *not* fine-tuning) on the **full
official Chronos pre-training data**, varying **only** the patch geometry
`input_patch_size` (P) and `input_patch_stride` (S). It produces one checkpoint per
`(P, S, seed)`, named so the downstream structural-aliasing pipeline (signal generation,
probing, Bayesian analysis) can consume and compare them.

The training data reproduces the official diet:
- **TSMixup** (`training_corpus_tsmixup_10m`, 10M series) — augmented from 28 real-world
  open-source datasets (Monash, M-competitions, Kaggle; energy, transport, weather, finance,
  etc.) via Dirichlet-weighted convex combinations.
- **KernelSynth** (`training_corpus_kernel_synth_1m`, 1M series) — purely synthetic,
  sampled from Gaussian Process priors with a composite kernel bank.
- Interleaved at the **official 9:1 ratio** (9 TSMixup series per 1 KernelSynth).

Everything except P/S is held fixed across runs, so any downstream difference is
attributable to the patch geometry and not to some other setup variable — the necessary
condition for the quantitative comparison in the report.

The retrained variants are compared **only against each other**, never against the stock
pretrained checkpoint. That is why a small fixed step budget is acceptable (the
under-training is identical across runs and cancels out).

## How to run

No flags. Edit the config block at the top of the file, then:

```sh
python train_sweep.py
```

The default sweep is 6 configs × 1 seed = 6 runs. It is **resumable**: any run that has
already written a `DONE` marker is skipped, so you can relaunch after an interruption.
For a quick end-to-end validation on a new machine, temporarily set `MAX_STEPS` low
(e.g. 20) **and** `SHUFFLE_BUFFER_SIZE` low (e.g. 100 — at the official 100 000 the stream
must download ~100k series before the first batch), then restore `10_000` / `100_000`.

## Configuration (the `(P, S)` grid)

Two-axis design (`S ≤ P` everywhere, so no unobserved gaps):

| config  | P  | S  | overlap `or=(P−S)/P` | ~patches @ ctx 2048 | axis |
|---------|----|----|----------------------|---------------------|------|
| p16-s16 | 16 | 16 | 0.00 | 128 | baseline (contiguous) |
| p16-s12 | 16 | 12 | 0.25 | 170 | stride/overlap (P fixed) |
| p16-s8  | 16 | 8  | 0.50 | 255 | stride/overlap (P fixed) |
| p8-s8   | 8  | 8  | 0.00 | 256 | patch-size (overlap fixed) |
| p24-s24 | 24 | 24 | 0.00 | 86  | patch-size (overlap fixed) |

> `S=4` is intentionally excluded: with `fs/S = 128 Hz` its stride-lock class `{128, 256, …}`
> has a single member below Nyquist, so it yields no informative H1/H3 contrast.

- The **P=16 row** varies only `S` → probes the overlap mechanics (H3) and the blind-spot
  set `F_lock = c·f_s/S`, which depends on `S`.
- The **contiguous row** (p8-s8 / p16-s16 / p24-s24) varies only `P` at overlap 0 → probes
  within-patch redundancy (`T_0 | P`).

Patch count spans ~3× across the grid (86 → 255), so p16-s8 is the memory/time-heavy
config. If it OOMs, lower `BATCH_SIZE` **globally** (the same value for every config) to
keep runs comparable.

## Outputs

One folder per model, under `chronos/models/weights/p{P}-s{S}-seed{seed}/`, holding every
artifact tied to that model:

- `run_config.json` — all hyperparameters + provenance (git commit, torch/GPU, precision)
  + final-loss summary;
- `loss_history.npy`, `loss_curve.png` — training-loss diagnostics (of the optimisation
  process, **not** a quality metric of the model);
- `checkpoint-{step}/` every `SAVE_EVERY`, and the final model at the run-dir root;
- `DONE` — written only after a successful final save (the resume marker).

Plus an aggregate `chronos/models/weights/manifest.csv`, rebuilt idempotently from every
finished run.

## Design decisions and why

The script builds on the open-source Chronos project by Amazon Science
(<https://github.com/amazon-science/chronos-forecasting>, Apache-2.0; paper: Ansari et al.,
*Chronos: Learning the Language of Time Series*, 2024, cited as
`ansariChronosLearningLanguage2024`). The `scripts.zip` in this directory is that
repository's `scripts/` folder, unmodified, with its original
`Copyright Amazon.com, Inc. / SPDX-License-Identifier: Apache-2.0` headers intact.
Checkpoints (`amazon/chronos-bolt-tiny`, …) are on HuggingFace.

Two things it is important to be precise about, because they justify the choices below:

- The official `scripts/training/train.py` is the trainer for the **original Chronos-T5**
  (value-quantization tokenizer, GluonTS arrow datasets, HF `Trainer`, DDP; 703 lines). It
  is **not** a Bolt trainer, and it is **not** the code base of `train_sweep.py`.
- `train_sweep.py` is **written directly against the real Chronos-Bolt API**
  (`chronos.chronos_bolt.ChronosBoltModelForForecasting`, the `input_patch_size` /
  `input_patch_stride` config keys, and the forward
  `context, mask, target, target_mask → .loss`). The model architecture is used as-is from
  the real library — none of it is reconstructed. From `train.py` we adopt only a handful of
  high-level choices (listed under "Adopted from the official reference"); no code is copied
  verbatim, so the Apache-2.0 header is not reproduced inside the file. In the source, every
  line whose approach, API, or default comes from the official reference is tagged with a
  `# [CHRONOS-REF]` marker (see the legend in the module docstring); untagged lines are project
  scaffolding.

### From scratch, not fine-tuning
Only the *architectural config* of `amazon/chronos-bolt-tiny` is loaded; weights are
reinitialised. This is **required**: changing `input_patch_size` changes the first
embedding block's input dimension (`input_patch_size · 2`, patch values concatenated with
the observation mask), so a P=16 checkpoint's weights cannot be loaded into a P≠16 model.
This is a real fact about the Bolt patch-embedding, not a modelling assumption.

### Full official training data: TSMixup + KernelSynth
Both official corpora are used, loaded in **streaming** mode (no full download):
- `training_corpus_tsmixup_10m` (10M series) — TSMixup augmentations of 28 real-world
  open-source datasets (the real datasets are already baked into this corpus).
- `training_corpus_kernel_synth_1m` (1M series) — synthetic GP time series.
Interleaved at the **official 9:1 ratio** (9 TSMixup series per 1 KernelSynth), matching
the paper's training mixture (Ansari et al. 2024).

### Only P and S vary between runs
`context_length`, `prediction_length`, `quantiles`, batch size, learning rate, schedule,
etc. are fixed at the Chronos/Bolt defaults. This is the only way to attribute a behavioural
difference to P/S.

### Aligned to the official reference (tagged `# [CHRONOS-REF]` in the source)
All optimisation and data-pipeline **values** come from the official published pipeline
(`train.py` + `chronos-t5-tiny.yaml`, the only Chronos training recipe Amazon has released),
and the model-facing code follows the `chronos_bolt.py` source exactly:
- **LR 1e-3, weight decay 0.0, warmup 0.0, linear LR decay** — the exact official values.
  (The official pipeline trains from scratch *without* warmup; `initializer_factor=0.05`,
  which the Bolt-tiny config ships, is what keeps early steps stable.)
- **AdamW (fused on CUDA)** — official `optim: adamw_torch_fused`; **grad-clip 1.0** — the
  HF Trainer default `max_grad_norm` implicitly used by `train.py`.
- **Batch 32** — official `per_device_train_batch_size: 32` with grad-accum 1.
- **Shuffle buffer 100 000** — official `shuffle_buffer_length` (lower it only for smoke tests).
- **Window sampling**: split point uniform with **`min_past=60`** context points and a full
  64-step future (mirrors `ExpectedNumInstanceSampler` + `InstanceSplitter`); series with
  fewer than `min_past + 64` points or **> 90% missing** are dropped (`has_enough_observations`);
  **`drop_prob=0.2`** random NaN-injection augmentation per series (official
  `ChronosDataset` default).
- **fp32 compute with TF32 matmuls** on compute-capability ≥ 8 GPUs — official
  `tf32: true` behaviour, including the capability guard. No AMP, no GradScaler.
- **Seeding via `transformers.set_seed`** — as in `train.py`.
- **The NaN contract** (critical, verified in `chronos_bolt.py`): Bolt encodes "unobserved"
  as **NaN**. `InstanceNorm` computes loc/scale with `nanmean` *before* the mask is applied,
  so missing values and the left-padding of short contexts are passed as **NaN, never
  zero-filled** — the model itself zeroes them *after* normalization. Zero-filling would
  corrupt the normalization statistics (empirically: ~2.6× inflated loss at init).

### Deliberate deviations from the official regime (uniform across all runs)
- **`MAX_STEPS` 100 000 vs official 200 000** — fixed compute budget; valid because variants
  are compared only against each other and the budget is identical for every run.
- **Log/save cadence** (50/1000 vs official 500/100k) — denser diagnostics for short runs.
- **No `transformers.Trainer`** — a production wrapper; the manual loop reproduces the same
  optimisation mathematics (verified value-by-value above) without the dependency.
- **`num_workers=0`** — equivalent single-stream loading (official used 1 worker).
- **No `torch_compile`** — startup overhead across 6 short runs; no change to the math.

### Project scaffolding written here (untagged in the source)
- **Early abort on non-finite loss**: one bad run fails fast instead of burning hours; the
  sweep continues with the next config.
- **The `(P, S, seed)` sweep loop, resumability (`DONE` marker), the per-model
  `run_config.json` provenance, the aggregate `manifest.csv`, and the tqdm progress bar.**
