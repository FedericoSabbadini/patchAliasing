# Chronos-Bolt Tiny — from-scratch P/S retraining sweep (`train_sweep.py`)

## Purpose and scope

`train_sweep.py` does **training only** — no evaluation, no inference. It retrains
Chronos-Bolt Tiny **from scratch** (random weights, *not* fine-tuning) on the official
Chronos pre-training corpus, varying **only** the patch geometry `input_patch_size` (P)
and `input_patch_stride` (S). It produces one checkpoint per `(P, S, seed)`, named so the
downstream structural-aliasing pipeline (signal generation, probing, Bayesian analysis)
can consume and compare them.

Everything except P/S is held fixed across runs, so any downstream difference is
attributable to the patch geometry and not to some other setup variable — the necessary
condition for the quantitative comparison in the report.

The retrained variants are compared **only against each other**, never against the stock
pretrained checkpoint. That is why a small fixed step budget is acceptable (the
under-training is identical across runs and cancels out) and why each `(P, S)` is repeated
over several seeds — to separate a real P/S effect from seed noise.

## How to run

No flags. Edit the config block at the top of the file, then:

```sh
python train_sweep.py
```

The default sweep is 6 configs × 3 seeds = 18 runs. It is **resumable**: any run that has
already written a `DONE` marker is skipped, so you can relaunch after an interruption.
For a quick end-to-end validation on a new machine, temporarily set `MAX_STEPS` low
(e.g. 20), run, then set it back to `10_000`.

## Configuration (the `(P, S)` grid)

Two-axis design (`S ≤ P` everywhere, so no unobserved gaps):

| config  | P  | S  | overlap `or=(P−S)/P` | ~patches @ ctx 2048 | axis |
|---------|----|----|----------------------|---------------------|------|
| p16-s16 | 16 | 16 | 0.00 | 128 | baseline (contiguous) |
| p16-s12 | 16 | 12 | 0.25 | 170 | stride/overlap (P fixed) |
| p16-s8  | 16 | 8  | 0.50 | 255 | stride/overlap (P fixed) |
| p16-s4  | 16 | 4  | 0.75 | 509 | stride/overlap (P fixed) |
| p8-s8   | 8  | 8  | 0.00 | 256 | patch-size (overlap fixed) |
| p24-s24 | 24 | 24 | 0.00 | 86  | patch-size (overlap fixed) |

- The **P=16 row** varies only `S` → probes the overlap mechanics (H3) and the blind-spot
  set `F_lock = c·f_s/S`, which depends on `S`.
- The **contiguous row** (p8-s8 / p16-s16 / p24-s24) varies only `P` at overlap 0 → probes
  within-patch redundancy (`T_0 | P`).

Patch count spans ~6× across the grid (86 → 509), so p16-s4 is the memory/time-heavy
config. If it OOMs, lower `BATCH_SIZE` **globally** (the same value for every config) to
keep runs comparable.

## Outputs

One folder per model, under `chronos/outputs/models/p{P}-s{S}-seed{seed}/`, holding every
artifact tied to that model:

- `run_config.json` — all hyperparameters + provenance (git commit, torch/GPU, precision)
  + final-loss summary;
- `loss_history.npy`, `loss_curve.png` — training-loss diagnostics (of the optimisation
  process, **not** a quality metric of the model);
- `checkpoint-{step}/` every `SAVE_EVERY`, and the final model at the run-dir root;
- `DONE` — written only after a successful final save (the resume marker).

Plus an aggregate `chronos/outputs/models/manifest.csv`, rebuilt idempotently from every
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

### One corpus: `training_corpus_tsmixup_10m`
The official TSMixup pre-training corpus (10M series), loaded in **streaming** mode (no full
download). KernelSynth (1M, auxiliary, ~10% in the Chronos paper) is not mixed in: a single
corpus avoids two-stream mixing logic without losing representativeness for a "tiny"
retraining.

### Only P and S vary between runs
`context_length`, `prediction_length`, `quantiles`, batch size, learning rate, schedule,
etc. are fixed at the Chronos/Bolt defaults. This is the only way to attribute a behavioural
difference to P/S.

### Adopted from the official reference — kept
Design choices taken from `train.py` (tagged `# [CHRONOS-REF]` in the source) that we keep because they
protect comparability between P/S runs:
- **Warmup + linear LR schedule** (`WARMUP_RATIO`, `LR_SCHEDULER_TYPE`): from random weights
  a constant LR is unstable in early steps; if that instability hit different P/S configs
  differently it could masquerade as an aliasing effect.
- **Stream shuffle buffer** (`SHUFFLE_BUFFER_SIZE`): a streamed dataset reads shards in
  order; shuffling breaks correlations between consecutive batches.
- **AdamW + weight decay + gradient-norm clipping**, and **fixed-step training**, as in the
  official setup.
Plus the Bolt API itself (model class, config keys, forward signature, quantiles, default
`context_length`/`prediction_length`) — all `# [CHRONOS-REF]`.

### Adopted from the reference but deliberately NOT added
- **`transformers.Trainer` / `TrainingArguments`** — a production wrapper (logging,
  distributed checkpointing); scientifically irrelevant to a P/S comparison, adds
  dependencies and complexity.
- **Length-weighted window sampling** (`ExpectedNumInstanceSampler`) — it changes the data
  distribution *identically for every P/S config*, so it does not help isolate P/S; it only
  raises fidelity to the official pre-training regime, which this study explicitly does not
  claim to reproduce. We use uniform random window extraction instead.

### Project scaffolding written here (untagged in the source)
Everything not tagged `# [CHRONOS-REF]` is written for this study, mostly to make the first (remote,
unattended) run safe and the results comparable:
- **bfloat16 autocast** when supported (fp16 + `GradScaler` fallback, fp32 on CPU): a
  T5-backbone trained from scratch is prone to fp16 overflow → NaN.
- **NaN handling**: unobserved samples are zero-filled while the real mask is preserved;
  windows with a fully-unobserved context or target are skipped.
- **Early abort on non-finite loss**: one bad run fails fast instead of burning hours; the
  sweep continues with the next config.
- **The `(P, S, seed)` sweep loop, per-run seeding, resumability (`DONE` marker), the
  per-model `run_config.json` provenance, and the aggregate `manifest.csv`.**
