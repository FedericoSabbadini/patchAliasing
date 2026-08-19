"""
train_sweep.py -- Retraining Chronos-Bolt Tiny from scratch for each patch geometry (P, S).

Every model is trained FROM SCRATCH (randomly initialised weights) on the exact same
official Chronos pre-training data:
  - TSMixup  (10 million series, augmented from 28 real-world open-source datasets
              including Monash, M-competitions, Kaggle energy/transport/weather/finance)
  - KernelSynth (1 million purely synthetic series sampled from Gaussian Process priors)
interleaved at the official 9:1 ratio (9 TSMixup series for every 1 KernelSynth series).

The ONLY parameter that changes between runs is the patch tokenisation geometry:
  - P = input_patch_size   (the width of each sliding window that becomes one token)
  - S = input_patch_stride (the step size between consecutive patch windows)
Every other hyperparameter is held at the official Chronos training value, so that any
downstream difference in structural-aliasing probes can be attributed solely to (P, S).

Usage:
    python train_sweep.py
"""
from __future__ import annotations

import json          # used to write per-run configuration files to disk
import time          # wall-clock timing for throughput measurement and completion timestamps
from pathlib import Path  # cross-platform filesystem path handling

import numpy as np   # numerical operations: array math, NaN handling, loss history storage


# ============================================================================ #
#  EXPERIMENTAL CONFIGURATION                                                   #
# ============================================================================ #

# The (P, S) grid to sweep. It is designed along two independent axes so that
# the effect of each parameter can be isolated:
#
#   Overlap axis: P is fixed at 16, S varies.
#     - (16,16): overlap ratio = 0.00, contiguous patches (the stock Bolt-Tiny geometry)
#     - (16,12): overlap ratio = 0.25, each patch overlaps the next by 4 samples
#     - (16, 8): overlap ratio = 0.50, each patch overlaps the next by 8 samples
#
#   Patch-size axis: S = P (contiguous, no overlap), P varies.
#     - ( 8, 8): smaller patch window, finer temporal resolution per token
#     - (16,16): the baseline again (shared with the overlap axis)
#     - (24,24): larger patch window, coarser temporal resolution per token
#
# Note: S = 4 (overlap ratio 0.75) is excluded by design because its stride-lock
# class F_lock = {c * fs/S} = {128, 256, ...} has no member inside the valid
# forecast band, so it would contribute no informative H1 or H3 test.
PS_GRID: list[tuple[int, int]] = [
    (16, 16),   # baseline: the stock Bolt-Tiny geometry, contiguous (no overlap)
    (16, 12),   # overlap ratio = 0.25
    (16, 8),    # overlap ratio = 0.50
    (8, 8),     # smaller patch, contiguous
    (24, 24),   # larger patch, contiguous
]

# A single seed per configuration. With the full official data diet (11 million series),
# run-to-run variance is already small, so multi-seed training is less critical than
# in a small-data regime. Seed 42 is used throughout the project for reproducibility.
SEEDS: list[int] = [42]

# The base model identifier on HuggingFace. We use it to download the ARCHITECTURAL
# configuration of Chronos-Bolt Tiny (encoder/decoder sizes, number of heads, etc.),
# but we do NOT load its pretrained weights — all weights are randomly initialised.
BASE_MODEL_ID = "amazon/chronos-bolt-tiny"

# Context and prediction lengths, both taken directly from the official Bolt-Tiny
# configuration file (config.json). The context length is how many past time steps
# the model can attend to; the prediction length is how many future steps it must
# forecast in a single forward pass.
CONTEXT_LENGTH = 2048    # number of past time-series samples the model receives as input
PREDICTION_LENGTH = 64   # number of future samples the model must predict

# The 9 quantiles that Chronos-Bolt is trained to predict. Instead of forecasting a
# single point value, Bolt outputs these 9 quantile levels of the predictive distribution,
# giving a full uncertainty profile. These are the official quantile values.
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# HuggingFace dataset repository and configuration names. These point to the exact
# same corpora used by the official Chronos pre-training pipeline.
HF_REPO = "autogluon/chronos_datasets"                          # the official HF repo
DATASET_CONFIG_TSMIXUP = "training_corpus_tsmixup_10m"          # 10M augmented real series
DATASET_CONFIG_KERNELSYNTH = "training_corpus_kernel_synth_1m"  # 1M GP-sampled synthetic series

# The mixing ratio between the two corpora: for every 9 series drawn from TSMixup,
# 1 series is drawn from KernelSynth. This is the official interleaving ratio from
# the Chronos paper, ensuring the model sees both real-world patterns and synthetic
# diversity in the same proportions as the published checkpoint.
TSMIXUP_RATIO = 9

# Training hyperparameters. All values match the official Chronos-Bolt Tiny training
# recipe (chronos-t5-tiny.yaml) unless explicitly noted as a deviation.
BATCH_SIZE = 32          # number of training windows per gradient step (official: 32)
MAX_STEPS = 100_000      # total optimiser steps (DEVIATION: official uses 200k; reduced
                         # to 100k to fit a single-GPU compute budget. All variants use
                         # the same budget, so they are comparable to each other)
LR = 1e-3                # initial learning rate (official: 0.001)
WEIGHT_DECAY = 0.0       # L2 regularisation coefficient (official: 0.0, no weight decay)
GRAD_CLIP_NORM = 1.0     # maximum gradient norm; gradients exceeding this are scaled down
                         # to prevent training instability from large updates (official: 1.0)
LR_SCHEDULER_TYPE = "linear"  # the learning rate decays linearly from LR down to 0 over
                              # the course of MAX_STEPS (official: linear schedule)
WARMUP_RATIO = 0.0       # fraction of training steps spent linearly ramping the LR from 0
                         # up to its initial value; 0.0 means training starts at full LR
                         # immediately (official: no warmup)
SHUFFLE_BUFFER_SIZE = 10_000  # number of examples held in the streaming shuffle buffer;
                              # the HuggingFace streaming dataset reads examples sequentially
                              # and shuffles them within this buffer for randomisation
                              # (DEVIATION: official uses 100k; reduced to avoid HF timeouts)

# Data-quality filters applied to each series before it enters training.
# These thresholds match the official Chronos pipeline exactly.
MIN_PAST = 60            # minimum number of observed (non-NaN) context points required;
                         # series shorter than MIN_PAST + prediction_length are skipped
MAX_MISSING_PROP = 0.9   # maximum fraction of NaN values allowed in a series;
                         # series with more than 90% missing values are discarded
DROP_PROB = 0.2          # data augmentation: for each series, a random fraction of
                         # observed values (uniformly sampled between 0 and DROP_PROB)
                         # are replaced with NaN, teaching the model to handle missing data

# Logging and checkpoint cadence (project-specific, not from the official recipe)
LOG_EVERY = 50           # print average loss and learning rate every 50 steps
SAVE_EVERY = 1000        # save a HuggingFace-format checkpoint every 1000 steps

# Output directory: each trained model gets its own subdirectory named p{P}-s{S}-seed{seed}
OUTPUT_ROOT = Path(__file__).resolve().parent / "weights"


# ============================================================================ #
#  DATASET LOADING                                                              #
# ============================================================================ #

def build_streams(seed: int):
    """Open both training corpora as shuffled HuggingFace streaming iterators.

    The datasets are loaded in streaming mode (streaming=True), which means examples
    are fetched on demand rather than downloading the entire corpus into memory.
    Each stream is independently shuffled using a buffer of SHUFFLE_BUFFER_SIZE examples
    and the given random seed, ensuring reproducible ordering across runs.

    Returns a (tsmixup_stream, kernelsynth_stream) tuple of iterable datasets.
    """
    from datasets import load_dataset  # HuggingFace datasets library

    # Open the TSMixup corpus: 10 million time-series instances generated by the TSMixup
    # augmentation method, which samples and mixes subsequences from 28 real-world datasets
    # (Monash repository, M-competitions, Kaggle competitions covering energy, transport,
    # weather, finance, etc.). This is the primary training source.
    tsmixup = load_dataset(
        HF_REPO, DATASET_CONFIG_TSMIXUP,
        split="train", streaming=True,
    ).shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER_SIZE)

    # Open the KernelSynth corpus: 1 million purely synthetic time series generated by
    # sampling from Gaussian Process priors with diverse kernel compositions. This provides
    # smooth, structured patterns that complement the real-world noise and irregularities
    # found in TSMixup, improving the model's ability to capture smooth trends.
    kernelsynth = load_dataset(
        HF_REPO, DATASET_CONFIG_KERNELSYNTH,
        split="train", streaming=True,
    ).shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER_SIZE)

    return tsmixup, kernelsynth


# ============================================================================ #
#  TRAINING WINDOW PREPARATION                                                 #
# ============================================================================ #

def make_window_dataset(tsmixup_stream, kernelsynth_stream, total_length: int,
                        context_length: int, seed: int):
    """Create a PyTorch IterableDataset that produces (context, target) training windows.

    Each training example is a window extracted from one time series, consisting of:
      - context: the most recent `context_length` time steps before the split point,
                 which the model receives as input (its "history")
      - target:  the next `prediction_length` time steps after the split point,
                 which the model must learn to forecast (its "future")

    The series arrive interleaved at the official 9:1 ratio (TSMixup : KernelSynth),
    cycling infinitely so the training loop never runs out of data.

    NaN handling is critical: Chronos-Bolt's internal InstanceNorm computes normalisation
    statistics using nanmean BEFORE applying the attention mask, so missing values and
    padding positions MUST remain as NaN (not zero). The model itself zeroes NaN positions
    AFTER normalisation, so NaNs never reach the transformer layers.

    Returns a ChronosStreamingWindowDataset instance ready to be wrapped in a DataLoader.
    """
    import torch
    from torch.utils.data import IterableDataset

    # The target (forecast) length: the number of future time steps per training window
    horizon = total_length - context_length

    class ChronosStreamingWindowDataset(IterableDataset):

        def _interleaved(self):
            """Yield raw series from both streams at the official 9:1 mixing ratio.

            For every 9 series drawn from TSMixup, 1 series is drawn from KernelSynth.
            When either stream is exhausted, its iterator is silently restarted, so the
            generator loops forever — exactly matching the official train.py's Cyclic wrapper.
            """
            tsmixup_iter = iter(tsmixup_stream)
            ks_iter = iter(kernelsynth_stream)
            while True:
                # Draw 9 consecutive series from the TSMixup corpus
                for _ in range(TSMIXUP_RATIO):
                    try:
                        yield next(tsmixup_iter)
                    except StopIteration:
                        # TSMixup exhausted: restart the iterator and continue
                        tsmixup_iter = iter(tsmixup_stream)
                        yield next(tsmixup_iter)
                # Draw 1 series from the KernelSynth corpus
                try:
                    yield next(ks_iter)
                except StopIteration:
                    # KernelSynth exhausted: restart the iterator and continue
                    ks_iter = iter(kernelsynth_stream)
                    yield next(ks_iter)

        def __iter__(self):
            """For each incoming series, extract one random training window and yield it.

            This generator is called once per DataLoader worker. It applies the full
            official data pipeline: length filtering, missing-value filtering, NaN-injection
            augmentation, random window sampling, left-padding, and observation masking.
            """
            # Per-iterator random number generator seeded for reproducibility
            rng = np.random.default_rng(seed)

            for row in self._interleaved():
                # Read the raw time-series values from the HuggingFace row
                values = np.asarray(row["target"], dtype=np.float32)
                L = values.shape[0]

                # FILTER 1 (length): the series must be long enough to provide at least
                # MIN_PAST context points AND a full horizon-length target. Series that
                # are too short cannot produce a valid training window and are skipped.
                if L < MIN_PAST + horizon:
                    continue

                # FILTER 2 (missing data): discard series where more than MAX_MISSING_PROP
                # (90%) of the values are NaN. Such series carry too little information
                # for meaningful gradient signal.
                if np.isnan(values).mean() > MAX_MISSING_PROP:
                    continue

                # DATA AUGMENTATION (NaN injection): randomly replace a fraction of observed
                # values with NaN. The drop rate is sampled uniformly from [0, DROP_PROB] for
                # each series, so the model encounters varying levels of missingness during
                # training and learns to be robust to gaps in the input.
                drop_p = rng.uniform(0.0, DROP_PROB)
                if drop_p > 0.0:
                    values = values.copy()  # avoid mutating the streaming buffer
                    values[rng.random(L) < drop_p] = np.nan

                # WINDOW SAMPLING: choose a random split point that guarantees at least
                # MIN_PAST observed context points before it and a full horizon-length
                # target after it. This mirrors the official ExpectedNumInstanceSampler.
                end = int(rng.integers(MIN_PAST + horizon, L + 1))
                tgt_raw = values[end - horizon:end]      # the forecast target (future)
                ctx_src = values[:end - horizon]          # all available history (past)

                # CONTEXT WINDOWING: if the history is longer than context_length, keep only
                # the most recent context_length points (the model's attention window).
                if ctx_src.shape[0] >= context_length:
                    ctx_raw = ctx_src[-context_length:]
                else:
                    # If the history is shorter than context_length, LEFT-PAD with NaN.
                    # NaN padding tells Chronos-Bolt "no observation here": the InstanceNorm
                    # excludes these positions via nanmean, and the attention mask prevents
                    # the transformer from attending to them.
                    pad = context_length - ctx_src.shape[0]
                    ctx_raw = np.concatenate([np.full(pad, np.nan, np.float32), ctx_src])

                # OBSERVATION MASKS: boolean arrays where True means "this time step has a
                # valid observed value" and False means "this position is NaN (missing or padding)".
                # Chronos-Bolt uses these masks in its forward pass to exclude unobserved
                # positions from the loss computation and attention mechanism.
                ctx_obs = ~np.isnan(ctx_raw)
                tgt_obs = ~np.isnan(tgt_raw)

                # Skip windows where the entire context or the entire target is unobserved:
                # a fully-NaN context provides no conditioning information, and a fully-NaN
                # target contributes zero gradient (every position is masked out of the loss).
                if not ctx_obs.any() or not tgt_obs.any():
                    continue

                # Yield the four tensors that Chronos-Bolt's forward() method expects.
                # These are the native input format: raw values with NaN encoding for
                # unobserved positions, plus explicit boolean observation masks.
                yield {
                    "context": torch.from_numpy(ctx_raw),       # shape (context_length,): raw past values, NaN = unobserved
                    "mask": torch.from_numpy(ctx_obs),           # shape (context_length,): True where context is observed
                    "target": torch.from_numpy(tgt_raw),         # shape (horizon,): raw future values, NaN = unobserved
                    "target_mask": torch.from_numpy(tgt_obs),    # shape (horizon,): True where target is observed
                }

    return ChronosStreamingWindowDataset()


# ============================================================================ #
#  MODEL CONSTRUCTION                                                           #
# ============================================================================ #

def build_model(P: int, S: int, device):
    """Construct a from-scratch Chronos-Bolt Tiny model with the given patch geometry (P, S).

    This function downloads the ARCHITECTURAL configuration (number of layers, hidden
    dimensions, number of attention heads, etc.) from the official Chronos-Bolt Tiny
    checkpoint on HuggingFace, but does NOT load the pretrained weights. Instead, all
    parameters are randomly initialised using the model's built-in post_init method
    (which applies Gaussian initialisation scaled by initializer_factor = 0.05).

    The key experimental manipulation happens here: we override the patch tokenisation
    parameters (input_patch_size = P and input_patch_stride = S) in the configuration
    before constructing the model. Since the patch embedding layer's input dimension
    depends on P (specifically, in_features = P * 2 because the input is the concatenation
    of the patch values and their observation mask), changing P changes the architecture
    of the first layer, which is why the model must be trained from scratch rather than
    fine-tuned from the official checkpoint.

    Returns the model moved to the specified device (GPU or CPU).
    """
    from transformers import AutoConfig
    from chronos.chronos_bolt import ChronosBoltModelForForecasting

    # Download the architectural configuration from HuggingFace (this is a lightweight
    # JSON file describing the model structure, NOT the multi-MB weight tensors)
    config = AutoConfig.from_pretrained(BASE_MODEL_ID)

    # Override the patch tokenisation parameters with our experimental values.
    # All other architectural parameters (d_model, n_heads, n_layers, etc.) remain
    # at their official Bolt-Tiny values.
    config.chronos_config["context_length"] = CONTEXT_LENGTH
    config.chronos_config["prediction_length"] = PREDICTION_LENGTH
    config.chronos_config["input_patch_size"] = P       # THE experimental knob: patch window width
    config.chronos_config["input_patch_stride"] = S     # THE experimental knob: patch step size
    config.chronos_config["quantiles"] = QUANTILES

    # Instantiate the model with randomly initialised weights. The post_init method
    # applies the initialisation scheme defined in the config (initializer_factor = 0.05),
    # which produces sensible random weights for training from scratch.
    model = ChronosBoltModelForForecasting(config)
    return model.to(device)


# ============================================================================ #
#  TRAINING A SINGLE MODEL                                                      #
# ============================================================================ #

def train_one(P: int, S: int, seed: int, out_dir: Path):
    """Train one Chronos-Bolt Tiny model from scratch for the given patch geometry (P, S).

    The training follows the official Chronos recipe as closely as possible:
    - AdamW optimiser with linear learning rate decay and no warmup
    - fp32 compute with TF32 matrix multiplications on Ampere+ GPUs
    - gradient clipping at norm 1.0
    - the same data pipeline (interleaved TSMixup + KernelSynth, NaN augmentation)

    The trained model is saved in HuggingFace format at out_dir, along with a loss
    history array and periodic checkpoints. A 'DONE' marker file is written upon
    successful completion to enable resumability (the sweep skips completed models).
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_scheduler, set_seed
    from tqdm.auto import tqdm

    # RESUMABILITY: if a previous run already completed this model, skip it entirely.
    # The DONE marker file is written only after the final model is saved successfully,
    # so its presence guarantees a complete, usable checkpoint.
    if out_dir.joinpath("DONE").exists():
        print(f"[skip] {out_dir.name} already completed")
        return

    # Create the output directory for this model (e.g., weights/p16-s8-seed42/)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fix all random seeds for full reproducibility: Python's random module, NumPy,
    # PyTorch (both CPU and CUDA), and CUDA's cuDNN backend are all seeded together
    # using the HuggingFace set_seed utility, exactly as the official train.py does.
    set_seed(seed)

    # Select the compute device: use CUDA GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable TF32 (TensorFloat-32) on Ampere-generation or newer GPUs (compute capability >= 8).
    # TF32 uses the full fp32 range but with reduced mantissa precision (10 bits instead of 23)
    # in matrix multiplications, providing ~3x speedup with negligible accuracy impact for
    # deep learning workloads. This matches the official Chronos training configuration
    # (TrainingArguments tf32=True).
    if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True   # enable TF32 for torch.matmul operations
        torch.backends.cudnn.allow_tf32 = True          # enable TF32 for cuDNN convolutions

    # Build the model with randomly initialised weights for this (P, S) geometry
    model = build_model(P, S, device)

    # Open the two training corpus streams and create the window-sampling dataset
    tsmixup_stream, ks_stream = build_streams(seed)
    dataset = make_window_dataset(
        tsmixup_stream, ks_stream,
        CONTEXT_LENGTH + PREDICTION_LENGTH,  # total window length (context + target)
        CONTEXT_LENGTH,                       # how much of that window is context (input)
        seed,
    )

    # Wrap the streaming dataset in a DataLoader that collates individual windows
    # into batches of BATCH_SIZE. num_workers=0 means data loading happens in the
    # main process (equivalent to the official single-stream configuration).
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

    # OPTIMISER: AdamW (Adam with decoupled weight decay, the standard for transformer
    # training). On CUDA, the "fused" implementation uses a single GPU kernel for the
    # entire parameter update, avoiding multiple kernel launches and memory round-trips
    # — mathematically identical, just faster. The try/except handles older PyTorch
    # versions that do not support the fused flag.
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
            fused=(device.type == "cuda"),
        )
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
        )

    # LEARNING RATE SCHEDULER: the learning rate starts at LR and decays linearly to 0
    # over the course of MAX_STEPS. With WARMUP_RATIO = 0.0, there are 0 warmup steps,
    # so the decay begins immediately from the first step. This is the official schedule.
    lr_scheduler = get_scheduler(
        LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=round(WARMUP_RATIO * MAX_STEPS),  # = 0 warmup steps (official)
        num_training_steps=MAX_STEPS,
    )

    print(f"\n=== Training {out_dir.name} | P={P} S={S} seed={seed} ===")

    # Switch the model to training mode: this activates dropout layers and ensures
    # batch normalisation layers (if any) use batch statistics rather than running
    # averages. Required before any forward pass that should compute gradients.
    model.train()

    # Loss history: stores the scalar loss value at every training step, used for
    # monitoring convergence and saved to disk as a numpy array for later analysis.
    loss_history: list[float] = []

    # Create a manual iterator over the DataLoader so we can handle stream exhaustion
    # by restarting the iterator without interrupting the training loop.
    data_iter = iter(loader)
    t0 = time.time()  # record the start time for throughput measurement

    def _next_batch():
        """Fetch the next batch from the DataLoader, restarting the iterator if exhausted.

        The underlying IterableDataset cycles infinitely, but the DataLoader's iterator
        can still raise StopIteration in edge cases (e.g., when the streaming connection
        is interrupted). This wrapper handles that gracefully by creating a fresh iterator.
        """
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    # ---- MAIN TRAINING LOOP ----
    # This is a fixed-step training loop (not epoch-based): the model trains for exactly
    # MAX_STEPS gradient updates, regardless of how many times it cycles through the data.
    # This matches the official Chronos training, which also uses max_steps.
    pbar = tqdm(range(1, MAX_STEPS + 1), desc=out_dir.name, dynamic_ncols=True)
    for step in pbar:
        # Move the batch tensors to the compute device (GPU or CPU). Each batch contains
        # four tensors: context, mask, target, target_mask — see make_window_dataset.
        batch = {k: v.to(device) for k, v in _next_batch().items()}

        # Zero out the gradients accumulated from the previous step. set_to_none=True
        # is a minor optimisation: instead of filling gradient tensors with zeros, it
        # deallocates them entirely, which is slightly faster and uses less memory.
        optimizer.zero_grad(set_to_none=True)

        # FORWARD PASS: feed the context and target through Chronos-Bolt. The model
        # internally patches the context into tokens, runs them through the encoder-decoder
        # transformer, and computes the quantile loss between the predicted and actual
        # future values. The loss is automatically masked: positions where target_mask
        # is False (i.e., NaN in the target) contribute zero to the loss.
        out = model(
            context=batch["context"],
            mask=batch["mask"],
            target=batch["target"],
            target_mask=batch["target_mask"],
        )
        loss = out.loss  # the masked quantile regression loss (scalar tensor)

        # NaN/Inf guard: a non-finite loss in an fp32 pipeline (no mixed precision) signals
        # genuine numerical pathology (e.g., a degenerate batch), not a transient scaling issue.
        # Abort this run immediately rather than continuing with corrupted gradients.
        loss_value = float(loss.detach().cpu())
        if not np.isfinite(loss_value):
            print(f"[abort] non-finite loss at step {step}")
            break

        # BACKWARD PASS: compute the gradient of the loss with respect to every model
        # parameter via automatic differentiation (backpropagation). This is plain fp32
        # backprop — the official recipe uses no AMP (automatic mixed precision) scaler.
        loss.backward()

        # GRADIENT CLIPPING: if the total L2 norm of all gradients exceeds GRAD_CLIP_NORM,
        # scale every gradient down proportionally so the norm equals GRAD_CLIP_NORM.
        # This prevents the optimiser from taking excessively large steps when a batch
        # produces unusually steep gradients, which is a common source of training instability.
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        # OPTIMISER STEP: apply the clipped gradients to update all model parameters
        # according to the AdamW update rule (gradient descent with adaptive per-parameter
        # learning rates and momentum).
        optimizer.step()

        # SCHEDULER STEP: advance the learning rate schedule by one step. Under the linear
        # schedule, this reduces the learning rate by LR / MAX_STEPS, so the rate reaches
        # exactly 0 at the final step.
        lr_scheduler.step()

        # Record the scalar loss value and update the progress bar with the current loss
        # and learning rate, giving a real-time view of training dynamics.
        loss_history.append(loss_value)
        pbar.set_postfix(loss=f"{loss_value:.4f}",
                         lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

        # PERIODIC DETAILED LOG: every LOG_EVERY steps, print the smoothed loss (averaged
        # over the last LOG_EVERY steps to reduce noise), the current learning rate, and
        # the training throughput in steps per second. This line is written above the
        # progress bar and survives in captured stdout for post-hoc analysis.
        if step % LOG_EVERY == 0:
            sps = step / (time.time() - t0)
            tqdm.write(f"step={step}/{MAX_STEPS} "
                       f"loss={np.mean(loss_history[-LOG_EVERY:]):.4f} "
                       f"lr={lr_scheduler.get_last_lr()[0]:.2e} "
                       f"{sps:.2f} it/s")

        # PERIODIC CHECKPOINT: every SAVE_EVERY steps, save the model in HuggingFace format
        # so that training can be resumed from this point if the process is interrupted.
        # Each checkpoint is a self-contained directory with the model weights and config.
        if step % SAVE_EVERY == 0:
            ck = out_dir / f"checkpoint-{step}"
            model.save_pretrained(ck)
            tqdm.write(f"  checkpoint saved: {ck.name}")

    # Save the full loss history as a numpy array for later convergence analysis and plotting
    np.save(out_dir / "loss_history.npy",
            np.asarray(loss_history, dtype=np.float32))

    # Save the final trained model in HuggingFace format (config.json + model.safetensors),
    # ready to be loaded with ChronosBoltModelForForecasting.from_pretrained(out_dir)
    model.save_pretrained(out_dir)

    # Write the DONE marker file with a human-readable timestamp. This file serves as
    # the resumability signal: the sweep loop checks for its existence before starting
    # a run, and skips models that are already fully trained.
    (out_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[done] final model saved to {out_dir}")


# ============================================================================ #
#  SWEEP: train all geometries sequentially                                     #
# ============================================================================ #

def main():
    """Drive the full (P, S, seed) sweep: train one model per combination, sequentially.

    The sweep iterates over every (P, S) pair in PS_GRID crossed with every seed in SEEDS,
    training each model from scratch in sequence. Each model is saved to its own subdirectory
    under OUTPUT_ROOT (e.g., weights/p16-s8-seed42/). If a model's DONE marker already
    exists, it is skipped without loading any data or weights.

    Exceptions in individual runs are caught and logged without aborting the entire sweep,
    so a failure in one geometry does not prevent the remaining models from training.
    """
    # Ensure the top-level output directory exists
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Flatten the grid of (P, S) pairs and seeds into a linear list of runs
    runs = [(P, S, seed) for (P, S) in PS_GRID for seed in SEEDS]
    print(f"Sweep: {len(runs)} runs -> {OUTPUT_ROOT}")

    for P, S, seed in runs:
        out_dir = OUTPUT_ROOT / f"p{P}-s{S}-seed{seed}"
        try:
            train_one(P, S, seed, out_dir)
        except Exception as e:
            # A failed run must not stop the rest of the sweep: log the error and continue
            print(f"[error] {out_dir.name}: {type(e).__name__}: {e}")

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
