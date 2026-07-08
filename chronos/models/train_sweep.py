"""
train_sweep.py — Chronos-Bolt Tiny from-scratch (P, S, seed) sweep driver. FINAL VERSION.

Training only (no evaluation/inference). Each run retrains Chronos-Bolt Tiny FROM SCRATCH
(random weights) on the official Chronos pre-training corpus
(autogluon/chronos_datasets / training_corpus_tsmixup_10m, streamed), varying ONLY the
patch geometry input_patch_size (P) and input_patch_stride (S). Everything else is fixed
at the official Chronos training values so any downstream difference in
structural-aliasing probes is attributable to P/S.

Just run the file:

    python train_sweep.py

Edit the CONFIG block below (PS_GRID, SEEDS, hyperparameters) to change the experiment.
The sweep is resumable: a run that already wrote a DONE marker is skipped.
For a quick smoke test set MAX_STEPS low (e.g. 20) and SHUFFLE_BUFFER_SIZE low (e.g. 100).

Outputs: one folder per model at chronos/outputs/models/p{P}-s{S}-seed{seed}/, holding
EVERY artifact tied to that model (run_config.json, loss_history.npy, loss_curve.png,
checkpoint-{step}/, final model, DONE). An aggregate chronos/outputs/models/manifest.csv
is rebuilt from every finished run.

---------------------------------------------------------------------------------------
Provenance markers on code lines:
  "# [CHRONOS-REF] ..." -> the line's approach, API, or VALUE is taken from the official
               Chronos reference (Apache-2.0): github.com/amazon-science/chronos-forecasting
               (scripts/training/train.py + scripts/training/configs/chronos-t5-tiny.yaml)
               and the chronos.chronos_bolt library source. Adopted/adapted, NOT copied.
  plain "# ..." -> scaffolding written for this project (sweep loop, resume, provenance,
               NaN-loss guard, manifest, progress bar).
Deliberate deviations from the official regime (uniform across all runs, documented in
README_retraining.md): MAX_STEPS 10k vs 200k (compute budget; variants are compared only
against each other), LOG/SAVE cadence, num_workers=0 (single-stream equivalence),
torch_compile off (startup overhead across 18 short runs; no math change).
---------------------------------------------------------------------------------------
NaN CONTRACT (critical, verified against chronos_bolt.py source): Chronos-Bolt encodes
"unobserved" as NaN. InstanceNorm computes loc/scale with nanmean BEFORE the mask is
used, so padding/missing values MUST be NaN — zero-filling them would pollute the
normalization statistics. The model itself zeroes NaN positions AFTER normalization
(patched_context = where(patched_mask > 0, ., 0.0)), so NaNs never reach the network.
---------------------------------------------------------------------------------------
"""
from __future__ import annotations          # allow modern type hints on older runtimes

import json                                 # write/read per-run run_config.json + manifest
import subprocess                           # shell out to `git` to record the commit hash
import time                                 # wall-clock timing + DONE timestamp
from dataclasses import dataclass, asdict   # typed per-run result record -> dict for JSON/CSV
from pathlib import Path                     # filesystem paths (output dirs, markers)

import numpy as np                          # [CHRONOS-REF] array math for windows (train.py also uses numpy)


# ============================================================================ #
#  CONFIG — the only things to edit                                            #
# ============================================================================ #
# (P, S) grid. Two-axis design (S <= P everywhere, so no unobserved gaps):
#   - P=16 row varies S (overlap/stride axis):  or = (P-S)/P = 0, .25, .5, .75
#   - contiguous row (or=0) varies P (patch-size axis): P = 8, 16, 24
PS_GRID: list[tuple[int, int]] = [          # our experimental design (not from Chronos)
    (16, 16),   # baseline, contiguous (16/16 is the stock Bolt-tiny geometry)  # [CHRONOS-REF]
    (16, 12),   # or = 0.25
    (16, 8),    # or = 0.50
    (16, 4),    # or = 0.75  (heaviest: ~509 patches @ context 2048)
    (8, 8),     # patch-size axis, contiguous
    (24, 24),   # patch-size axis, contiguous
]
SEEDS: list[int] = [42, 43, 44]             # 3 seeds per config to separate P/S effect from seed noise

BASE_MODEL_ID = "amazon/chronos-bolt-tiny"  # [CHRONOS-REF] official Bolt-tiny checkpoint (architecture ref only; weights NOT loaded)

CONTEXT_LENGTH = 2048                        # [CHRONOS-REF] Bolt-tiny config value (config.json: context_length)
PREDICTION_LENGTH = 64                       # [CHRONOS-REF] Bolt-tiny config value (config.json: prediction_length)
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # [CHRONOS-REF] the 9 quantiles Bolt is trained on

HF_REPO = "autogluon/chronos_datasets"       # [CHRONOS-REF] official Chronos datasets repository on HuggingFace
DATASET_CONFIG = "training_corpus_tsmixup_10m"  # [CHRONOS-REF] official TSMixup pre-training corpus (10M series)

BATCH_SIZE = 32                              # [CHRONOS-REF] per_device_train_batch_size=32, grad-accum=1 (chronos-t5-tiny.yaml)
MAX_STEPS = 10_000                           # DEVIATION: 10k vs official 200k — fixed compute budget, identical across runs
LR = 1e-3                                     # [CHRONOS-REF] learning_rate: 0.001 (chronos-t5-tiny.yaml)
WEIGHT_DECAY = 0.0                            # [CHRONOS-REF] official uses HF Trainer default weight_decay=0.0 (no override anywhere)
GRAD_CLIP_NORM = 1.0                          # [CHRONOS-REF] HF Trainer default max_grad_norm=1.0, implicitly used by train.py
LR_SCHEDULER_TYPE = "linear"                  # [CHRONOS-REF] lr_scheduler_type: linear (chronos-t5-tiny.yaml)
WARMUP_RATIO = 0.0                            # [CHRONOS-REF] warmup_ratio: 0.0 (chronos-t5-tiny.yaml) — official trains from scratch WITHOUT warmup
SHUFFLE_BUFFER_SIZE = 10_000                  # reduced from official 100k to avoid HF streaming timeouts; still provides good randomisation for 10k-step runs
MIN_PAST = 60                                 # [CHRONOS-REF] min_past: 60 (chronos-t5-tiny.yaml) — window sampler requires >= 60 context points
MAX_MISSING_PROP = 0.9                        # [CHRONOS-REF] max_missing_prop: 0.9 — drop series with > 90% missing values
DROP_PROB = 0.2                               # [CHRONOS-REF] drop_prob=0.2 (train.py ChronosDataset default): random NaN injection augmentation
LOG_EVERY = 50                               # console logging cadence (our diagnostic; official log_steps=500)
SAVE_EVERY = 1000                            # checkpoint cadence in steps (our diagnostic; official save_steps=100k)

# All trained models + artifacts go under the shared chronos/outputs tree.
# This file is chronos/models/train_sweep.py, so parent.parent is chronos/.
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "models"


# ============================================================================ #
#  Provenance helpers                                                           #
# ============================================================================ #
def _git_commit() -> str:                    # record which commit produced a run (reproducibility)
    try:                                     # git may be absent or this may not be a repo
        return subprocess.check_output(      # ask git for the current HEAD hash
            ["git", "rev-parse", "HEAD"],    # the command
            cwd=Path(__file__).resolve().parent,  # run it from this file's directory
            stderr=subprocess.DEVNULL,       # silence git's error chatter
        ).decode().strip()                   # bytes -> str, drop trailing newline
    except Exception:                        # any failure -> unknown, never crash the run
        return "unknown"


def _approx_num_patches(context_length: int, P: int, S: int) -> int:
    """Patch count over a full context (provenance only; +1 [REG] token goes to the encoder)."""  # [CHRONOS-REF] Patch pads then unfolds; use_reg_token appends 1 token
    l_pad = -(-context_length // P) * P      # [CHRONOS-REF] Patch left-pads with NaN up to a multiple of P (chronos_bolt.Patch.forward)
    return (l_pad - P) // S + 1              # [CHRONOS-REF] unfold(size=P, step=S) window count over the padded context


# ============================================================================ #
#  Dataset                                                                       #
# ============================================================================ #
def build_stream(seed: int):                 # open the HF corpus as a shuffled streaming iterator
    from datasets import load_dataset        # [CHRONOS-REF] HuggingFace datasets (the corpus lives there)
    stream = load_dataset(                    # [CHRONOS-REF] open the official corpus...
        HF_REPO, DATASET_CONFIG,             # [CHRONOS-REF] ...repo + config selected above
        split="train", streaming=True,       # [CHRONOS-REF] stream instead of downloading the whole 10M-series corpus
    )
    return stream.shuffle(                    # [CHRONOS-REF] shuffle the training stream (train.py PseudoShuffledIterableDataset)...
        seed=seed, buffer_size=SHUFFLE_BUFFER_SIZE,  # [CHRONOS-REF] ...with the official buffer length (100k)
    )


def make_window_dataset(hf_stream, total_length: int, context_length: int, seed: int):
    import torch                              # tensors are built here, per worker
    from torch.utils.data import IterableDataset  # streaming source has no length -> IterableDataset

    horizon = total_length - context_length  # = PREDICTION_LENGTH (target length per instance)

    class ChronosStreamingWindowDataset(IterableDataset):  # yields ready-to-forward training windows
        """Yield (context, mask, target, target_mask) windows from the HF TSMixup stream.

        Mirrors the official train.py pipeline (ExpectedNumInstanceSampler + InstanceSplitter
        + FilterTransformation), adapted from GluonTS to a plain HF stream:
          - series with < MIN_PAST + horizon points or > MAX_MISSING_PROP missing are dropped;
          - random NaN-injection augmentation with per-series rate ~ U(0, DROP_PROB);
          - one window per series pass: split point uniform with >= MIN_PAST context points
            and a full horizon-length future;
          - context left-padded to context_length with NaN (InstanceSplitter dummy_value=nan);
          - instances with zero observed context points are filtered out.
        NaNs are KEPT (never zero-filled): Bolt's InstanceNorm excludes them via nanmean and
        the model zeroes them after normalization — see NaN CONTRACT in the module docstring.
        """

        def __iter__(self):                  # generator called once per DataLoader worker
            rng = np.random.default_rng(seed)  # per-iterator RNG so window sampling is reproducible
            while True:                       # [CHRONOS-REF] cycle the corpus forever (train.py wraps datasets in Cyclic)
                for row in hf_stream:         # [CHRONOS-REF] iterate raw series from the official stream
                    if "target" not in row:   # schema guard: the value column must be present
                        raise KeyError(       # fail loudly with a helpful message if the schema changed
                            f"Row has no 'target' field (keys={list(row)}). "
                            f"Check the schema of {HF_REPO}/{DATASET_CONFIG}."
                        )
                    values = np.asarray(row["target"], dtype=np.float32)  # [CHRONOS-REF] the full series values
                    L = values.shape[0]        # series length
                    if L < MIN_PAST + horizon: # [CHRONOS-REF] has_enough_observations: min_length = min_past + prediction_length
                        continue               # too short -> next series
                    if np.isnan(values).mean() > MAX_MISSING_PROP:  # [CHRONOS-REF] has_enough_observations: max_missing_prop
                        continue               # mostly missing -> next series

                    drop_p = rng.uniform(0.0, DROP_PROB)  # [CHRONOS-REF] preprocess_entry: drop_p ~ U(0, drop_prob)
                    if drop_p > 0.0:           # [CHRONOS-REF] randomly turn observations into missing values (NaN)
                        values = values.copy() # do not mutate the stream's buffer
                        values[rng.random(L) < drop_p] = np.nan  # [CHRONOS-REF] element-wise drop with prob drop_p

                    end = int(rng.integers(MIN_PAST + horizon, L + 1))  # [CHRONOS-REF] split uniform with min_past context + full future (ExpectedNumInstanceSampler)
                    tgt_raw = values[end - horizon:end]  # the forecast target (length horizon; may contain NaN)
                    ctx_src = values[:end - horizon]     # all history before the target (length >= MIN_PAST)

                    if ctx_src.shape[0] >= context_length:      # enough history: keep the most recent context_length points
                        ctx_raw = ctx_src[-context_length:]      # [CHRONOS-REF] InstanceSplitter past_length window (most recent values)
                    else:                                        # [CHRONOS-REF] short context -> LEFT-pad to context_length with NaN (dummy_value=np.nan)
                        pad = context_length - ctx_src.shape[0]  # number of padding positions on the left
                        ctx_raw = np.concatenate([np.full(pad, np.nan, np.float32), ctx_src])  # NaN padding: excluded by nanmean, masked in attention

                    ctx_obs = ~np.isnan(ctx_raw)     # True where a context value is observed (padding/missing = False)
                    tgt_obs = ~np.isnan(tgt_raw)     # True where a target value is observed
                    if not ctx_obs.any():            # [CHRONOS-REF] FilterTransformation: >= 1 observed past point required
                        continue                     # nothing to condition on -> skip
                    if not tgt_obs.any():            # our guard: a fully-missing target contributes zero loss — skip the wasted sample
                        continue

                    yield {                   # one training example, keyed for the Bolt forward()
                        "context": torch.from_numpy(ctx_raw),        # [CHRONOS-REF] raw context, NaN = unobserved (Bolt's native encoding)
                        "mask": torch.from_numpy(ctx_obs),           # [CHRONOS-REF] observation mask for the context (Bolt forward arg)
                        "target": torch.from_numpy(tgt_raw),         # [CHRONOS-REF] raw target, NaN = unobserved (loss masks them)
                        "target_mask": torch.from_numpy(tgt_obs),    # [CHRONOS-REF] observation mask for the target (Bolt forward arg)
                    }

    return ChronosStreamingWindowDataset()   # instance the DataLoader will wrap


# ============================================================================ #
#  Model                                                                         #
# ============================================================================ #
def build_model(P: int, S: int, device):     # construct a from-scratch Bolt model for this (P, S)
    from transformers import AutoConfig       # [CHRONOS-REF] load the architecture config from HuggingFace
    from chronos.chronos_bolt import ChronosBoltModelForForecasting  # [CHRONOS-REF] the real Chronos-Bolt model class

    config = AutoConfig.from_pretrained(BASE_MODEL_ID)  # [CHRONOS-REF] pull Bolt-tiny's architectural config (weights NOT fetched; ships initializer_factor=0.05 for sane random init)
    config.chronos_config["context_length"] = CONTEXT_LENGTH        # [CHRONOS-REF] keep Bolt default (2048)
    config.chronos_config["prediction_length"] = PREDICTION_LENGTH  # [CHRONOS-REF] keep Bolt default (64)
    config.chronos_config["input_patch_size"] = P                   # [CHRONOS-REF] THE experimental knob: patch size P
    config.chronos_config["input_patch_stride"] = S                 # [CHRONOS-REF] THE experimental knob: patch stride S
    config.chronos_config["quantiles"] = QUANTILES                  # [CHRONOS-REF] keep Bolt's 9 quantile heads

    model = ChronosBoltModelForForecasting(config)  # [CHRONOS-REF] random weights via post_init (= train.py random_init path); required: embedding in_features = P*2 changes with P
    return model.to(device), config.chronos_config  # move to GPU/CPU; return the resolved sub-config for provenance


def _setup_precision(device) -> str:          # official regime: fp32 compute with TF32 matmuls on capable GPUs
    import torch                               # local import: torch only needed inside the run
    if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:  # [CHRONOS-REF] train.py: tf32=true only on compute capability >= 8
        torch.backends.cuda.matmul.allow_tf32 = True   # [CHRONOS-REF] enable TF32 matmuls (TrainingArguments tf32=True)
        torch.backends.cudnn.allow_tf32 = True         # [CHRONOS-REF] enable TF32 convs (same flag)
        return "fp32+tf32"                    # human-readable label for provenance
    return "fp32"                             # CPU or pre-Ampere GPU: plain fp32 (official fallback behaviour)


# ============================================================================ #
#  One run                                                                       #
# ============================================================================ #
@dataclass
class RunResult:                              # one row of the manifest: the outcome of a single run
    P: int                                    # patch size
    S: int                                    # stride
    seed: int                                 # seed
    overlap_ratio: float                      # (P - S) / P
    approx_num_patches: int                   # patch tokens per full context (excl. the +1 [REG] token)
    n_params_millions: float                  # model size actually built
    max_steps: int                            # planned optimizer steps
    steps_completed: int                      # steps actually run (< max_steps if aborted)
    final_loss: float                         # last step's loss
    mean_last_100: float                      # smoothed tail loss
    steps_per_sec: float                      # throughput (sanity-check time estimates)
    status: str                               # "done" | "failed-nan"
    precision: str                            # fp32 | fp32+tf32
    device: str                               # cuda | cpu


def train_one(P: int, S: int, seed: int, out_dir: Path) -> RunResult:
    import torch                               # heavy imports kept inside the run (clean sweep startup)
    from torch.utils.data import DataLoader    # batches windows from the IterableDataset
    from transformers import get_scheduler, set_seed  # [CHRONOS-REF] HF scheduler factory + official seeding helper (train.py uses transformers.set_seed)
    from tqdm.auto import tqdm                  # live progress bar over training steps

    if out_dir.joinpath("DONE").exists():      # resumability: this model is already fully trained
        print(f"[skip] {out_dir.name} already DONE")
        prev = json.loads(out_dir.joinpath("run_config.json").read_text())  # reload its recorded result
        return RunResult(**{k: prev["result"][k] for k in RunResult.__dataclass_fields__})  # reconstruct the record

    out_dir.mkdir(parents=True, exist_ok=True)  # create this model's dedicated folder

    set_seed(seed)                             # [CHRONOS-REF] seeds python/numpy/torch/cuda in one call, as train.py does

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pick GPU if present
    prec_name = _setup_precision(device)       # [CHRONOS-REF] fp32 + TF32 (official); no AMP, no GradScaler

    model, chronos_config = build_model(P, S, device)  # from-scratch Bolt for this (P, S)
    n_params = sum(p.numel() for p in model.parameters())  # count parameters actually built

    stream = build_stream(seed)                # open the shuffled corpus stream (seed-dependent order)
    dataset = make_window_dataset(stream, CONTEXT_LENGTH + PREDICTION_LENGTH, CONTEXT_LENGTH, seed)  # window generator
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)  # single-process loading (equivalent to official dataloader_num_workers=1 single stream)

    try:                                       # [CHRONOS-REF] optim: adamw_torch_fused (chronos-t5-tiny.yaml)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                                      fused=(device.type == "cuda"))  # fused kernel on CUDA = adamw_torch_fused
    except (RuntimeError, TypeError):          # older torch / unsupported device: plain AdamW (same math)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = get_scheduler(              # [CHRONOS-REF] linear decay schedule (train.py setup)
        LR_SCHEDULER_TYPE, optimizer=optimizer,          # [CHRONOS-REF] lr_scheduler_type: linear
        num_warmup_steps=round(WARMUP_RATIO * MAX_STEPS),  # [CHRONOS-REF] warmup_ratio: 0.0 -> 0 warmup steps (official)
        num_training_steps=MAX_STEPS,          # [CHRONOS-REF] total steps for the decay schedule
    )

    prov = {                                   # provenance block, written up-front so a crash still leaves a trail
        "P": P, "S": S, "seed": seed,          # the experimental coordinates
        "overlap_ratio": round((P - S) / P, 4),  # derived geometry
        "approx_num_patches": _approx_num_patches(CONTEXT_LENGTH, P, S),  # derived token-sequence length (excl. [REG])
        "base_model_id": BASE_MODEL_ID, "hf_repo": HF_REPO, "dataset_config": DATASET_CONFIG,  # [CHRONOS-REF] official sources used
        "context_length": CONTEXT_LENGTH, "prediction_length": PREDICTION_LENGTH,  # [CHRONOS-REF] Bolt defaults in effect
        "quantiles": QUANTILES, "batch_size": BATCH_SIZE,  # [CHRONOS-REF] Bolt quantiles / official batch size
        "max_steps": MAX_STEPS, "lr": LR, "weight_decay": WEIGHT_DECAY,  # budget (ours) + official LR/WD
        "grad_clip_norm": GRAD_CLIP_NORM, "lr_scheduler": LR_SCHEDULER_TYPE,  # [CHRONOS-REF] clipping + schedule
        "warmup_ratio": WARMUP_RATIO, "shuffle_buffer": SHUFFLE_BUFFER_SIZE,  # [CHRONOS-REF] official values
        "min_past": MIN_PAST, "max_missing_prop": MAX_MISSING_PROP, "drop_prob": DROP_PROB,  # [CHRONOS-REF] official data-pipeline values
        "precision": prec_name, "device": device.type,  # resolved runtime policy
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,  # which GPU (if any)
        "git_commit": _git_commit(),           # code version
        "torch": torch.__version__,            # library version
        "n_params_millions": round(n_params / 1e6, 3),  # built model size
    }
    (out_dir / "run_config.json").write_text(json.dumps({"provenance": prov}, indent=2))  # persist provenance now

    print(f"\n=== train {out_dir.name} | P={P} S={S} seed={seed} | "  # run header for the log
          f"{prec_name} | ~{prov['approx_num_patches']}+1 tokens | {n_params/1e6:.2f}M params ===")

    model.train()                              # [CHRONOS-REF] put the model in training mode
    loss_history: list[float] = []             # per-step losses (saved + plotted)
    data_iter = iter(loader)                   # manual iterator so we can refill on exhaustion
    t0 = time.time()                           # start the throughput clock

    def _next_batch():                         # fetch next batch, restarting the iterator if it ends
        nonlocal data_iter                     # rebind the outer iterator
        try:
            return next(data_iter)             # normal path
        except StopIteration:                  # stream/loader exhausted -> start a new pass
            data_iter = iter(loader)
            return next(data_iter)

    status = "done"                            # optimistic; flipped to "failed-nan" on a bad step
    pbar = tqdm(range(1, MAX_STEPS + 1),       # live progress bar over the fixed step budget...
                desc=out_dir.name, dynamic_ncols=True)  # ...labelled with this run's id, auto-width
    for step in pbar:                          # [CHRONOS-REF] fixed-step training loop (train.py trains by max_steps)
        batch = {k: v.to(device) for k, v in _next_batch().items()}  # move the batch to the device
        optimizer.zero_grad(set_to_none=True)  # [CHRONOS-REF] clear grads before the step

        out = model(                           # [CHRONOS-REF] Chronos-Bolt forward pass (official signature), fp32 compute...
            context=batch["context"], mask=batch["mask"],        # [CHRONOS-REF] NaN-encoded context + its observation mask
            target=batch["target"], target_mask=batch["target_mask"],  # [CHRONOS-REF] NaN-encoded target + its observation mask
        )
        loss = out.loss                        # [CHRONOS-REF] Bolt returns the masked quantile loss on .loss

        loss_value = float(loss.detach().cpu())  # scalarise for logging + NaN check
        if not np.isfinite(loss_value):        # guard: non-finite loss now signals genuine pathology (fp32 pipeline)
            status = "failed-nan"              # mark the run failed
            tqdm.write(f"[abort] non-finite loss at step {step} — skipping rest of this run.")  # note above the bar
            break                              # stop this run; the sweep continues with the next config

        loss.backward()                        # [CHRONOS-REF] plain fp32 backward (official regime has no AMP scaler)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)  # [CHRONOS-REF] clip gradient norm (Trainer default)
        optimizer.step()                       # [CHRONOS-REF] AdamW step
        lr_scheduler.step()                    # [CHRONOS-REF] advance the LR schedule

        loss_history.append(loss_value)        # record the step loss
        pbar.set_postfix(loss=f"{loss_value:.4f}",  # live per-step readout on the bar: current loss...
                         lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")  # ...and current learning rate
        if step % LOG_EVERY == 0:              # periodic durable log line (survives in the captured stdout)
            sps = step / (time.time() - t0)    # steps/sec so far
            tqdm.write(f"step={step}/{MAX_STEPS} loss={np.mean(loss_history[-LOG_EVERY:]):.4f} "  # smoothed loss
                       f"lr={lr_scheduler.get_last_lr()[0]:.2e} {sps:.2f} it/s")  # written above the bar
        if step % SAVE_EVERY == 0:             # periodic checkpoint
            ck = out_dir / f"checkpoint-{step}"  # checkpoint subfolder inside this model's dir
            model.save_pretrained(ck)          # [CHRONOS-REF] HF-style checkpoint save
            tqdm.write(f"  saved {ck.name}")   # note above the bar

    steps_per_sec = (len(loss_history) / (time.time() - t0)) if loss_history else 0.0  # final throughput
    result = RunResult(                        # assemble the manifest record for this run
        P=P, S=S, seed=seed, overlap_ratio=prov["overlap_ratio"],  # coordinates + geometry
        approx_num_patches=prov["approx_num_patches"],             # token-sequence length
        n_params_millions=prov["n_params_millions"], max_steps=MAX_STEPS,  # model size + planned steps
        steps_completed=len(loss_history),     # steps actually completed
        final_loss=float(loss_history[-1]) if loss_history else float("nan"),   # last loss
        mean_last_100=float(np.mean(loss_history[-100:])) if loss_history else float("nan"),  # tail-smoothed loss
        steps_per_sec=round(steps_per_sec, 3), status=status,      # throughput + status
        precision=prec_name, device=device.type,                  # runtime policy
    )

    np.save(out_dir / "loss_history.npy", np.asarray(loss_history, dtype=np.float32))  # raw loss curve data
    _plot_loss(loss_history, P, S, seed, out_dir / "loss_curve.png")  # rendered loss curve
    (out_dir / "run_config.json").write_text(  # overwrite provenance with provenance + result
        json.dumps({"provenance": prov, "result": asdict(result)}, indent=2))

    if status == "done":                       # only a fully successful run gets the final model + marker
        model.save_pretrained(out_dir)         # [CHRONOS-REF] save the final model at the run-dir root (HF format)
        (out_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))  # resume marker (skip on relaunch)
        print(f"[done] final model saved to {out_dir}")
    else:                                      # failed run: no DONE, so a relaunch will retry it
        print(f"[failed] {out_dir.name} left without DONE marker (status={status}).")
    return result                              # hand the record back to the sweep for the manifest


def _plot_loss(loss_history, P, S, seed, path):  # render the training-loss curve to PNG
    import matplotlib                          # local import (headless-safe)
    matplotlib.use("Agg")                      # non-interactive backend (no display on a server)
    import matplotlib.pyplot as plt            # plotting API
    plt.figure(figsize=(8, 4))                 # figure size
    plt.plot(loss_history)                     # loss vs step
    plt.xlabel("step"); plt.ylabel("training loss")  # axis labels
    plt.title(f"Chronos-Bolt retraining loss (P={P}, S={S}, seed={seed})")  # title identifies the model
    plt.tight_layout(); plt.savefig(path); plt.close()  # lay out, write file, free the figure


# ============================================================================ #
#  Manifest                                                                      #
# ============================================================================ #
def rebuild_manifest(root: Path) -> None:      # aggregate all finished runs into one CSV (our tooling)
    """Idempotently rebuild manifest.csv from every finished run_config.json under root."""
    import csv                                 # CSV writer
    rows = []                                  # collected result records
    for cfg in sorted(root.glob("*/run_config.json")):  # every run folder's config file
        doc = json.loads(cfg.read_text())      # load it
        if "result" in doc:                    # only finished runs have a "result" block
            rows.append(doc["result"])         # collect the record
    if not rows:                               # nothing finished yet -> no manifest
        return
    fields = list(RunResult.__dataclass_fields__)  # column order from the dataclass
    with open(root / "manifest.csv", "w", newline="") as f:  # (re)write the manifest
        w = csv.DictWriter(f, fieldnames=fields)  # header from the field list
        w.writeheader()                        # write the header row
        for r in rows:                         # one CSV row per finished run
            w.writerow({k: r.get(k) for k in fields})
    print(f"manifest.csv updated ({len(rows)} runs) -> {root / 'manifest.csv'}")


# ============================================================================ #
#  Sweep                                                                         #
# ============================================================================ #
def main():                                    # drive the full (P, S, seed) sweep
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # ensure chronos/outputs/models/ exists
    runs = [(P, S, seed) for (P, S) in PS_GRID for seed in SEEDS]  # flatten grid x seeds into a run list
    print(f"Sweep: {len(runs)} runs -> {OUTPUT_ROOT}")  # announce the plan

    for P, S, seed in runs:                    # run them sequentially
        out_dir = OUTPUT_ROOT / f"p{P}-s{S}-seed{seed}"  # this model's dedicated folder
        try:
            train_one(P, S, seed, out_dir)     # train (or skip if DONE)
        except Exception as e:                 # one run must not sink the whole sweep
            print(f"[error] {out_dir.name} raised {type(e).__name__}: {e}")
        rebuild_manifest(OUTPUT_ROOT)          # refresh the manifest after each run

    print("\nSweep finished.")                 # done


if __name__ == "__main__":                     # run the sweep when invoked as a script
    main()
