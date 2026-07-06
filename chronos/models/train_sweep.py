"""
train_sweep.py — Chronos-Bolt Tiny from-scratch (P, S, seed) sweep driver.

Training only (no evaluation/inference). Each run retrains Chronos-Bolt Tiny FROM SCRATCH
(random weights) on the official Chronos pre-training corpus
(autogluon/chronos_datasets / training_corpus_tsmixup_10m, streamed), varying ONLY the
patch geometry input_patch_size (P) and input_patch_stride (S). Everything else is fixed so
any downstream difference in structural-aliasing probes is attributable to P/S.

Just run the file:

    python train_sweep.py

Edit the CONFIG block below (PS_GRID, SEEDS, hyperparameters) to change the experiment.
The sweep is resumable: a run that already wrote a DONE marker is skipped.

Outputs: one folder per model at chronos/outputs/models/p{P}-s{S}-seed{seed}/, holding
EVERY artifact tied to that model (run_config.json, loss_history.npy, loss_curve.png,
checkpoint-{step}/, final model, DONE). An aggregate chronos/outputs/models/manifest.csv is
rebuilt from every finished run.

---------------------------------------------------------------------------------------
Provenance markers on code lines:
  "# [CHRONOS-REF] ..."  -> the line's approach, API, or default value is taken from the official
               Chronos reference (Apache-2.0): github.com/amazon-science/chronos-forecasting
               (scripts/training/train.py) and the chronos.chronos_bolt library API.
               Adopted / adapted, NOT copied verbatim.
  plain "# ..." -> scaffolding written specifically for this project (sweep loop, seeding,
               resume, precision/NaN robustness, provenance/manifest).
---------------------------------------------------------------------------------------
"""
from __future__ import annotations          # allow modern type hints on older runtimes

import json                                 # write/read per-run run_config.json + manifest
import subprocess                           # shell out to `git` to record the commit hash
import time                                 # wall-clock timing + DONE timestamp
from dataclasses import dataclass, asdict   # typed per-run result record -> dict for JSON/CSV
from pathlib import Path                     # filesystem paths (output dirs, markers)

import numpy as np                          # [CHRONOS-REF] array math for windows/loss (train.py also uses numpy)


# ============================================================================ #
#  CONFIG — the only things to edit                                             #
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

CONTEXT_LENGTH = 2048                        # [CHRONOS-REF] Chronos-Bolt default context length
PREDICTION_LENGTH = 64                       # [CHRONOS-REF] Chronos-Bolt default forecast horizon
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # [CHRONOS-REF] the 9 quantiles Bolt is trained on

HF_REPO = "autogluon/chronos_datasets"       # [CHRONOS-REF] official Chronos datasets repository on HuggingFace
DATASET_CONFIG = "training_corpus_tsmixup_10m"  # [CHRONOS-REF] official TSMixup pre-training corpus (10M series)

BATCH_SIZE = 32                              # micro-batch; lower globally (same for every config) if the heavy config OOMs
MAX_STEPS = 10_000                           # optimizer steps, identical across runs (set low for a quick test)
LR = 1e-4                                     # [CHRONOS-REF] learning rate in the range used by the Chronos training configs
WEIGHT_DECAY = 1e-2                           # [CHRONOS-REF] AdamW weight decay, as in the official training setup
GRAD_CLIP_NORM = 1.0                          # [CHRONOS-REF] gradient-norm clipping, as in train.py
LR_SCHEDULER_TYPE = "linear"                  # [CHRONOS-REF] linear LR decay, as in train.py
WARMUP_RATIO = 0.05                           # [CHRONOS-REF] LR warmup fraction, as in train.py (stabilises early-step training)
SHUFFLE_BUFFER_SIZE = 1000                    # [CHRONOS-REF] stream shuffle buffer (train.py shuffles the training stream too)
LOG_EVERY = 50                               # console logging cadence (our diagnostic)
SAVE_EVERY = 1000                            # checkpoint cadence in steps (our diagnostic)

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
    """Best-effort patch count for provenance (Chronos-Bolt left-pads to a multiple of P)."""  # [CHRONOS-REF] padding rule from Bolt
    l_pad = -(-context_length // P) * P      # [CHRONOS-REF] ceil(context_length / P) * P  (pad up to a multiple of P)
    return (l_pad - P) // S + 1              # [CHRONOS-REF] number of length-P windows at stride S over the padded context


# ============================================================================ #
#  Dataset                                                                       #
# ============================================================================ #
def build_stream(seed: int):                 # open the HF corpus as a shuffled streaming iterator
    from datasets import load_dataset        # [CHRONOS-REF] HuggingFace datasets (the corpus lives there)
    stream = load_dataset(                    # [CHRONOS-REF] open the official corpus...
        HF_REPO, DATASET_CONFIG,             # [CHRONOS-REF] ...repo + config selected above
        split="train", streaming=True,       # [CHRONOS-REF] stream instead of downloading the whole 10M-series corpus
    )
    return stream.shuffle(                    # [CHRONOS-REF] shuffle the stream (train.py shuffles too)...
        seed=seed, buffer_size=SHUFFLE_BUFFER_SIZE,  # ...reproducibly, with a bounded buffer
    )


def make_window_dataset(hf_stream, total_length: int, context_length: int, seed: int):
    import torch                              # tensors are built here, per worker
    from torch.utils.data import IterableDataset  # streaming source has no length -> IterableDataset

    class ChronosStreamingWindowDataset(IterableDataset):  # yields ready-to-forward training windows
        """Yield (context, mask, target, target_mask) windows from the HF TSMixup stream.

        Series shorter than context+prediction are skipped. Unobserved (NaN) samples are
        zero-filled but flagged in the mask, and windows with a fully-unobserved context
        or target are skipped (their loss would be undefined).
        """

        def __iter__(self):                  # generator called once per DataLoader worker
            rng = np.random.default_rng(seed)  # per-iterator RNG so window sampling is reproducible
            while True:                       # loop the corpus forever (only matters if corpus < steps*batch)
                for row in hf_stream:         # [CHRONOS-REF] iterate raw series from the official stream
                    if "target" not in row:   # schema guard: the value column must be present
                        raise KeyError(       # fail loudly with a helpful message if the schema changed
                            f"Row has no 'target' field (keys={list(row)}). "
                            f"Check the schema of {HF_REPO}/{DATASET_CONFIG}."
                        )
                    values = np.asarray(row["target"], dtype=np.float32)  # [CHRONOS-REF] the full series values
                    if values.shape[0] < total_length:  # [CHRONOS-REF] drop series too short for context+prediction (train.py filters short series)
                        continue              # skip and try the next series
                    start = int(rng.integers(0, values.shape[0] - total_length + 1))  # uniform random window start (our simplification; train.py uses length-weighted sampling, deliberately not adopted)
                    window = values[start:start + total_length]  # slice one context+target window

                    ctx_raw = window[:context_length]   # the context part
                    tgt_raw = window[context_length:]   # the target (forecast) part
                    ctx_mask = ~np.isnan(ctx_raw)        # True where a context value is observed
                    tgt_mask = ~np.isnan(tgt_raw)        # True where a target value is observed
                    if not ctx_mask.any() or not tgt_mask.any():  # nothing to condition on / predict
                        continue              # skip degenerate windows

                    yield {                   # one training example, keyed for the Bolt forward()
                        "context": torch.from_numpy(np.nan_to_num(ctx_raw, nan=0.0)),  # zero-fill NaNs, keep real mask (our robustness choice)
                        "mask": torch.from_numpy(ctx_mask),          # [CHRONOS-REF] observation mask for the context (Bolt forward arg)
                        "target": torch.from_numpy(np.nan_to_num(tgt_raw, nan=0.0)),   # zero-fill target NaNs
                        "target_mask": torch.from_numpy(tgt_mask),   # [CHRONOS-REF] observation mask for the target (Bolt forward arg)
                    }

    return ChronosStreamingWindowDataset()   # instance the DataLoader will wrap


# ============================================================================ #
#  Model                                                                         #
# ============================================================================ #
def build_model(P: int, S: int, device):     # construct a from-scratch Bolt model for this (P, S)
    from transformers import AutoConfig       # [CHRONOS-REF] load the architecture config from HuggingFace
    from chronos.chronos_bolt import ChronosBoltModelForForecasting  # [CHRONOS-REF] the real Chronos-Bolt model class

    config = AutoConfig.from_pretrained(BASE_MODEL_ID)  # [CHRONOS-REF] pull Bolt-tiny's architectural config (weights NOT fetched)
    config.chronos_config["context_length"] = CONTEXT_LENGTH        # [CHRONOS-REF] set context length in the Bolt sub-config
    config.chronos_config["prediction_length"] = PREDICTION_LENGTH  # [CHRONOS-REF] set forecast horizon
    config.chronos_config["input_patch_size"] = P                   # [CHRONOS-REF] THE experimental knob: patch size P
    config.chronos_config["input_patch_stride"] = S                 # [CHRONOS-REF] THE experimental knob: patch stride S
    config.chronos_config["quantiles"] = QUANTILES                  # [CHRONOS-REF] set the quantile heads

    model = ChronosBoltModelForForecasting(config)  # [CHRONOS-REF] build with RANDOM weights -> from-scratch (changing P changes embedding in_features = P*2, so stock weights can't load)
    return model.to(device), config.chronos_config  # move to GPU/CPU; return the resolved sub-config for provenance


def _precision(device):                       # choose autocast dtype + whether a GradScaler is needed
    """Pick (autocast_dtype, use_grad_scaler). bf16 when available; fp16 fallback; fp32 CPU."""
    import torch                               # local import: torch only needed inside the run
    if device.type != "cuda":                  # no GPU -> plain fp32, no autocast, no scaler
        return None, False
    if torch.cuda.is_bf16_supported():         # prefer bf16: a T5 backbone from scratch is fp16-unstable
        return torch.bfloat16, False           # bf16 has fp32-range exponent -> no loss scaler needed
    return torch.float16, True                 # fallback: fp16 autocast + GradScaler to avoid underflow


# ============================================================================ #
#  One run                                                                       #
# ============================================================================ #
@dataclass
class RunResult:                              # one row of the manifest: the outcome of a single run
    P: int                                    # patch size
    S: int                                    # stride
    seed: int                                 # seed
    overlap_ratio: float                      # (P - S) / P
    approx_num_patches: int                   # patches per context window (provenance)
    n_params_millions: float                  # model size actually built
    max_steps: int                            # planned optimizer steps
    steps_completed: int                      # steps actually run (< max_steps if aborted)
    final_loss: float                         # last step's loss
    mean_last_100: float                      # smoothed tail loss
    steps_per_sec: float                      # throughput (sanity-check time estimates)
    status: str                               # "done" | "failed-nan"
    precision: str                            # fp32 | float16 | bfloat16
    device: str                               # cuda | cpu


def train_one(P: int, S: int, seed: int, out_dir: Path) -> RunResult:
    import torch                               # heavy imports kept inside the run (fast --help, clean sweep startup)
    from torch.utils.data import DataLoader    # batches windows from the IterableDataset
    from transformers import get_scheduler     # [CHRONOS-REF] HF LR scheduler factory (warmup+linear, as in train.py)

    if out_dir.joinpath("DONE").exists():      # resumability: this model is already fully trained
        print(f"[skip] {out_dir.name} already DONE")
        prev = json.loads(out_dir.joinpath("run_config.json").read_text())  # reload its recorded result
        return RunResult(**{k: prev["result"][k] for k in RunResult.__dataclass_fields__})  # reconstruct the record

    out_dir.mkdir(parents=True, exist_ok=True)  # create this model's dedicated folder

    import random                              # stdlib RNG also needs seeding for full reproducibility
    random.seed(seed)                          # [CHRONOS-REF] seed python RNG (train.py seeds all RNGs too)
    np.random.seed(seed)                       # [CHRONOS-REF] seed numpy RNG
    torch.manual_seed(seed)                    # [CHRONOS-REF] seed torch CPU RNG
    torch.cuda.manual_seed_all(seed)           # [CHRONOS-REF] seed torch CUDA RNGs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pick GPU if present
    amp_dtype, use_scaler = _precision(device)  # resolve mixed-precision policy for this device
    prec_name = "fp32" if amp_dtype is None else str(amp_dtype).replace("torch.", "")  # human-readable label

    model, chronos_config = build_model(P, S, device)  # from-scratch Bolt for this (P, S)
    n_params = sum(p.numel() for p in model.parameters())  # count parameters actually built

    stream = build_stream(seed)                # open the shuffled corpus stream (seed-dependent order)
    dataset = make_window_dataset(stream, CONTEXT_LENGTH + PREDICTION_LENGTH, CONTEXT_LENGTH, seed)  # window generator
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)  # single-process loading (streaming source)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # [CHRONOS-REF] AdamW, as in train.py
    lr_scheduler = get_scheduler(              # [CHRONOS-REF] build the warmup+linear schedule (train.py setup)
        LR_SCHEDULER_TYPE, optimizer=optimizer,          # [CHRONOS-REF] linear decay
        num_warmup_steps=round(WARMUP_RATIO * MAX_STEPS),  # [CHRONOS-REF] warmup steps = ratio * total
        num_training_steps=MAX_STEPS,          # [CHRONOS-REF] total steps for the decay schedule
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)  # fp16 loss scaler (disabled for bf16/fp32)

    prov = {                                   # provenance block, written up-front so a crash still leaves a trail
        "P": P, "S": S, "seed": seed,          # the experimental coordinates
        "overlap_ratio": round((P - S) / P, 4),  # derived geometry
        "approx_num_patches": _approx_num_patches(CONTEXT_LENGTH, P, S),  # derived token-sequence length
        "base_model_id": BASE_MODEL_ID, "hf_repo": HF_REPO, "dataset_config": DATASET_CONFIG,  # [CHRONOS-REF] official sources used
        "context_length": CONTEXT_LENGTH, "prediction_length": PREDICTION_LENGTH,  # [CHRONOS-REF] Bolt defaults in effect
        "quantiles": QUANTILES, "batch_size": BATCH_SIZE,  # [CHRONOS-REF] quantiles / our batch size
        "max_steps": MAX_STEPS, "lr": LR, "weight_decay": WEIGHT_DECAY,  # optimisation budget/hyperparams
        "grad_clip_norm": GRAD_CLIP_NORM, "lr_scheduler": LR_SCHEDULER_TYPE,  # [CHRONOS-REF] clipping + schedule (train.py)
        "warmup_ratio": WARMUP_RATIO, "shuffle_buffer": SHUFFLE_BUFFER_SIZE,  # [CHRONOS-REF] warmup + shuffle (train.py)
        "precision": prec_name, "device": device.type,  # resolved runtime policy
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,  # which GPU (if any)
        "git_commit": _git_commit(),           # code version
        "torch": torch.__version__,            # library version
        "n_params_millions": round(n_params / 1e6, 3),  # built model size
    }
    (out_dir / "run_config.json").write_text(json.dumps({"provenance": prov}, indent=2))  # persist provenance now

    print(f"\n=== train {out_dir.name} | P={P} S={S} seed={seed} | "  # run header for the log
          f"{prec_name} | ~{prov['approx_num_patches']} patches | {n_params/1e6:.2f}M params ===")

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
    for step in range(1, MAX_STEPS + 1):       # [CHRONOS-REF] fixed-step training loop (train.py trains by max_steps)
        batch = {k: v.to(device) for k, v in _next_batch().items()}  # move the batch to the device
        optimizer.zero_grad(set_to_none=True)  # [CHRONOS-REF] clear grads before the step

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):  # mixed precision (our policy)
            out = model(                       # [CHRONOS-REF] Chronos-Bolt forward pass (official signature)...
                context=batch["context"], mask=batch["mask"],        # [CHRONOS-REF] context + its observation mask
                target=batch["target"], target_mask=batch["target_mask"],  # [CHRONOS-REF] target + its observation mask
            )
            loss = out.loss                    # [CHRONOS-REF] Bolt returns the quantile loss on .loss

        loss_value = float(loss.detach().cpu())  # scalarise for logging + NaN check
        if not np.isfinite(loss_value):        # guard: non-finite loss (fp16 overflow / bad data)
            status = "failed-nan"              # mark the run failed
            print(f"[abort] non-finite loss at step {step} — skipping rest of this run.")
            break                              # stop this run; the sweep continues with the next config

        scaler.scale(loss).backward()          # [CHRONOS-REF] backward (scaled for fp16; identity for bf16/fp32)
        scaler.unscale_(optimizer)             # unscale before clipping so the norm is in real units
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)  # [CHRONOS-REF] clip gradient norm (train.py)
        scaler.step(optimizer)                 # [CHRONOS-REF] optimizer step (scaler-aware)
        scaler.update()                        # update the fp16 scale factor
        lr_scheduler.step()                    # [CHRONOS-REF] advance the LR schedule

        loss_history.append(loss_value)        # record the step loss
        if step % LOG_EVERY == 0:              # periodic console log
            sps = step / (time.time() - t0)    # steps/sec so far
            print(f"step={step}/{MAX_STEPS} loss={np.mean(loss_history[-LOG_EVERY:]):.4f} "  # smoothed loss
                  f"lr={lr_scheduler.get_last_lr()[0]:.2e} {sps:.2f} it/s")
        if step % SAVE_EVERY == 0:             # periodic checkpoint
            ck = out_dir / f"checkpoint-{step}"  # checkpoint subfolder inside this model's dir
            model.save_pretrained(ck)          # [CHRONOS-REF] HF-style checkpoint save
            print(f"  saved {ck.name}")

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
