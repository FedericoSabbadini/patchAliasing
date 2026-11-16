"""Resumable pure-tone and background-conditioned MDL probing experiments.

These entry points are independent of the frozen forecasting/Bayesian experiments.
No inference or filesystem writes occur on import.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

if __package__:
    from . import checkpointing as cp, comparison_lib as comparison, model_loader
    from . import probing_protocol as protocol
else:
    import checkpointing as cp
    import comparison_lib as comparison
    import model_loader
    import probing_protocol as protocol

ProbeConfig = protocol.ProbeConfig
MODELS = comparison.MODELS
STAGES = protocol.STAGES
FS = 512
CANONICAL_LENGTH = 544
FAMILIES = ("tsmixup", "kernelsynth")


def _csv(path, frame):
    cp.atomic_write(Path(path), lambda p: frame.to_csv(p, index=False))


def _npz(path, **arrays):
    def write(p):
        with p.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
    cp.atomic_write(Path(path), write)


@contextmanager
def _run_lock(path):
    """An OS-held lock releases on process exit, including interrupted runs."""
    import os
    path = Path(path)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another process is using {path.parent}") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _prepare(root, kind, cfg, models, device):
    models = tuple(models)
    if not models or len({s.tag for s in models}) != len(models):
        raise ValueError("Model tags must be nonempty and unique")
    known = {s.tag: s for s in MODELS}
    if any(known.get(s.tag) != s for s in models):
        raise ValueError("Only the frozen fifteen geometries and official checkpoint are supported")
    if cfg.mode == "full" and set(s.tag for s in models) != set(known):
        raise ValueError("Full mode requires all 16 checkpoints; use smoke for bounded diagnostics")
    sources = [Path(__file__), Path(protocol.__file__), Path(comparison.__file__),
               Path(model_loader.__file__), Path(cp.__file__)]
    sources += sorted((Path(__file__).parents[1] / "data" / "synthetic").glob("*.py"))
    sources += [Path(__file__).with_name("probe_lib.py")]
    manifest = {
        "protocol": "frequency-grouped-prequential-v2", "kind": kind,
        "config": asdict(cfg), "device": device,
        "models": [{**asdict(s), "checkpoint": comparison.checkpoint_identity(s)} for s in models],
        "code": {p.name: cp.sha256_file(p) for p in sources},
        "packages": {p: importlib.metadata.version(p) for p in
                     ("numpy", "pandas", "scikit-learn", "torch", "chronos-forecasting",
                      "transformers", "threadpoolctl")},
        "background_recipe": {
            "canonical_length": CANONICAL_LENGTH, "first_seed": 42001,
            "normalisation": "canonical realisation: zero mean and population SD one",
            "tsmixup": {"components": [1, 4], "min_separation_hz": 10, "alpha": 1.5},
            "kernelsynth": {"J": 5, "jitter": 1e-4}, "injected_amplitude_sd": 1.5,
        },
    }
    key = cp.fingerprint(manifest)
    out = Path(root).resolve() / kind / cfg.mode / key[:16]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Manifest mismatch: refusing cache reuse")
    cp.atomic_json(path, manifest)
    manifest_stages = {"stages": list(STAGES), "encoder_reg_position": "last",
                       "decoder_reg_position": 0}
    cp.atomic_json(out / "stages.json", manifest_stages)
    design = protocol.make_design(cfg)
    _csv(out / "design.csv", design)
    return models, out, key, design


class StageExtractor(comparison.Forecaster):
    """Capture the ten probe points of Deliverable 1 v1 in one forward pass.

    The [REG] vector is the last position after each encoder block and the single
    query token (position 0) after each decoder block; `output_reg` is the input of
    the output projection and `output_head` its output, the quantile head.
    """

    def __init__(self, spec, device="cpu", batch_size=64):
        # Prefer an already cached immutable snapshot to avoid a network round trip
        # for every model. The standard loader remains the fallback for missing weights.
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        local = None if spec.official else model_loader.resolve_local_checkpoint(spec.P, spec.S)
        if local is None:
            try:
                snapshot = Path(snapshot_download(
                    repo_id=comparison.OFFICIAL_ID if spec.official else model_loader.SWEEP_REPO,
                    revision=comparison.OFFICIAL_REVISION if spec.official else model_loader.SWEEP_REVISION,
                    local_files_only=True))
                candidate = snapshot if spec.official else snapshot / f"p{spec.P}-s{spec.S}-seed42"
                if all((candidate / name).is_file() for name in ("config.json", "model.safetensors")):
                    local = candidate
            except LocalEntryNotFoundError:
                pass
        if local is None:
            super().__init__(spec, device=device, batch_size=batch_size)
        else:
            from chronos import BaseChronosPipeline
            self.spec, self.batch_size, self.device = spec, batch_size, device
            self.identity = comparison.checkpoint_identity(spec)
            self.pipe = BaseChronosPipeline.from_pretrained(str(local), device_map=device,
                                                          local_files_only=True)
            config = self.pipe.model.config.chronos_config
            if (int(config["input_patch_size"]), int(config["input_patch_stride"])) != (spec.P, spec.S):
                self.close()
                raise ValueError(f"Wrong checkpoint geometry: {spec.tag}")
            if int(config["prediction_length"]) != 64:
                self.close()
                raise ValueError("Expected the native 64-step horizon")
            self.qi = list(config["quantiles"]).index(.5)
            self.pipe.model.eval()
        if len(self.pipe.model.encoder.block) != 4 or len(self.pipe.model.decoder.block) != 4:
            self.close()
            raise ValueError("This protocol requires exactly four encoder and four decoder blocks")
        self.pipe.model.requires_grad_(False)

    def randomised(self, seed=42):
        """An untrained copy of the same architecture (random-initialisation control).

        Whatever a probe reads from this network is exposed by the architecture and
        the tokenisation alone; only the gap to the trained network speaks of training.
        The global torch generator is restored, so the draw leaves no trace elsewhere.
        """
        import copy
        import torch
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(int(seed))
            model = type(self.pipe.model)(copy.deepcopy(self.pipe.model.config))
        finally:
            torch.random.set_rng_state(state)
        model = model.eval().to(self.device)
        model.requires_grad_(False)
        clone = copy.copy(self)
        clone.pipe = copy.copy(self.pipe)
        clone.pipe.model = model
        if hasattr(clone.pipe, "inner_model"):
            clone.pipe.inner_model = model
        return clone

    def capture(self, contexts):
        import torch
        x = np.asarray(contexts, dtype=np.float32)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("Expected finite nonempty [N, context] inputs")
        captured, result, handles = {}, {s: [] for s in STAGES}, []

        def hook(stage, position=0, use_input=False):
            def save(_module, inputs, output):
                value = inputs[0] if use_input else output
                value = value[0] if isinstance(value, tuple) else value
                value = value.detach().float().cpu().numpy()
                captured[stage] = value[:, position, :] if value.ndim == 3 else value
            return save

        try:
            model = self.pipe.model
            for i, block in enumerate(model.encoder.block):
                handles.append(block.register_forward_hook(hook(f"enc_{i}", position=-1)))
            for i, block in enumerate(model.decoder.block):
                handles.append(block.register_forward_hook(hook(f"dec_{i}")))
            head = model.output_patch_embedding
            handles.append(head.register_forward_hook(hook("output_reg", use_input=True)))
            handles.append(head.register_forward_hook(hook("output_head")))
            with torch.inference_mode():
                for start in range(0, len(x), self.batch_size):
                    captured.clear()
                    self.pipe.predict(torch.as_tensor(x[start:start+self.batch_size], device=self.device),
                                      prediction_length=64)
                    if set(captured) != set(STAGES):
                        raise ValueError("A required probe stage was not captured")
                    for stage in STAGES:
                        result[stage].append(captured[stage])
        finally:
            for handle in handles:
                handle.remove()
        result = {stage: np.concatenate(blocks) for stage, blocks in result.items()}
        if any(a.ndim != 2 or len(a) != len(x) or not np.isfinite(a).all() for a in result.values()):
            raise ValueError("Incomplete or nonfinite representation capture")
        return result


def _trial_design(spec, family, bg_id, cfg):
    """Lock and control tones of one dataset (all sites for pure tones, one pair per background)."""
    if family == "pure":
        return protocol.pure_trials(spec.P, spec.S, cfg)
    trials = protocol.background_trials(spec.P, spec.S, cfg.n_backgrounds, cfg)
    return trials.loc[trials.pair.eq(bg_id)].reset_index(drop=True)


def _contexts(design, length, background=None):
    frequencies = design["frequency"].to_numpy(dtype=float)[:, None]
    phases = design["phase"].to_numpy(dtype=float)[:, None]
    tones = np.sin(2 * np.pi * frequencies * np.arange(length)[None, :] / FS + phases)
    if background is not None:
        tones = background[None, :length] + (1.5 * float(np.std(background))) * tones
    return tones.astype(np.float32)


def _backgrounds(out, family, cfg):
    path = out / "inputs" / f"{family}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            signals, seeds = z["signals"], z["seeds"]
            metadata = json.loads(str(z["metadata"]))
            expected_hash = str(z["input_hash"])
        if comparison.array_hash(signals) != expected_hash:
            raise ValueError(f"Background hash mismatch: {family}")
    else:
        recipe = comparison.RecoveryConfig(n_backgrounds=cfg.n_backgrounds)
        signals, seeds, metadata = comparison.generate_backgrounds(family, recipe)
        _npz(path, signals=signals, seeds=seeds, metadata=np.array(json.dumps(metadata)),
             input_hash=np.array(comparison.array_hash(signals)))
    if (signals.shape != (cfg.n_backgrounds, CANONICAL_LENGTH) or not np.isfinite(signals).all()
            or not np.array_equal(seeds, np.arange(42001, 42001+cfg.n_backgrounds))
            or not np.allclose(signals.std(axis=1), 1, atol=1e-6)
            or not np.allclose(signals.mean(axis=1), 0, atol=1e-6)
            or len(metadata) != cfg.n_backgrounds
            or len({comparison.array_hash(a) for a in signals}) != cfg.n_backgrounds):
        raise ValueError(f"Invalid background archive: {family}")
    return signals, seeds


def _validate_shard(frame, metadata, design, random_init=False):
    expected = {(stage, task[0]) for stage in STAGES for task in protocol.TASKS}
    if (len(frame) != len(expected) or frame.duplicated(["stage", "task"]).any()
            or set(zip(frame.stage, frame.task)) != expected):
        raise ValueError("Incomplete or duplicated stage/task shard")
    for name, value in metadata.items():
        if name not in frame or not frame[name].eq(value).all():
            raise ValueError(f"Shard identity mismatch: {name}")
    numeric = ["n", "L_bits", "L_uniform", "SV", "L_shuffled_bits", "SV_shuffled", "first_chunk_n"]
    if random_init:
        numeric += ["L_random_init_bits", "SV_random_init"]
        if set(numeric) - set(frame.columns):
            raise ValueError("Shard lacks the random-initialisation control")
        if (not np.allclose(frame.SV_random_init, 1-frame.L_random_init_bits/frame.n, atol=1e-12)
                or (frame.L_random_init_bits < frame.first_chunk_n - 1e-9).any()):
            raise ValueError("Invalid random-initialisation codelength")
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("Nonfinite probe result")
    for task, low, high, _boundary in protocol.TASKS:
        n = int(design.frequency.between(low, high).sum())
        subset = frame.loc[frame.task.eq(task)]
        if not subset.n.eq(n).all() or not subset.L_uniform.eq(n).all():
            raise ValueError("Probe sample count mismatch")
    if (not np.allclose(frame.SV, 1-frame.L_bits/frame.n, atol=1e-12)
            or not np.allclose(frame.SV_shuffled, 1-frame.L_shuffled_bits/frame.n, atol=1e-12)
            or (frame.L_bits < frame.first_chunk_n - 1e-9).any()
            or (frame.L_shuffled_bits < frame.first_chunk_n - 1e-9).any()):
        raise ValueError("Invalid prequential codelength")


def _validate_trials(frame, metadata, trials, random_init=False):
    networks = protocol.NETWORKS if random_init else protocol.NETWORKS[:1]
    tones = trials[["pair", "arm", "frequency"]].drop_duplicates()
    keys = ["network", "stage", "pair", "arm", "frequency"]
    if (len(frame) != len(tones) * len(STAGES) * len(networks) or frame.duplicated(keys).any()
            or set(frame.network) != set(networks) or set(frame.stage) != set(STAGES)):
        raise ValueError("Incomplete or duplicated lock/control shard")
    for name, value in metadata.items():
        if name not in frame or not frame[name].eq(value).all():
            raise ValueError(f"Trial shard identity mismatch: {name}")
    expected = np.sort(np.round(tones.frequency.to_numpy(float), 6))
    for _, part in frame.groupby(["network", "stage"]):
        if not np.allclose(np.sort(part.frequency.to_numpy(float)), expected, atol=1e-5):
            raise ValueError("Trial shard does not match the registered tones")
    counts = frame[["located", "level1", "level2", "level3"]].to_numpy(float)
    if (not np.isfinite(frame.bits_per_context.to_numpy(float)).all() or (counts < 0).any()
            or (counts > frame.n_contexts.to_numpy(float)[:, None]).any()
            or (frame.located > frame[["level1", "level2", "level3"]].min(axis=1)).any()
            or not frame.n_contexts.eq(trials.phase_index.nunique()).all()):
        raise ValueError("Invalid lock/control outcome")


def _dataset(out, key, spec, family, bg_id, bg_seed, background, cfg, design, extractor, device):
    """Score one dataset; returns (SV frame, lock/control frame, extractor, cached)."""
    length = protocol.context_length(spec.P, spec.S)
    trials = _trial_design(spec, family, bg_id, cfg)
    # Sweep and trial tones go through the network together and are split afterwards.
    contexts = np.concatenate([_contexts(design, length, background),
                               _contexts(trials, length, background)])
    n_sweep = len(design)
    metadata = {
        "model": spec.tag, "P": spec.P, "S": spec.S, "official": spec.official,
        "family": family, "background_id": bg_id, "background_seed": bg_seed,
        "context_length": length, "checkpoint": cp.canonical_json(comparison.checkpoint_identity(spec)),
        "run_key": key, "input_hash": comparison.array_hash(contexts),
        "background_hash": "none" if background is None else comparison.array_hash(background),
    }
    folder = out / "shards" / spec.tag / family
    path = folder / f"background_{bg_id:03d}.csv"
    trials_path = folder / f"background_{bg_id:03d}_trials.csv"
    features_path = folder / f"background_{bg_id:03d}_features.npz"
    if path.exists() and trials_path.exists():
        frame, trial_frame = pd.read_csv(path), pd.read_csv(trials_path)
        _validate_shard(frame, metadata, design, cfg.random_init)
        _validate_trials(trial_frame, metadata, trials, cfg.random_init)
        return frame, trial_frame, extractor, True
    created = False
    try:
        if features_path.exists():
            with np.load(features_path, allow_pickle=False) as z:
                if str(z["input_hash"]) != metadata["input_hash"] or str(z["run_key"]) != key:
                    raise ValueError("Feature cache mismatch")
                features = {s: z[s] for s in STAGES}
                control = {s: z[f"random_init__{s}"] for s in STAGES} if cfg.random_init else None
        else:
            if extractor is None:
                extractor = StageExtractor(spec, device=device, batch_size=cfg.batch_size)
                created = True
            features = extractor.capture(contexts)
            control = None
            if cfg.random_init:
                if getattr(extractor, "untrained", None) is None:
                    extractor.untrained = extractor.randomised(cfg.seed)
                control = extractor.untrained.capture(contexts)
            _npz(features_path, **features,
                 **({f"random_init__{s}": a for s, a in control.items()} if control else {}),
                 input_hash=np.array(metadata["input_hash"]), run_key=np.array(key))
        sweep = {s: a[:n_sweep] for s, a in features.items()}
        probe = {s: a[n_sweep:] for s, a in features.items()}
        sweep_control = {s: a[:n_sweep] for s, a in control.items()} if control else None
        frame = protocol.score_features(sweep, design, cfg, metadata, sweep_control)
        _validate_shard(frame, metadata, design, cfg.random_init)
        parts = [protocol.score_trials(sweep, design, probe, trials, cfg, metadata, "trained")]
        if control:
            parts.append(protocol.score_trials(sweep_control, design,
                                               {s: a[n_sweep:] for s, a in control.items()},
                                               trials, cfg, metadata, "random_init"))
        trial_frame = pd.concat(parts, ignore_index=True)
        _validate_trials(trial_frame, metadata, trials, cfg.random_init)
        _csv(trials_path, trial_frame)
        _csv(path, frame)
    except BaseException:
        if created:
            extractor.close()
        raise
    # A committed result shard is the resume cache; retain feature blocks only until committed.
    # This bounds storage instead of retaining tens of GB of reproducible intermediate states.
    features_path.unlink(missing_ok=True)
    return frame, trial_frame, extractor, False


def _run(root, kind, cfg, models, device):
    import torch
    if __package__:
        from . import probing_exports as exports
    else:
        import probing_exports as exports
    models, out, key, design = _prepare(root, kind, cfg, models, device)
    started, frames, trial_frames, completed = time.monotonic(), [], [], 0
    total = len(models) * (1 if kind == "pure" else 2*cfg.n_backgrounds)
    progress = {"kind": kind, "mode": cfg.mode, "run_key": key, "total_datasets": total,
                "completed_datasets": 0, "status": "running"}
    print(f"{kind.upper()} {cfg.mode}: {total} datasets; {out}", flush=True)
    old_threads = torch.get_num_threads()
    with _run_lock(out / ".run.lock"), threadpool_limits(limits=cfg.threads):
        torch.set_num_threads(cfg.threads)
        try:
            backgrounds = ({family: _backgrounds(out, family, cfg) for family in FAMILIES}
                           if kind == "complex" else {})
            for spec in models:
                extractor = None
                try:
                    items = ([("pure", -1, -1, None)] if kind == "pure" else
                             [(family, i, int(seeds[i]), signals[i])
                              for family, (signals, seeds) in backgrounds.items()
                              for i in range(cfg.n_backgrounds)])
                    for family, bg_id, bg_seed, background in items:
                        current = f"{spec.tag}/{family}/{bg_id}"
                        progress.update(current=current, completed_datasets=completed,
                                        elapsed_seconds=time.monotonic()-started)
                        cp.atomic_json(out / "progress.json", progress)
                        tick = time.monotonic()
                        frame, trial_frame, extractor, cached = _dataset(
                            out, key, spec, family, bg_id, bg_seed, background, cfg, design,
                            extractor, device)
                        frames.append(frame)
                        trial_frames.append(trial_frame)
                        completed += 1
                        progress.update(completed_datasets=completed, current=current,
                                        elapsed_seconds=time.monotonic()-started)
                        cp.atomic_json(out / "progress.json", progress)
                        print(f"[{completed}/{total}] {current}: {'cached' if cached else 'computed'} "
                              f"({time.monotonic()-tick:.1f}s)", flush=True)
                finally:
                    if extractor is not None:
                        extractor.close()
            result = pd.concat(frames, ignore_index=True)
            trial_result = pd.concat(trial_frames, ignore_index=True)
            _csv(out / "results.csv", result)
            _csv(out / "lock_control.csv", trial_result)
            artifacts = (exports.export_pure(result, out, mode=cfg.mode, trials=trial_result)
                         if kind == "pure" else
                         exports.export_complex(result, out, models, cfg, trials=trial_result))
            cp.atomic_json(out / "artifacts.json", {k: str(v) for k, v in artifacts.items()})
            progress.update(status="complete", elapsed_seconds=time.monotonic()-started)
            cp.atomic_json(out / "progress.json", progress)
            result.attrs["lock_control"] = trial_result
            return result, out
        except BaseException as exc:
            progress.update(status="error", error=f"{type(exc).__name__}: {exc}",
                            elapsed_seconds=time.monotonic()-started)
            cp.atomic_json(out / "progress.json", progress)
            raise
        finally:
            torch.set_num_threads(old_threads)


def run_pure(root, cfg=None, models=MODELS, device="cpu"):
    """Run/resume the pure-tone dataset for each selected frozen checkpoint.

    Returns the SV rows and the run directory; the lock/control rows are in
    `result.attrs["lock_control"]` and in lock_control.csv.
    """
    return _run(root, "pure", cfg or ProbeConfig(), models, device)


def run_complex(root, cfg=None, models=MODELS, device="cpu"):
    """Run/resume 100 independent background-conditioned probe datasets per family."""
    return _run(root, "complex", cfg or ProbeConfig(), models, device)
