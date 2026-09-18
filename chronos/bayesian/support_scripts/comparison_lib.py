"""Reproducible 15-retrained + official comparisons, independent of figure settings.

No inference or filesystem writes occur on import. All point forecasts are q50.
The frozen registry deliberately ignores PATCHALIASING_POPULATION/MODELS.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import checkpointing as cp
import model_loader as ml

FS, CTX, HORIZON = 512, 480, 64
BAND = (2., 250.)
OFFICIAL_ID = "amazon/chronos-bolt-tiny"
OFFICIAL_REVISION = "a0e552de83495b5c28c14c71c374f3e33280b340"
FROZEN15 = ((8, 8), (16, 8), (16, 12), (16, 16),
            (24, 8), (24, 12), (24, 16), (24, 20), (24, 24),
            (32, 8), (32, 12), (32, 16), (32, 20), (32, 24), (32, 32))


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    P: int
    S: int
    official: bool = False

    @property
    def label(self):
        return "official p16-s16" if self.official else f"retrained p{self.P}-s{self.S}"


MODELS = (ModelSpec("official-p16-s16", 16, 16, True),) + tuple(
    ModelSpec(f"retrained-p{p}-s{s}", p, s) for p, s in FROZEN15)


@dataclass(frozen=True)
class RecoveryConfig:
    mode: str = "full"
    n_backgrounds: int = 100
    seed: int = 42001
    amplitude: float = 1.5
    k_components: int = 4
    k_random: bool = True
    alpha: float = 1.5
    min_sep_draw: float = 10.
    n_peaks: int = 8
    spectral_cut: float = .35
    tolerance_hz: float = 1.
    control_guard_hz: float = 2.000001
    batch_size: int = 64


@dataclass(frozen=True)
class SolarConfig:
    mode: str = "full"
    n_origins: int = 50_000
    seed: int = 42002
    context_length: int = 2048
    horizon: int = 64
    min_observed: int = 60
    channel: str = "PV_Power"
    spike_max: float = 1000.
    nan_policy: str = "zero_interp_short"
    short_gap: int = 15
    batch_size: int = 64
    block_size: int = 1024


def array_hash(array):
    import hashlib
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def save_csv(path, frame):
    cp.atomic_write(Path(path), lambda p: frame.to_csv(p, index=False))


def save_npz(path, **arrays):
    def write(p):
        with p.open("wb") as f:
            np.savez_compressed(f, **arrays)
    cp.atomic_write(Path(path), write)


def checkpoint_identity(spec):
    if spec.official:
        return {"repo": OFFICIAL_ID, "revision": OFFICIAL_REVISION,
                "geometry": [16, 16]}
    return ml.checkpoint_identity(spec.P, spec.S)


class Forecaster:
    """One checkpoint, exact median, bounded batches, explicit lifecycle."""
    def __init__(self, spec, device=None, batch_size=64):
        import torch
        from chronos import BaseChronosPipeline
        self.spec, self.batch_size = spec, batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.identity = checkpoint_identity(spec)
        if spec.official:
            args = dict(pretrained_model_name_or_path=OFFICIAL_ID, revision=OFFICIAL_REVISION)
        else:
            local = ml.resolve_local_checkpoint(spec.P, spec.S)
            args = (dict(pretrained_model_name_or_path=str(local)) if local else
                    dict(pretrained_model_name_or_path=ml.SWEEP_REPO,
                         subfolder=f"p{spec.P}-s{spec.S}-seed42", revision=ml.SWEEP_REVISION))
        self.pipe = BaseChronosPipeline.from_pretrained(device_map=self.device, **args)
        cfg = self.pipe.model.config.chronos_config
        if (int(cfg["input_patch_size"]), int(cfg["input_patch_stride"])) != (spec.P, spec.S):
            raise ValueError(f"Wrong checkpoint geometry: {spec.tag}")
        if int(cfg["prediction_length"]) != HORIZON:
            raise ValueError("Expected the native 64-step horizon")
        self.qi = list(cfg["quantiles"]).index(.5)
        self.pipe.model.eval()

    def forecast(self, contexts):
        import torch
        x = np.asarray(contexts, dtype=np.float32)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("Expected nonempty [batch, context] inputs")
        result = []
        with torch.inference_mode():
            for i in range(0, len(x), self.batch_size):
                tensor = torch.tensor(x[i:i+self.batch_size], device=self.device)
                y = self.pipe.predict(tensor, prediction_length=HORIZON)
                result.append(y[:, self.qi, :].float().cpu().numpy())
        result = np.concatenate(result)
        if result.shape != (len(x), HORIZON) or not np.isfinite(result).all():
            raise ValueError(f"Incomplete/nonfinite predictions from {self.spec.tag}")
        return result

    def close(self):
        import gc
        import torch
        self.pipe = None
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


def make_run(root, kind, config, data_identity, models):
    import importlib.metadata
    manifest = {"kind": kind, "config": asdict(config), "data": data_identity,
                "models": [{**asdict(s), "checkpoint": checkpoint_identity(s)} for s in models],
                "code": {p.name: cp.sha256_file(p) for p in
                         (Path(__file__), Path(ml.__file__), Path(__file__).with_name("probe_lib.py"))},
                "packages": {p: importlib.metadata.version(p) for p in
                             ("numpy", "pandas", "torch", "chronos-forecasting", "transformers")}}
    key = cp.fingerprint(manifest)
    out = Path(root) / kind / config.mode / key[:16]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Manifest mismatch; refusing to reuse results")
    cp.atomic_json(path, manifest)
    return out, key


def cached_forecast(path, contexts, predictor, key):
    """Atomic block cache bound to exact inputs and full run identity."""
    path = Path(path)
    input_hash = array_hash(contexts)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            y = z["prediction"]
            valid = (str(z["run_key"]) == key and str(z["input_hash"]) == input_hash
                     and y.shape == (len(contexts), HORIZON) and np.isfinite(y).all())
            if not valid:
                raise ValueError(f"Invalid cached block: {path}")
            return y
    y = predictor(contexts)
    if y.shape != (len(contexts), HORIZON) or not np.isfinite(y).all():
        raise ValueError("Incomplete block; nothing was saved")
    save_npz(path, prediction=y, run_key=np.array(key), input_hash=np.array(input_hash))
    return y


def lock_frequencies(p, s):
    return np.array(sorted(set(round(k*FS/p, 6) for k in range(1, p//2+1)
                               if BAND[0] <= k*FS/p <= BAND[1]) |
                           set(round(k*FS/s, 6) for k in range(1, s//2+1)
                               if BAND[0] <= k*FS/s <= BAND[1])))


def control_delta(p, s, frequency, guard=2.000001):
    locks = lock_frequencies(p, s)
    for d in np.arange(.25*FS/s, 1.-1e-9, -.05):
        controls = np.array([frequency-d, frequency+d])
        if (np.all((controls >= BAND[0]) & (controls <= BAND[1])) and
                np.min(abs(controls[:, None]-locks)) >= guard):
            return float(d)
    raise ValueError(f"No disjoint controls at p{p}-s{s}, f={frequency}")


def recovery_pool():
    # Explicit population prevents unrelated environment settings changing the pool.
    import probe_lib as pl
    return np.asarray(pl.tsmixup_pool(models=list(FROZEN15)), dtype=float)


def generate_backgrounds(family, cfg):
    """Original decomposition recipes: K<=4 Light TSMixup or J=5 KernelSynth."""
    import probe_lib as pl  # wires the generator imports, no model is loaded
    seeds = np.arange(cfg.seed, cfg.seed+cfg.n_backgrounds, dtype=np.int64)
    pool = recovery_pool()
    signals, metadata = [], []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        if family == "tsmixup":
            k = int(rng.integers(1, cfg.k_components+1)) if cfg.k_random else cfg.k_components
            frequencies = []
            for _ in range(10000):
                f = float(rng.choice(pool))
                if all(abs(f-g) >= cfg.min_sep_draw for g in frequencies):
                    frequencies.append(f)
                if len(frequencies) == k:
                    break
            if len(frequencies) != k:
                raise ValueError("Cannot draw separated background components")
            components = np.array([pl.make_tone(f, 0., CTX+HORIZON, 1.) for f in frequencies])
            components /= np.mean(abs(components), axis=1)[:, None]
            weights = rng.dirichlet(np.full(k, cfg.alpha))
            x = weights @ components
            meta = {"frequencies": frequencies, "weights": weights.tolist()}
        elif family == "kernelsynth":
            import signalGenerator as sg
            params = {"J": 5, "l_syn": CTX+HORIZON, "fs": FS, "jitter": 1e-4, "P": 16}
            scratch = Path(__file__).parents[1] / "_gen_tmp"
            scratch.mkdir(exist_ok=True)
            gen = sg.runKernelSynth(params, int(seed), scratch)
            x = np.asarray(gen.generate(), float).ravel()
            meta = {"kernels": list(getattr(gen, "last_kernels", [])), "params": params}
        else:
            raise ValueError(family)
        if len(x) != CTX+HORIZON or not np.isfinite(x).all() or np.std(x) <= 1e-8:
            raise ValueError(f"Invalid {family} background seed={seed}")
        signals.append(((x-x.mean())/x.std()).astype(np.float32))
        metadata.append({"seed": int(seed), **meta})
    return np.array(signals), seeds, metadata


def injection_trials(backgrounds, seeds, p, s, cfg):
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, p, s, 131]))
    n = len(backgrounds)
    locks = lock_frequencies(p, s)
    schedule = np.resize(rng.permutation(locks), n)
    rng.shuffle(schedule)
    signs = np.resize(np.array([-1., 1.]), n)
    rng.shuffle(signs)
    phases = rng.uniform(0, 2*np.pi, n)
    t = np.arange(CTX+HORIZON)/FS
    inputs, rows = [], []
    for i, (bg, fk, sign, phase) in enumerate(zip(backgrounds, schedule, signs, phases)):
        fc = fk + sign*control_delta(p, s, fk, cfg.control_guard_hz)
        if min(abs(fc-locks)) <= 2:
            raise AssertionError("Control overlaps a lock tolerance interval")
        amplitude = cfg.amplitude * float(np.std(bg, dtype=np.float64))
        for arm, frequency in (("lock", fk), ("control", fc)):
            # Use the same sinusoid convention as the original notebook.
            import probe_lib as pl
            signal = bg + pl.make_tone(float(frequency), float(phase), len(t), amplitude)
            inputs.append(signal.astype(np.float32))
            rows.append(dict(background=i, background_seed=int(seeds[i]), P=p, S=s,
                             arm=arm, frequency_hz=float(frequency), lock_hz=float(fk),
                             amplitude=amplitude, phase=float(phase)))
    return np.array(inputs), pd.DataFrame(rows)


def select_peaks(signal, n_peaks=8, cut=.35, min_sep=8., min_rel=.05):
    """Same local maxima / Hann magnitude membership rule as decomposition."""
    y = np.asarray(signal, float)
    if y.ndim != 1 or len(y) < 3 or not np.isfinite(y).all():
        raise ValueError("Nonfinite or invalid output spectrum")
    y = y-y.mean()
    fr = np.fft.rfftfreq(max(512, len(y)), 1/FS)
    w = np.hanning(len(y))
    mag = abs(np.fft.rfft(y*w, n=max(512, len(y))))*2/w.sum()
    local = np.zeros(len(mag), bool)
    local[1:-1] = (mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])
    inside = (fr >= BAND[0]) & (fr <= BAND[1])
    peak = float(mag[inside].max())
    candidates = []
    if peak > 1e-12:
        indices = np.flatnonzero(local & inside & (mag >= min_rel*peak))
        for j in indices[np.argsort(mag[indices], kind="stable")[::-1]]:
            if all(abs(fr[j]-fr[q]) >= max(min_sep, FS/len(y)) for q in candidates):
                candidates.append(int(j))
            if len(candidates) == n_peaks:
                break
    strongest = max((mag[j] for j in candidates), default=0.)
    floor = cut*strongest
    kept = sorted(float(fr[j]) for j in candidates if mag[j] >= floor)
    return dict(frequencies=kept, candidates=sorted(float(fr[j]) for j in candidates),
                floor=float(floor), strongest=float(strongest),
                magnitude_by_frequency={float(fr[j]): float(mag[j]) for j in candidates},
                rule=f"{100*cut:g}% of strongest prediction peak")


def recovered(peaks, injected, tolerance=1.):
    return any(abs(float(f)-float(injected)) <= tolerance for f in peaks)


def run_recovery(root, cfg=RecoveryConfig(), models=MODELS, device=None):
    if cfg.mode == "full" and (cfg.n_backgrounds != 100 or tuple(models) != MODELS):
        raise ValueError("Full recovery requires 100 backgrounds and all 16 checkpoints")
    results = {}
    for family in ("tsmixup", "kernelsynth"):
        # Dataset cache is independent of checkpoints and figure controls.
        recipe = {"family": family, "config": asdict(cfg),
                  "code": cp.sha256_file(Path(__file__)), "pool": array_hash(recovery_pool()),
                  "generator": cp.sha256_file(Path(__file__).parents[2]/"data/synthetic/signalGenerator.py")}
        pool_dir = Path(root)/"backgrounds"/cfg.mode/cp.fingerprint(recipe)[:16]
        pool_dir.mkdir(parents=True, exist_ok=True)
        if (pool_dir/"signals.npz").exists() and (pool_dir/"draws.json").exists():
            with np.load(pool_dir/"signals.npz") as z:
                backgrounds, seeds = z["backgrounds"], z["seeds"]
        else:
            backgrounds, seeds, meta = generate_backgrounds(family, cfg)
            cp.atomic_json(pool_dir/"draws.json", meta)
            save_npz(pool_dir/"signals.npz", backgrounds=backgrounds, seeds=seeds)
        out, key = make_run(root, "recovery_"+family, cfg,
                            {"recipe": recipe, "backgrounds": array_hash(backgrounds)}, models)
        rows = []
        for spec in models:
            inputs, trials = injection_trials(backgrounds, seeds, spec.P, spec.S, cfg)
            forecast = None
            def predict(x):
                nonlocal forecast
                if forecast is None:
                    forecast = Forecaster(spec, device, cfg.batch_size)
                return forecast.forecast(x)
            try:
                predictions = cached_forecast(out/spec.tag/"predictions.npz", inputs[:, :CTX], predict, key)
            finally:
                if forecast is not None:
                    forecast.close()
            peaks = [select_peaks(y, cfg.n_peaks, cfg.spectral_cut)["frequencies"] for y in predictions]
            trials["model"] = spec.label
            trials["selected_peaks_hz"] = [json.dumps(p) for p in peaks]
            trials["recovered"] = [recovered(p, f, cfg.tolerance_hz) for p, f in zip(peaks, trials.frequency_hz)]
            save_csv(out/spec.tag/"trials.csv", trials)
            row = dict(model=spec.label, P=spec.P, S=spec.S)
            for arm in ("lock", "control"):
                part = trials[trials.arm == arm]
                if len(part) != cfg.n_backgrounds:
                    raise AssertionError("Wrong trial denominator")
                row[arm+"_recovered"] = int(part.recovered.sum())
                row[arm+"_total"] = len(part)
                row[arm+"_percent"] = 100*part.recovered.mean()
            rows.append(row)
            print(f"{family} {spec.label}: {row['lock_percent']:.1f}% lock, "
                  f"{row['control_percent']:.1f}% control", flush=True)
            save_csv(out/"summary.partial.csv", pd.DataFrame(rows))
        table = pd.DataFrame(rows)
        save_csv(out/"summary.csv", table)
        cp.atomic_json(out/"complete.json", {"rows": len(table), "mode": cfg.mode, "run_key": key})
        results[family] = (table, out)
    return results


def solar_elevation(timestamps, lat=45.5028, lon=9.1560, tz=1.):
    """Same fixed-CET solar-elevation convention as the original notebook."""
    t = pd.DatetimeIndex(timestamps)-pd.Timedelta(hours=tz)
    n = t.dayofyear.to_numpy(float)
    hr = t.hour.to_numpy(float)+t.minute.to_numpy(float)/60+t.second.to_numpy(float)/3600
    g = 2*np.pi/365*(n-1+(hr-12)/24)
    eq = 229.18*(.000075+.001868*np.cos(g)-.032077*np.sin(g)
                  -.014615*np.cos(2*g)-.040849*np.sin(2*g))
    dec = (.006918-.399912*np.cos(g)+.070257*np.sin(g)-.006758*np.cos(2*g)
           +.000907*np.sin(2*g)-.002697*np.cos(3*g)+.00148*np.sin(3*g))
    ha = np.deg2rad(((hr*60+eq+4*lon) % 1440)/4-180)
    la = np.deg2rad(lat)
    return np.rad2deg(np.arcsin(np.clip(np.sin(la)*np.sin(dec)+np.cos(la)*np.cos(dec)*np.cos(ha), -1, 1)))


def fill_context(values, night, policy="zero_interp_short", short_gap=15):
    """Only the supplied history is visible, including both interpolation endpoints."""
    v = np.asarray(values, float).copy()
    missing = ~np.isfinite(v)
    if policy == "mask":
        return v
    if policy == "zero_night":
        v[missing & night] = 0.
        return v
    if policy not in ("zero", "zero_interp_short"):
        raise ValueError(policy)
    if policy == "zero_interp_short":
        bounds = np.diff(np.r_[False, missing, False].astype(np.int8))
        for start, end in zip(np.flatnonzero(bounds == 1), np.flatnonzero(bounds == -1)):
            if (end-start <= short_gap and start > 0 and end < len(v)
                    and not np.any(night[start:end])):
                v[start:end] = np.linspace(v[start-1], v[end], end-start+2)[1:-1]
    v[~np.isfinite(v)] = 0.
    return v


def load_solar(path, cfg):
    if cfg.channel != "PV_Power":
        raise ValueError("Nighttime zero targets are defined only for PV_Power")
    raw = pd.read_csv(path, sep=";")
    timestamps = pd.to_datetime(raw.Time, dayfirst=True)
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("Solar timestamps must be ordered and unique")
    values = pd.to_numeric(raw[cfg.channel], errors="coerce").to_numpy(dtype=float, copy=True)
    values[(values == -999999.) | (values > cfg.spike_max)] = np.nan
    values[~np.isfinite(values)] = np.nan
    night = solar_elevation(timestamps) <= 0
    targets = values.copy()
    targets[night & ~np.isfinite(values)] = 0.
    return timestamps, values, night, targets


def eligible_origins(timestamps, values, targets, cfg):
    n = len(values)
    cuts = np.arange(cfg.context_length, n-cfg.horizon+1, dtype=np.int64)
    observed = np.r_[0, np.cumsum(np.isfinite(values))]
    valid = np.r_[0, np.cumsum(np.isfinite(targets))]
    steps = pd.DatetimeIndex(timestamps).as_unit("ns").asi8
    breaks = np.r_[0, np.diff(steps) != pd.Timedelta(minutes=1).value]
    prefix = np.r_[0, np.cumsum(breaks)]
    contiguous = prefix[cuts+cfg.horizon]-prefix[cuts-cfg.context_length+1] == 0
    ok = ((observed[cuts]-observed[cuts-cfg.context_length] >= cfg.min_observed)
          & (valid[cuts+cfg.horizon]-valid[cuts] > 0) & contiguous)
    return cuts[ok]


def sample_origins(eligible, cfg):
    if len(eligible) < cfg.n_origins:
        raise ValueError(f"Only {len(eligible)} eligible origins; need {cfg.n_origins}")
    return np.sort(np.random.default_rng(cfg.seed).choice(eligible, cfg.n_origins, replace=False))


def context_batch(values, night, cuts, cfg):
    return np.stack([fill_context(values[c-cfg.context_length:c], night[c-cfg.context_length:c],
                                  cfg.nan_policy, cfg.short_gap) for c in cuts]).astype(np.float32)


class ErrorMoments:
    """Stable pooled population variance using block-wise centered moments."""
    def __init__(self):
        self.n, self.mean, self.m2, self.absolute = 0, 0., 0., 0.

    def update(self, prediction, target):
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise ValueError("Predictions must be complete on the common target mask")
        valid = np.isfinite(target)
        e = np.asarray(prediction, np.float64)[valid]-np.asarray(target, np.float64)[valid]
        n = len(e)
        if not n:
            return
        mean = float(e.mean())
        delta = mean-self.mean
        self.m2 += float(np.sum((e-mean)**2))+delta**2*self.n*n/(self.n+n)
        self.mean += delta*n/(self.n+n)
        self.absolute += float(abs(e).sum())
        self.n += n

    def result(self):
        if not self.n:
            raise ValueError("No evaluated targets")
        return {"n_errors": self.n, "MAE_W": self.absolute/self.n,
                "bias_W": self.mean, "variance_W2": self.m2/self.n}


def run_solar(csv_path, root, cfg=SolarConfig(), models=MODELS, device=None):
    if cfg.mode == "full" and (cfg.n_origins != 50000 or tuple(models) != MODELS):
        raise ValueError("Full solar requires 50,000 origins and all 16 checkpoints")
    timestamps, values, night, targets = load_solar(csv_path, cfg)
    eligible = eligible_origins(timestamps, values, targets, cfg)
    cuts = sample_origins(eligible, cfg)
    out, key = make_run(root, "solar", cfg,
                        {"dataset_sha256": cp.sha256_file(Path(csv_path)),
                         "origins_sha256": array_hash(cuts)}, models)
    save_csv(out/"origins.csv", pd.DataFrame({"position": cuts, "time": timestamps.iloc[cuts].to_numpy()}))
    counts = dict(eligible_origins=int(len(eligible)), sampled_origins=int(len(cuts)),
                  observed=0, assumed_night_zero=0, excluded_daytime=0)
    for c in cuts:
        observed = np.isfinite(values[c:c+cfg.horizon])
        nocturnal = night[c:c+cfg.horizon]
        counts["observed"] += int(observed.sum())
        counts["assumed_night_zero"] += int((~observed & nocturnal).sum())
        counts["excluded_daytime"] += int((~observed & ~nocturnal).sum())
    cp.atomic_json(out/"target_counts.json", counts)
    rows = []
    for spec in models:
        start_time = time.perf_counter()
        forecast = None
        def predict(x):
            nonlocal forecast
            if forecast is None:
                print(f"Loading {spec.label}", flush=True)
                forecast = Forecaster(spec, device, cfg.batch_size)
            return forecast.forecast(x)
        moments = ErrorMoments()
        try:
            for start in range(0, len(cuts), cfg.block_size):
                block = cuts[start:start+cfg.block_size]
                x = context_batch(values, night, block, cfg)
                y = cached_forecast(out/spec.tag/f"block_{start:06d}.npz", x, predict, key)
                truth = np.stack([targets[c:c+cfg.horizon] for c in block])
                moments.update(y, truth)
                elapsed = time.perf_counter()-start_time
                done = start+len(block)
                cp.atomic_json(out/"progress.json", {"model": spec.tag, "origins_completed": done,
                                "total_origins": len(cuts), "seconds": elapsed, "run_key": key})
                if start == 0 or done == len(cuts) or done % (cfg.block_size*5) == 0:
                    print(f"{spec.tag}: {done:,}/{len(cuts):,} origins, {elapsed:.1f}s", flush=True)
        finally:
            if forecast is not None:
                forecast.close()
        metric = moments.result()
        assert metric["n_errors"] == counts["observed"]+counts["assumed_night_zero"]
        rows.append(dict(model=spec.label, P=spec.P, S=spec.S, n_origins=len(cuts), **metric))
        save_csv(out/"summary.partial.csv", pd.DataFrame(rows))
    table = pd.DataFrame(rows)
    save_csv(out/"summary.csv", table)
    cp.atomic_json(out/"complete.json", {"rows": len(table), "mode": cfg.mode, "run_key": key})
    return table, out


def benchmark_solar(csv_path, root, cfg=SolarConfig(), models=MODELS, device=None):
    """Warm and time one real batch per geometry; loading excluded from ETA."""
    timestamps, values, night, targets = load_solar(csv_path, cfg)
    cuts = sample_origins(eligible_origins(timestamps, values, targets, cfg), cfg)[:cfg.batch_size]
    x = context_batch(values, night, cuts, cfg)
    rows = []
    for spec in models:
        forecaster = Forecaster(spec, device, cfg.batch_size)
        try:
            forecaster.forecast(x)
            start = time.perf_counter()
            forecaster.forecast(x)
            seconds = time.perf_counter()-start
        finally:
            forecaster.close()
        rows.append({"model": spec.label, "batch": len(x), "seconds": seconds,
                     "estimated_full_seconds": seconds/len(x)*cfg.n_origins})
        print(f"Benchmark {spec.label}: {seconds:.3f}s/{len(x)} origins", flush=True)
    result = pd.DataFrame(rows)
    save_csv(Path(root)/"benchmark_solar.csv", result)
    print(f"Estimated total inference: {result.estimated_full_seconds.sum()/3600:.2f} hours "
          "(excludes loading, preparation and disk I/O)", flush=True)
    return result
