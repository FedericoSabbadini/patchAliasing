"""Shared-window solar benchmark, including the Colab quantile diagnostics.

CPU workers own separate checkpoints. Importing this module does not start workers.
The legacy 50k comparison API and its caches are deliberately preserved.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from . import comparison_lib as c
else:
    import comparison_lib as c
if __package__:
    from . import checkpointing as cp
else:
    import checkpointing as cp


@dataclass(frozen=True)
class SolarBenchmarkConfig:
    mode: str = "full"
    n_draws: int = 150
    allow_overlap: bool = False
    seed: int = 42
    channel: str = "PV_Power"
    context_length: int = 2048
    horizon: int = 64
    min_observed: int = 60
    spike_max: float = 1000.
    nan_policy: str = "zero_interp_short"
    short_gap: int = 15
    batch_size: int = 64
    workers: int = 4
    threads_per_worker: int = 2
    n_bootstrap: int = 2000


def sample_windows(eligible, cfg):
    """Seeded random packing of entire context+target intervals, or uniform draws.

    If random packing is too sparse, a maximum-cardinality earliest-first packing
    provides a feasible fallback. Nonoverlap is not a uniform subset design.
    """
    eligible = np.unique(np.asarray(eligible, dtype=np.int64))
    if cfg.n_draws < 1 or len(eligible) < cfg.n_draws:
        raise ValueError(f"Need {cfg.n_draws} draws; only {len(eligible)} eligible origins")
    rng = np.random.default_rng(cfg.seed)
    if cfg.allow_overlap:
        return np.sort(rng.choice(eligible, cfg.n_draws, replace=False))
    length = cfg.context_length + cfg.horizon
    selected = []
    available = eligible.copy()
    while len(available) and len(selected) < cfg.n_draws:
        cut = int(rng.choice(available))
        selected.append(cut)
        available = available[np.abs(available-cut) >= length]
    if len(selected) < cfg.n_draws:
        packed, next_cut = [], -1
        for cut in eligible:
            if cut >= next_cut:
                packed.append(cut)
                next_cut = cut + length
        if len(packed) < cfg.n_draws:
            raise ValueError(f"At most {len(packed)} disjoint windows fit; requested {cfg.n_draws}")
        selected = rng.choice(packed, cfg.n_draws, replace=False)
    return np.sort(np.asarray(selected, dtype=np.int64))


def dominant_context_frequency(contexts):
    """Hann dominant non-DC frequency in cycles/sample (= cycles/minute here)."""
    x = np.asarray(contexts, float)
    finite = np.isfinite(x).all(axis=1)
    x = np.where(np.isfinite(x), x, 0.)
    x = x - x.mean(axis=1, keepdims=True)
    mag = abs(np.fft.rfft(x*np.hanning(x.shape[1]), axis=1))
    mag[:, 0] = 0
    f = np.fft.rfftfreq(x.shape[1])[mag.argmax(axis=1)]
    f[(np.max(abs(x), axis=1) <= 1e-12) | ~finite] = np.nan
    return f


def load_data(path, cfg):
    if cfg.channel == "PV_Power":
        return c.load_solar(path, cfg)
    raw = pd.read_csv(path, sep=";")
    timestamps = pd.to_datetime(raw.Time, dayfirst=True)
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("Timestamps must be ordered and unique")
    values = pd.to_numeric(raw[cfg.channel], errors="coerce").to_numpy(float, copy=True)
    values[(values == -999999) | ~np.isfinite(values)] = np.nan
    return timestamps, values, c.solar_elevation(timestamps) <= 0, values.copy()


def prepare(path, root, cfg, models, extra_identity=None, kind="solar_benchmark"):
    if cfg.context_length != 2048 or cfg.horizon != 64:
        raise ValueError("This benchmark requires context 2048 and horizon 64")
    if cfg.mode not in ("full", "smoke") or cfg.workers < 1 or cfg.batch_size < 1:
        raise ValueError("Invalid execution configuration")
    times, values, night, targets = load_data(path, cfg)
    eligible = c.eligible_origins(times, values, targets, cfg)
    if cfg.channel != "PV_Power":
        # Colab G_h protocol: complete raw windows, no nighttime assumptions.
        p = np.r_[0, np.cumsum(np.isfinite(values))]
        eligible = eligible[p[eligible+cfg.horizon]-p[eligible-cfg.context_length]
                            == cfg.context_length+cfg.horizon]
    cuts = sample_windows(eligible, cfg)
    out, key = c.make_run(root, kind, cfg,
        {"dataset_sha256": cp.sha256_file(Path(path)), "origins": c.array_hash(cuts),
         "benchmark_code": cp.sha256_file(Path(__file__)), "extra": extra_identity}, models)
    x_path = out/"contexts.npy"
    if not x_path.exists():
        def write_contexts(temp):
            x = np.lib.format.open_memmap(temp, mode="w+", dtype="float32",
                                         shape=(len(cuts), cfg.context_length))
            for start in range(0, len(cuts), cfg.batch_size):
                block = cuts[start:start+cfg.batch_size]
                x[start:start+len(block)] = (c.context_batch(values, night, block, cfg)
                    if cfg.channel == "PV_Power" else
                    np.stack([values[i-cfg.context_length:i] for i in block]))
            x.flush()
            del x
        cp.atomic_write(x_path, write_contexts)
    truth = np.stack([targets[i:i+cfg.horizon] for i in cuts])
    observed = np.stack([np.isfinite(values[i:i+cfg.horizon]) for i in cuts])
    assumed = np.isfinite(truth) & ~observed
    counts = {"eligible_origins": len(eligible), "sampled_origins": len(cuts),
              "observed": int(observed.sum()), "assumed_night_zero": int(assumed.sum()),
              "excluded_daytime": int((~np.isfinite(truth)).sum())}
    c.save_csv(out/"target_counts.csv", pd.DataFrame([counts]))
    cp.atomic_json(out/"target_counts.json", counts)
    cp.atomic_npy(out/"targets.npy", truth)
    x = np.load(x_path, mmap_mode="r")
    frequencies = np.concatenate([dominant_context_frequency(x[i:i+cfg.batch_size])
                                  for i in range(0, len(x), cfg.batch_size)])
    elevation = c.solar_elevation(times.iloc[cuts])
    windows = pd.DataFrame({"origin": cuts, "time": times.iloc[cuts].to_numpy(),
        "start": cuts-cfg.context_length, "end_exclusive": cuts+cfg.horizon,
        "frequency_cycles_per_minute": frequencies,
        "regime": np.where(elevation <= 0, "night", np.where(elevation >= 10, "day", "transition")),
        "n_targets": np.isfinite(truth).sum(axis=1)})
    c.save_csv(out/"windows.csv", windows)
    return out, key, windows, truth


def _worker(spec, cfg, out_string, key, device):
    # Top-level callable is necessary for multiprocessing spawn on Windows.
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(cfg.threads_per_worker)
    out = Path(out_string)
    x = np.load(out/"contexts.npy", mmap_mode="r")
    forecast = None
    started = time.perf_counter()
    with threadpool_limits(limits=cfg.threads_per_worker):
        try:
            for start in range(0, len(x), cfg.batch_size):
                inputs = np.asarray(x[start:start+cfg.batch_size])
                path = out/spec.tag/f"quantiles_{start:07d}.npz"
                input_hash = c.array_hash(inputs)
                if path.exists():
                    with np.load(path) as z:
                        if (str(z["key"]) != key or str(z["input_hash"]) != input_hash
                                or z["prediction"].shape != (len(inputs), 9, 64)
                                or not np.isfinite(z["prediction"]).all()):
                            raise ValueError(f"Invalid cache {path}")
                    continue
                if forecast is None:
                    forecast = c.Forecaster(spec, device, cfg.batch_size)
                quantiles = np.asarray(forecast.pipe.model.config.chronos_config["quantiles"])
                if not np.allclose(quantiles, np.arange(1, 10)/10):
                    raise ValueError("Expected quantiles 0.1 through 0.9")
                with torch.inference_mode():
                    prediction = forecast.pipe.predict(torch.tensor(inputs.copy(), device=device),
                        prediction_length=64).float().cpu().numpy()
                if prediction.shape != (len(inputs), 9, 64) or not np.isfinite(prediction).all():
                    raise ValueError(f"Incomplete predictions: {spec.tag}")
                c.save_npz(path, prediction=prediction, quantiles=quantiles, key=key, input_hash=input_hash)
                cp.atomic_json(out/spec.tag/"progress.json", {"done": start+len(inputs),
                    "total": len(x), "seconds": time.perf_counter()-started, "key": key})
        finally:
            if forecast is not None:
                forecast.close()
    return spec.tag, time.perf_counter()-started


def metrics(prediction, target):
    moments = c.ErrorMoments()
    moments.update(prediction, target)
    r = moments.result()
    return {"n_errors": r["n_errors"], "MAE": r["MAE_W"], "bias": r["bias_W"],
            "variance": r["variance_W2"], "RMSE": np.sqrt(r["variance_W2"]+r["bias_W"]**2)}


def summarize(out, windows, truth, cfg, models):
    rows, quantile_rows, profile_rows, regime_rows, horizon_rows = [], [], [], [], []
    window_abs, window_n = {}, np.isfinite(truth).sum(axis=1)
    valid = np.isfinite(truth)
    for spec in models:
        parts = sorted((out/spec.tag).glob("quantiles_*.npz"))
        predictions = []
        for part in parts:
            with np.load(part) as z:
                predictions.append(z["prediction"])
        y = np.concatenate(predictions)
        if y.shape != (len(truth), 9, cfg.horizon) or not np.isfinite(y).all():
            raise ValueError("Incomplete run: refusing to publish a full table")
        median = y[:, 4]
        row = dict(model=spec.label, tag=spec.tag, P=spec.P, S=spec.S,
                   n_origins=len(truth), channel=cfg.channel, **metrics(median, truth))
        row["MAE_unit"] = "W" if cfg.channel == "PV_Power" else "W/m²" if cfg.channel == "G_h" else cfg.channel
        row["variance_unit"] = "W²" if cfg.channel == "PV_Power" else "("+row["MAE_unit"]+")²"
        pinballs = []
        for qi, q in enumerate(np.arange(1, 10)/10):
            e = (truth-y[:, qi])[valid]
            loss = np.maximum(q*e, (q-1)*e)
            pinballs.append(loss.sum())
            quantile_rows.append(dict(model=spec.label, quantile=q,
                pinball=loss.mean(), doubled_pinball=2*loss.mean(), coverage=float((e <= 0).mean()),
                coverage_minus_nominal=float((e <= 0).mean()-q)))
        denominator = abs(truth[valid]).sum()
        row["WQL_sum"] = 2*sum(pinballs)/denominator if denominator else np.nan
        row["WQL_mean"] = row["WQL_sum"]/9
        rows.append(row)
        window_abs[spec.tag] = np.nansum(abs(median-truth), axis=1)
        for frequency in np.unique(windows.frequency_cycles_per_minute.dropna()):
            mask = windows.frequency_cycles_per_minute.to_numpy() == frequency
            profile_rows.append(dict(model=spec.label, P=spec.P, S=spec.S,
                frequency_cycles_per_minute=frequency, cpp=frequency*spec.P, cps=frequency*spec.S,
                n_origins=int(mask.sum()), **metrics(median[mask], truth[mask])))
        for regime in ("night", "transition", "day"):
            mask = windows.regime.to_numpy() == regime
            if mask.any():
                regime_rows.append(dict(model=spec.label, regime=regime, **metrics(median[mask], truth[mask])))
        for step in range(cfg.horizon):
            if np.isfinite(truth[:, step]).any():
                horizon_rows.append(dict(model=spec.label, step=step+1,
                    **metrics(median[:, step], truth[:, step])))
    table = pd.DataFrame(rows)
    frames = {"summary": table, "quantiles": pd.DataFrame(quantile_rows),
              "frequency_profiles": pd.DataFrame(profile_rows, columns=["model", "P", "S",
                  "frequency_cycles_per_minute", "cpp", "cps", "n_origins", "n_errors",
                  "MAE", "bias", "variance", "RMSE"]),
              "regimes": pd.DataFrame(regime_rows), "horizon": pd.DataFrame(horizon_rows)}
    # Shared weekly-cluster bootstrap; disjoint windows alone do not imply independence.
    week = pd.DatetimeIndex(windows.time).to_period("W").astype(str)
    groups = np.unique(week)
    rng = np.random.default_rng(cfg.seed+1)
    weights = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), cfg.n_bootstrap)
    group_n = np.array([window_n[week == g].sum() for g in groups])
    denominator = weights @ group_n
    absolute = {tag: np.array([a[week == g].sum() for g in groups]) for tag, a in window_abs.items()}
    for name, reference in (("versus_official", c.MODELS[0]), ("versus_retrained16", c.MODELS[4])):
        if reference.tag not in absolute:
            continue
        comparisons = []
        for spec in models:
            delta = absolute[spec.tag]-absolute[reference.tag]
            boot = weights @ delta/denominator
            lo, hi = np.quantile(boot, [.025, .975]) if len(groups) >= 2 and len(boot) else (np.nan, np.nan)
            comparisons.append(dict(model=spec.label, reference=reference.label,
                delta_MAE=delta.sum()/group_n.sum(), ci_low=lo, ci_high=hi,
                temporal_clusters=len(groups), bootstrap_draws=cfg.n_bootstrap))
        frames[name] = pd.DataFrame(comparisons)
    for name, frame in frames.items():
        c.save_csv(out/(name+".csv"), frame)
    return frames


def plot_results(frames, out):
    import matplotlib.pyplot as plt
    figures = {}
    table = frames["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 7), layout="constrained")
    for ax, metric in zip(axes, ("MAE", "bias", "variance")):
        ax.barh(table.model, table[metric], color=["#d62728"]+["#327da8"]*(len(table)-1))
        ax.invert_yaxis()
        ax.set_title(metric)
        ax.set_xlabel(table.variance_unit.iloc[0] if metric == "variance" else table.MAE_unit.iloc[0])
        ax.grid(axis="x", alpha=.2)
    fig.savefig(out/"metrics.png", dpi=150)
    figures["metrics"] = fig
    profile = frames["frequency_profiles"]
    for coordinate in ("cps", "cpp"):
        fig, axes = plt.subplots(int(np.ceil(len(table)/4)), 4, figsize=(19, 3.3*np.ceil(len(table)/4)),
                                  squeeze=False, layout="constrained")
        for ax, model in zip(axes.flat, table.model):
            p = profile[profile.model == model].sort_values(coordinate)
            other = ax.twinx()
            ax.plot(p[coordinate], p.MAE, "o-", color="#1678ac", label="MAE")
            other.plot(p[coordinate], p.variance, "s--", color="#ba3b35", label="Variance")
            ax.set(title=model, xlabel=coordinate, ylabel=f"MAE [{table.MAE_unit.iloc[0]}]")
            other.set_ylabel(f"Variance [{table.variance_unit.iloc[0]}]", color="#ba3b35")
            ax.tick_params(axis="y", colors="#1678ac")
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(table):]:
            ax.set_visible(False)
        fig.suptitle(f"Context dominant frequency: {coordinate} vs MAE and variance (separate y scales)")
        fig.savefig(out/f"{coordinate}_errors.png", dpi=150)
        figures[coordinate] = fig
    return figures


def run_benchmark(path, root, cfg=SolarBenchmarkConfig(), models=c.MODELS, device=None):
    import torch
    if cfg.mode == "full" and tuple(models) != c.MODELS:
        raise ValueError("Full benchmark requires all 16 checkpoints")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out, key, windows, truth = prepare(path, root, cfg, models)
    workers = min(cfg.workers, len(models)) if device == "cpu" else 1
    timings = []
    start = time.perf_counter()
    if workers == 1:
        for spec in models:
            tag, elapsed = _worker(spec, cfg, str(out), key, device)
            timings.append(dict(model=tag, seconds=elapsed))
            print(f"{tag}: {elapsed:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            jobs = [pool.submit(_worker, s, cfg, str(out), key, device) for s in models]
            for job in as_completed(jobs):
                tag, elapsed = job.result()
                timings.append(dict(model=tag, seconds=elapsed))
                print(f"{tag}: {elapsed:.1f}s ({len(timings)}/{len(models)})", flush=True)
    c.save_csv(out/"timings.csv", pd.DataFrame(timings))
    frames = summarize(out, windows, truth, cfg, models)
    cp.atomic_json(out/"complete.json", {"key": key, "mode": cfg.mode, "n_origins": len(windows),
        "models": len(models), "workers": workers, "wall_seconds": time.perf_counter()-start})
    return frames, out


def benchmark_cpu(path, root, cfg=SolarBenchmarkConfig(), models=c.MODELS):
    """Real warm CPU batch; sequential service-time estimate, not parallel wall ETA."""
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(cfg.threads_per_worker)
    times, values, night, targets = load_data(path, cfg)
    eligible = c.eligible_origins(times, values, targets, cfg)
    if cfg.channel != "PV_Power":
        prefix = np.r_[0, np.cumsum(np.isfinite(values))]
        eligible = eligible[prefix[eligible+cfg.horizon]-prefix[eligible-cfg.context_length]
                            == cfg.context_length+cfg.horizon]
    cuts = sample_windows(eligible, cfg)[:cfg.batch_size]
    x = (c.context_batch(values, night, cuts, cfg) if cfg.channel == "PV_Power" else
         np.stack([values[i-cfg.context_length:i] for i in cuts]).astype(np.float32))
    rows = []
    with threadpool_limits(cfg.threads_per_worker):
        for spec in models:
            model = c.Forecaster(spec, "cpu", cfg.batch_size)
            try:
                model.forecast(x)
                start = time.perf_counter()
                model.forecast(x)
                elapsed = time.perf_counter()-start
            finally:
                model.close()
            rows.append(dict(model=spec.label, batch=len(x), seconds=elapsed,
                estimated_service_seconds=elapsed*cfg.n_draws/len(x)))
            print(f"CPU {spec.tag}: {elapsed:.2f}s / {len(x)}", flush=True)
    result = pd.DataFrame(rows)
    c.save_csv(Path(root)/cfg.mode/"cpu_benchmark.csv", result)
    print(f"Estimated sequential inference: {result.estimated_service_seconds.sum():.1f}s; "
          f"{cfg.workers} workers requested. Loading, I/O and contention excluded.", flush=True)
    return result
