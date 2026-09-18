"""PV target comparison: Chronos-2 with past covariates and two Bolt baselines."""
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import pandas as pd
import checkpointing as cp
import comparison_lib as c
import solar_benchmark_lib as sb

CHRONOS2_ID = "amazon/chronos-2"
CHRONOS2_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
BASELINES = (c.MODELS[0], c.MODELS[4])


@dataclass(frozen=True)
class Chronos2Config(sb.SolarBenchmarkConfig):
    chronos2_revision: str = CHRONOS2_REVISION
    batch_size: int = 16
    threads_per_worker: int = 8


def past_covariates(raw, cut, context_length, target="PV_Power"):
    """All measured columns except target/time; never read beyond the forecast cut."""
    result = {}
    for col in raw.columns:
        if col in ("Time", target):
            continue
        x = pd.to_numeric(raw[col].iloc[cut-context_length:cut], errors="raise").to_numpy(float, copy=True)
        x[(x == -999999) | ~np.isfinite(x)] = np.nan
        result[col] = x.astype(np.float32)
    return result


def run_comparison(csv_path, root, cfg=Chronos2Config(), device="cpu"):
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(cfg.threads_per_worker)
    if cfg.channel != "PV_Power":
        raise ValueError("PV_Power is the dataset column corresponding to the requested pv_array")
    raw = pd.read_csv(csv_path, sep=";")
    columns = [col for col in raw.columns if col not in ("Time", cfg.channel)]
    out, key, windows, truth = sb.prepare(csv_path, root, cfg, BASELINES,
        extra_identity={"chronos2": CHRONOS2_ID, "revision": cfg.chronos2_revision,
            "covariates": columns, "cross_learning": False, "code": cp.sha256_file(Path(__file__))},
        kind="solar_chronos2")
    contexts = np.load(out/"contexts.npy", mmap_mode="r")
    predictions, timings = {}, []
    for spec in BASELINES:
        started = time.perf_counter()
        sb._worker(spec, cfg, str(out), key, device)
        blocks = []
        for path in sorted((out/spec.tag).glob("quantiles_*.npz")):
            with np.load(path) as z:
                blocks.append(z["prediction"][:, 4])
        predictions[spec.label] = np.concatenate(blocks)
        timings.append(dict(model=spec.label, seconds=time.perf_counter()-started))
    pipeline, blocks = None, []
    started = time.perf_counter()
    try:
        for start in range(0, len(windows), cfg.batch_size):
            cuts = windows.origin.iloc[start:start+cfg.batch_size].to_numpy()
            inputs = [dict(target=np.array(contexts[start+j]),
                past_covariates=past_covariates(raw, int(cut), cfg.context_length, cfg.channel))
                for j, cut in enumerate(cuts)]
            path = out/"chronos2"/f"block_{start:07d}.npz"
            input_hash = cp.fingerprint([{k: c.array_hash(v) for k, v in
                {"target": row["target"], **row["past_covariates"]}.items()} for row in inputs])
            if path.exists():
                with np.load(path) as z:
                    if str(z["key"]) != key or str(z["input_hash"]) != input_hash:
                        raise ValueError("Chronos-2 cache mismatch")
                    median = z["median"]
            else:
                if pipeline is None:
                    from huggingface_hub import snapshot_download
                    # Local pinned snapshot also avoids an upstream adapter lookup on unpinned main.
                    local = snapshot_download(CHRONOS2_ID, revision=cfg.chronos2_revision,
                        cache_dir=str(Path(root)/"model_cache"), allow_patterns=["*.json", "*.safetensors"])
                    pipeline = Chronos2Pipeline.from_pretrained(local, device_map=device)
                with torch.inference_mode():
                    # Each origin is an independent group; only its historical channels interact.
                    quantiles, _ = pipeline.predict_quantiles(inputs, prediction_length=64,
                        context_length=2048, quantile_levels=[.5], batch_size=cfg.batch_size*(len(columns)+1),
                        cross_learning=False)
                median = np.stack([q[0, :, 0].float().cpu().numpy() for q in quantiles])
                c.save_npz(path, median=median, input_hash=input_hash, key=key)
            if median.shape != (len(cuts), 64) or not np.isfinite(median).all():
                raise ValueError("Incomplete Chronos-2 predictions")
            blocks.append(median)
            cp.atomic_json(out/"chronos2"/"progress.json", dict(done=start+len(cuts), total=len(windows), key=key))
            print(f"Chronos-2: {start+len(cuts)}/{len(windows)}", flush=True)
    finally:
        del pipeline
    label = "Chronos-2 multivariate (past covariates)"
    predictions[label] = np.concatenate(blocks)
    timings.append(dict(model=label, seconds=time.perf_counter()-started))
    rows = [dict(model=name, input="PV_Power + "+", ".join(columns) if name == label else "PV_Power",
        context_length=2048, horizon=64, n_origins=len(windows),
        **sb.metrics(y, truth)) for name, y in predictions.items()]
    table = pd.DataFrame(rows)
    c.save_csv(out/"summary.csv", table)
    c.save_csv(out/"timings.csv", pd.DataFrame(timings))
    cp.atomic_json(out/"complete.json", dict(key=key, models=3, n_origins=len(windows), mode=cfg.mode))
    return table, predictions, truth, windows, out


def plot_comparison(table, predictions, truth, windows, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    labels = ["Official Bolt 16–16", "Retrained Bolt 16–16", "Chronos-2 + covariates"]
    for ax, metric in zip(axes, ("MAE", "bias", "variance")):
        ax.bar(labels, table[metric], color=["#888888", "#247eab", "#bc3d38"])
        ax.tick_params(axis="x", rotation=20)
        ax.set(title=metric, ylabel="W²" if metric == "variance" else "W")
    fig.savefig(out/"comparison_metrics.png", dpi=160)
    # Choose the example from targets only, identically for all models.
    index = int(np.nanargmax(np.nanmean(truth, axis=1)))
    example, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    ax.plot(np.arange(1,65), truth[index], "k", lw=2, label="Observed / assumed night zero")
    for name, y in predictions.items():
        ax.plot(np.arange(1,65), y[index], label=name)
    ax.set(xlabel="Minutes after forecast origin", ylabel="PV_Power [W]", title=str(windows.time.iloc[index]))
    ax.legend(fontsize=8)
    example.savefig(out/"forecast_example.png", dpi=160)
    return fig, example
