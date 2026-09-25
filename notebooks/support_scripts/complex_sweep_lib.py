"""Pagani-style binary probes and dominant-output sweeps on ten-component signals."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
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
if __package__:
    from . import probe_lib as pl
else:
    import probe_lib as pl
from threadpoolctl import threadpool_limits

MODELS16 = (c.MODELS[0], c.MODELS[4])


@dataclass(frozen=True)
class SweepConfig:
    mode: str = "full"
    seed: int = 43100
    n_backgrounds: int = 10
    components: int = 10
    amplitude_multiplier: float = 1.5
    context_length: int = 512
    rollout_length: int = 512
    n_phase_probe: int = 10
    n_phase_rollout: int = 4
    frequency_step: int = 1
    batch_size: int = 64
    alpha: float = 1.5
    min_separation: float = 10.


def backgrounds(family, cfg):
    n = cfg.context_length+cfg.rollout_length
    t = np.arange(n)/c.FS
    signals, rows = [], []
    for background in range(cfg.n_backgrounds):
        seed = cfg.seed+background
        rng = np.random.default_rng(seed)
        if family == "tsmixup":
            pool = c.recovery_pool()
            f = []
            for _ in range(10000):
                candidate = float(rng.choice(pool))
                if all(abs(candidate-x) >= cfg.min_separation for x in f):
                    f.append(candidate)
                if len(f) == cfg.components:
                    break
            if len(f) != cfg.components:
                raise ValueError("Cannot sample ten separated frequencies")
            f = np.asarray(f)
            phase = np.full(len(f), -np.pi/2)  # original TSM sine, stored as cosine phase
            unit = np.cos(2*np.pi*f[:, None]*t+phase[:, None])
            amp = rng.dirichlet(np.full(len(f), cfg.alpha))/np.mean(abs(unit), axis=1)
        elif family == "kernelsynth":
            import signalGenerator as sg
            scratch = Path(__file__).parents[1]/"_gen_tmp"
            scratch.mkdir(exist_ok=True)
            gen = sg.runKernelSynth(dict(J=5, l_syn=n, fs=c.FS, jitter=1e-4, P=16), seed, scratch)
            raw = np.asarray(gen.generate(), float).ravel()
            if len(raw) != n or not np.isfinite(raw).all():
                raise ValueError("Invalid KernelSynth realization")
            fft = np.fft.rfft(raw-raw.mean())
            grid = np.fft.rfftfreq(n, 1/c.FS)
            eligible = np.flatnonzero((grid >= 2) & (grid <= 250))
            # Exact Fourier terms, not ten covariance kernels or a broadband residual.
            idx = eligible[np.argsort(abs(fft[eligible]), kind="stable")[-cfg.components:]]
            f, amp, phase = grid[idx], 2*abs(fft[idx])/n, np.angle(fft[idx])
            unit = np.cos(2*np.pi*f[:, None]*t+phase[:, None])
        else:
            raise ValueError(family)
        signal = amp @ unit
        scale = signal.std()
        if scale <= 1e-12:
            raise ValueError("Flat background")
        amp = amp/scale
        signals.append(amp @ unit)
        rows.extend(dict(family=family, background=background, seed=seed, component=i,
            frequency_hz=float(freq), amplitude=float(a), cosine_phase=float(p))
            for i, (freq, a, p) in enumerate(zip(f, amp, phase)))
    return np.asarray(signals, dtype=np.float32), pd.DataFrame(rows)


def make_inputs(backgrounds, components, cfg, n_phases):
    records, inputs = [], []
    t = np.arange(cfg.context_length)/c.FS
    for f in np.arange(2, 251, cfg.frequency_step):
        for background, signal in enumerate(backgrounds):
            amplitude = cfg.amplitude_multiplier*components.loc[components.background == background, "amplitude"].max()
            for phase_index, phase in enumerate(pl.phases_Sf(float(f), n_phases)):
                inputs.append(signal[:cfg.context_length]+amplitude*np.sin(2*np.pi*f*t+phase))
                records.append(dict(background=background, seed=cfg.seed+background,
                    injected_hz=float(f), phase_index=phase_index, phase=float(phase), amplitude=amplitude))
    return np.asarray(inputs, dtype=np.float32), pd.DataFrame(records)


def dominant_output(y):
    y = np.asarray(y, float)
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite output")
    centered = y-y.mean(axis=1, keepdims=True)
    amplitude = abs(np.fft.rfft(centered, axis=1))
    amplitude[:, 0] = 0
    freq = np.fft.rfftfreq(y.shape[1], 1/c.FS)[amplitude.argmax(axis=1)]
    freq[np.max(abs(centered), axis=1) <= 1e-12] = np.nan
    return freq


def capture_features(forecaster, x):
    output = []
    def hook(module, args, value):
        value = value[0] if isinstance(value, tuple) else value
        a = value.detach().float().cpu().numpy()
        output.append(a[:, 0, :] if a.ndim == 3 else a)
    handle = forecaster.pipe.model.output_patch_embedding.register_forward_hook(hook)
    try:
        forecaster.forecast(x)
    finally:
        handle.remove()
    features = np.concatenate(output)
    if len(features) != len(x) or not np.isfinite(features).all():
        raise ValueError("Invalid output-head features")
    return features


@threadpool_limits.wrap(limits=2)
def binary_probes(features, trials, cfg):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score
    f = trials.injected_hz.to_numpy()
    records = []
    def evaluate(mask, labels, groups, task, coordinate=np.nan):
        x, y, g = features[mask], labels, groups
        folds = min(5, *(len(np.unique(g[y == cls])) for cls in (0, 1)))
        if folds < 2:
            return
        splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=cfg.seed)
        for fold, (train, test) in enumerate(splitter.split(x, y, g)):
            if len(np.unique(y[train])) < 2:
                continue
            pipe = make_pipeline(StandardScaler(), PCA(n_components=min(20, len(train)-1, x.shape[1]), random_state=cfg.seed),
                                 LogisticRegression(max_iter=2000, random_state=cfg.seed))
            pipe.fit(x[train], y[train])
            correct = pipe.predict(x[test]) == y[test]
            coordinates = f[mask][test] if np.isnan(coordinate) else np.full(len(test), coordinate)
            for point in np.unique(coordinates):
                subset = coordinates == point
                records.append(dict(task=task, frequency_hz=point, fold=fold,
                    accuracy=float(correct[subset].mean()), n_test=int(subset.sum())))
    for name, lo, hi, split in pl.BAND_TASKS:
        mask = (f >= lo) & (f <= hi)
        evaluate(mask, (f[mask] > split).astype(int), f[mask], name)
    # Local discrimination holds out entire backgrounds, so phases cannot leak across folds.
    for center in np.arange(3, 250, cfg.frequency_step):
        mask = (f == center-1) | (f == center+1)
        if mask.any():
            evaluate(mask, (f[mask] > center).astype(int), trials.background.to_numpy()[mask], "local ±1", center)
    return pd.DataFrame(records, columns=["task", "frequency_hz", "fold", "accuracy", "n_test"])


def run_family(family, root, cfg=SweepConfig(), models=MODELS16, device="cpu"):
    if cfg.components != 10 or cfg.rollout_length % 64 or cfg.rollout_length < 64:
        raise ValueError("Require ten components and a rollout in native 64-point chunks")
    if any((s.P, s.S) != (16, 16) for s in models):
        raise ValueError("This sweep is restricted to 16-16")
    bg, components = backgrounds(family, cfg)
    out, key = c.make_run(root, "complex_sweep_"+family, cfg,
        {"background_hash": c.array_hash(bg), "code": cp.sha256_file(Path(__file__))}, models)
    c.save_npz(out/"backgrounds.npz", signals=bg)
    c.save_csv(out/"components.csv", components)
    probe_x, probe_trials = make_inputs(bg, components, cfg, cfg.n_phase_probe)
    rollout_x, rollout_trials = make_inputs(bg, components, cfg, cfg.n_phase_rollout)
    c.save_csv(out/"probe_trials.csv", probe_trials)
    c.save_csv(out/"rollout_inputs.csv", rollout_trials)
    outputs = {}
    for spec in models:
        forecaster = None
        all_features, all_dominant = [], []
        started = time.perf_counter()
        try:
            for kind, inputs in (("probe", probe_x), ("rollout", rollout_x)):
                for start in range(0, len(inputs), cfg.batch_size):
                    x = inputs[start:start+cfg.batch_size]
                    path = out/spec.tag/f"{kind}_{start:07d}.npz"
                    h = c.array_hash(x)
                    if path.exists():
                        with np.load(path) as z:
                            if str(z["key"]) != key or str(z["input_hash"]) != h:
                                raise ValueError("Cache identity mismatch")
                            value = z["value"]
                    else:
                        if forecaster is None:
                            forecaster = c.Forecaster(spec, device, cfg.batch_size)
                        if kind == "probe":
                            value = capture_features(forecaster, x)
                        else:
                            history, chunks = x.copy(), []
                            for _ in range(cfg.rollout_length//64):
                                chunk = forecaster.forecast(history)
                                chunks.append(chunk)
                                history = np.concatenate([history, chunk], axis=1)[:, -cfg.context_length:]
                            value = np.concatenate(chunks, axis=1)
                        c.save_npz(path, key=key, input_hash=h, value=value)
                    if len(value) != len(x) or not np.isfinite(value).all():
                        raise ValueError("Incomplete sweep block")
                    if kind == "probe":
                        all_features.append(value)
                    else:
                        all_dominant.append(dominant_output(value))
                    cp.atomic_json(out/spec.tag/"progress.json", dict(kind=kind, done=start+len(x),
                        total=len(inputs), seconds=time.perf_counter()-started))
                print(f"{family} {spec.tag}: {kind} complete ({time.perf_counter()-started:.1f}s)", flush=True)
            probes = binary_probes(np.concatenate(all_features), probe_trials, cfg)
            red = rollout_trials.copy()
            red["dominant_output_hz"] = np.concatenate(all_dominant)
            aggregate = red.groupby("injected_hz").dominant_output_hz.agg(["mean", "std", "count"]).reset_index()
            aggregate["total_trials"] = red.groupby("injected_hz").size().to_numpy()
            c.save_csv(out/spec.tag/"binary_probes.csv", probes)
            c.save_csv(out/spec.tag/"dominant_trials.csv", red)
            c.save_csv(out/spec.tag/"dominant_summary.csv", aggregate)
            outputs[spec.label] = (probes, red, aggregate)
        finally:
            if forecaster is not None:
                forecaster.close()
    cp.atomic_json(out/"complete.json", dict(mode=cfg.mode, models=len(models), key=key,
        n_probe=len(probe_x), n_rollout=len(rollout_x)))
    return outputs, out


def plot_family(family, outputs, out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    colors = LinearSegmentedColormap.from_list("pagani", [(0.,"#8c1010"),(.2,"#d62728"),
        (.5,"#f2c14e"),(.8,"#8cc63f"),(1.,"#1a7a1a")])
    colors.set_bad("#dddddd")
    fig, axes = plt.subplots(len(outputs), 2, figsize=(16, 5*len(outputs)), squeeze=False, layout="constrained")
    for (label, (probes, trials, red)), (ax, heat) in zip(outputs.items(), axes):
        ax.scatter(trials.injected_hz, trials.dominant_output_hz, s=2, alpha=.035, color="#b61c1c")
        ax.plot(red.injected_hz, red["mean"], color="#c32020", lw=1, label="Mean dominant output")
        ax.fill_between(red.injected_hz, red["mean"]-red["std"], red["mean"]+red["std"], color="red", alpha=.12, label="±1 SD")
        ax.plot([2, 250], [2, 250], "k--", lw=.7, label="Identity")
        ax.set(xlim=(2,250), ylim=(0,256), xlabel="Injected frequency [Hz]", ylabel="Dominant output [Hz]", title=label)
        ax.legend(fontsize=8)
        matrix = np.full((8, 249), np.nan)
        for i, (name, lo, hi, split) in enumerate(pl.BAND_TASKS):
            part = probes[probes.task == name]
            for frequency, group in part.groupby("frequency_hz"):
                matrix[i, int(frequency)-2] = np.average(group.accuracy, weights=group.n_test)
        part = probes[probes.task == "local ±1"]
        for f, group in part.groupby("frequency_hz"):
            matrix[7, int(f)-2] = np.average(group.accuracy, weights=group.n_test)
        im = heat.imshow(matrix, aspect="auto", origin="upper", extent=(1.5,250.5,7.5,-.5),
                         cmap=colors, norm=Normalize(.5,1))
        heat.set(yticks=np.arange(8), yticklabels=[a[0] for a in pl.BAND_TASKS]+["local ±1"],
                 xlabel="Injected frequency [Hz]", title="Binary test accuracy (gray = not evaluated)")
        fig.colorbar(im, ax=heat, label="Held-out accuracy")
    fig.suptitle(f"{family}: ten components, injection 1.5 × maximum component amplitude")
    fig.savefig(out/f"{family}_pagani.png", dpi=160)
    return fig
