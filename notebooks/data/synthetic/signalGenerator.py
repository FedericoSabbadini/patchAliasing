"""
signalGenerator.py — Entry point for signal generation.

inject format: list of {freq_hz, amplitude, phase} dicts.
cpp is a derived read-only quantity (freq_hz * P / fs).
"""
from generators.tsmixup_generator import TSMixupGenerator
from generators.kernelsynth_generator import KernelSynthGenerator
import json
from pathlib import Path
import numpy as np

# ============================================================
DEFAULT_SEED = 3
BASE_DIR     = Path(__file__).resolve().parent
OUTPUT_DIR   = BASE_DIR / "signals"
# ============================================================


def runTSMixup(p: dict, seed: int, output_dir: Path = OUTPUT_DIR) -> TSMixupGenerator:
    return TSMixupGenerator(
        K=p["K"], alpha=p["alpha"], l_min=p["l_min"], l_max=p["l_max"],
        t_lengths=p.get("t_lengths"),
        data_mode=p.get("data_mode", "synthetic"),
        pool_freqs=p.get("pool_freqs"),
        output_dir=output_dir, seed=seed,
        inject=p.get("inject"),        # list of {freq_hz, amplitude, phase}
        P=p.get("P", 16),
        fs=p.get("fs", 1.0),
    )


def runKernelSynth(p: dict, seed: int, output_dir: Path = OUTPUT_DIR) -> KernelSynthGenerator:
    return KernelSynthGenerator(
        J=p["J"], l_syn=p["l_syn"],
        jitter=p.get("jitter", 1e-6),
        fs=p["fs"],
        output_dir=output_dir, seed=seed,
        inject=p.get("inject"),        # list of {freq_hz, amplitude, phase}
        P=p.get("P", 16),
    )


def plot_print(generator, signal=None):
    if signal is None:
        signal = generator.generate()
    generator.save(signal)
    generator.plot(signal)
    params = generator.getParameters()
    print(f"\n{Path(generator.path()).name}")
    print(f"  shape={signal.shape}  std={signal.std():.4f}")
    if params.get("inject"):
        for c in params["inject"]:
            print(f"  injected: {c['freq_hz']} Hz  amp={c['amplitude']}  cpp={c['cpp']:.4f}")
    return signal


def runRandomSignal(K=None, alpha=None, l_min=None, l_max=None, fs=None, seed=None):
    """
    Generate a random TSMixup signal with no injection.
    Use to build diverse background controls for SLD-MDL probing.

    All parameters are randomised if not provided. The caller is responsible
    for ensuring fs matches the downstream pipeline (e.g., pass fs=FS explicitly
    when feeding signals into a model with a fixed sampling rate assumption).
    """
    if seed is None:
        seed = np.random.randint(0, 10000)
    # Use a fresh RNG seeded reproducibly, no global state mutation
    local_rng = np.random.default_rng(seed)
    return runTSMixup({
        "K":     int(local_rng.integers(2, 6))           if K     is None else K,
        "alpha": float(local_rng.uniform(0.5, 2.0))      if alpha is None else alpha,
        "l_min": int(local_rng.integers(50, 200))        if l_min is None else l_min,
        "l_max": int(local_rng.integers(200, 500))       if l_max is None else l_max,
        "fs":    float(local_rng.choice([128, 256, 512])) if fs is None else fs,
    }, seed).generate()


if __name__ == "__main__":
    with open(BASE_DIR / "signals.json", "r") as f:
        configs = json.load(f)

    for cfg in configs:
        n_real    = cfg.get("n_realizations", 1)
        base_seed = cfg.get("seed", DEFAULT_SEED)
        for r in range(n_real):
            seed = base_seed + r
            if cfg["generator"] == "tsmixup":
                gen = runTSMixup(cfg["params"], seed, OUTPUT_DIR)
            elif cfg["generator"] == "kernelsynth":
                gen = runKernelSynth(cfg["params"], seed, OUTPUT_DIR)
            else:
                print(f"Unknown generator type: {cfg['generator']}")
                continue
            plot_print(gen)
