from generators.tsmixup_generator import TSMixupGenerator
from generators.kernelsynth_generator import KernelSynthGenerator
import json
from pathlib import Path
import numpy as np

# ============================================================
DEFAULT_SEED = 3
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR  = BASE_DIR / "signals"

# ============================================================

def runTSMixup(p, seed, output_dir=OUTPUT_DIR):
    gen = TSMixupGenerator(
        K=p["K"], alpha=p["alpha"], l_min=p["l_min"], l_max=p["l_max"],
        t_lengths=p.get("t_lengths"), output_dir=output_dir, seed=seed,
        inject=p.get("inject"), P=p.get("P", 16), fs=p.get("fs", 1.0)
    )
    return gen

def runKernelSynth(p, seed, output_dir=OUTPUT_DIR):
    gen = KernelSynthGenerator(
        J=p["J"], l_syn=p["l_syn"], jitter=p.get("jitter", 1e-6), fs=p["fs"],
        output_dir=output_dir, seed=seed, inject=p.get("inject"), P=p.get("P", 16)
    )
    return gen

def plot_print(generator, name):
    signal = generator.generate()
    generator.save(signal)
    generator.plot(signal)
    print(f"\n{Path(generator.path()).name} shape={signal.shape} std={signal.std():.4f}")

def runRandomSignal(K=None, alpha=None, l_min=None, l_max=None, fs=None):
    """
    Generates a random signal using a tsmixup generator by a random seed
    Then, every parameter is initialized with a random value, related to the tsmixup generator type, and the signal is generated and returned. 
    The random seed is used to ensure reproducibility of the generated signal, and the random parameters are used to create a diverse set of signals that can be used for testing and analysis. The function does not take any input parameters and does not return anything, but it generates and saves a signal based on the randomly chosen generator and parameters.
    """
    # set seed for reproducibility
    seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    return runTSMixup({
        "K": np.random.randint(2, 6) if K is None else K,
        "alpha": np.random.uniform(0.5, 2.0) if alpha is None else alpha,
        "l_min": np.random.randint(50, 200) if l_min is None else l_min,
        "l_max": np.random.randint(200, 500) if l_max is None else l_max,
        "fs": np.random.choice([128, 256, 512]) if fs is None else fs
    }, seed).generate()


if __name__ == "__main__":
    with open(BASE_DIR / "signals.json", "r") as f:
        configs = json.load(f)

    for cfg in configs:
        n_real    = cfg.get("n_realizations", 1)
        base_seed = cfg.get("seed", DEFAULT_SEED)
        for r in range(n_real):
            seed = base_seed + r
            rid  = r if n_real > 1 else None
            if cfg["generator"] == "tsmixup":
                gen = runTSMixup(cfg["params"], seed, OUTPUT_DIR)
                plot_print(gen, name)
            elif cfg["generator"] == "kernelsynth":
                gen = runKernelSynth(cfg["params"], seed, OUTPUT_DIR)
                plot_print(gen, name)
            else:
                print(f"Unknown generator type: {cfg['generator']}")

