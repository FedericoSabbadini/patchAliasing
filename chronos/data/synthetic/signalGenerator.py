from generators.tsmixup_generator import TSMixupGenerator
from generators.kernelsynth_generator import KernelSynthGenerator
import json
from pathlib import Path

# ============================================================
DEFAULT_SEED = 3
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR  = BASE_DIR / "signals"

# ============================================================

def runTSMixup(p, seed, name=None):
    gen = TSMixupGenerator(
        K=p["K"], alpha=p["alpha"], l_min=p["l_min"], l_max=p["l_max"],
        t_lengths=p.get("t_lengths"), output_dir=OUTPUT_DIR, seed=seed,
        inject=p.get("inject"), P=p.get("P", 16), fs=p.get("fs", 1.0), name=name
    )
    sig = gen.generate(); gen.save(sig); gen.plot(sig)
    print(f"TSMixup     -> {Path(gen.path()).name} shape={sig.shape} std={sig.std():.4f}")

def runKernelSynth(p, seed, name=None):
    gen = KernelSynthGenerator(
        J=p["J"], l_syn=p["l_syn"], jitter=p.get("jitter", 1e-6), fs=p["fs"],
        output_dir=OUTPUT_DIR, seed=seed, inject=p.get("inject"), P=p.get("P", 16), name=name
    )
    sig = gen.generate(); gen.save(sig); gen.plot(sig)
    print(f"KernelSynth -> {Path(gen.path()).name} shape={sig.shape} std={sig.std():.4f}")

if __name__ == "__main__":
    with open(BASE_DIR / "signals.json", "r") as f:
        configs = json.load(f)

    for cfg in configs:
        n_real    = cfg.get("n_realizations", 1)
        base_seed = cfg.get("seed", DEFAULT_SEED)
        name      = cfg.get("name", None)
        for r in range(n_real):
            seed = base_seed + r
            rid  = r if n_real > 1 else None
            if cfg["generator"] == "tsmixup":
                runTSMixup(cfg["params"], seed, name)
            elif cfg["generator"] == "kernelsynth":
                runKernelSynth(cfg["params"], seed, name)
            else:
                print(f"Unknown generator type: {cfg['generator']}")