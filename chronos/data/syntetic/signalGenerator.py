from generators.tsmixup_generator import TSMixupGenerator
from generators.kernelsynth_generator import KernelSynthGenerator
import json
from pathlib import Path

# ============================================================
SEED        = 3
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR  = BASE_DIR / "signals"

# ============================================================
def runTSMixup(**parameters):
    gen = TSMixupGenerator(
        parameters["K"], parameters["alpha"],
        parameters["l_min"], parameters["l_max"],
        parameters["data_mode"], parameters["t_lengths"],
        parameters["data_dir"], seed=SEED, output_dir=OUTPUT_DIR,
    )
    sig = gen.generate()
    path = gen.save(sig)
    print(f"\n\nTSMixup     -> {path}  shape={sig.shape} mean={sig.mean():.4f} std={sig.std():.4f}")


def runKernelSynth(**parameters):
    gen = KernelSynthGenerator(
        parameters["J"], parameters["l_syn"],
        parameters["jitter"], seed=SEED, output_dir=OUTPUT_DIR,
    )
    sig = gen.generate()
    path = gen.save(sig)
    print(f"\n\nKernelSynth -> {path}  shape={sig.shape} mean={sig.mean():.4f} std={sig.std():.4f}")

# ============================================================


if __name__ == "__main__":

    with open(BASE_DIR / "signals.json", "r") as f:
        configs = json.load(f)

    for gen_config in configs:
        if gen_config["generator"] == "tsmixup":
            runTSMixup(**gen_config["params"])
        elif gen_config["generator"] == "kernelsynth":
            runKernelSynth(**gen_config["params"])
        else:
            print(f"Unknown generator type: {gen_config['generator']}")