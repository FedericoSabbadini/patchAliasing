"""
tsmixup_generator.py — TSMixup augmentation as a self-contained class.
This module defines the TSMixupGenerator class, which generates synthetic time series data by mixing mean-scaled subsequences from a dataset with Dirichlet weights. 
The generator can work with either synthetic sine wave data or real data loaded from .npy files. 
The generated signals are saved as .npy files along with corresponding plots for visualization.

N.B. TSMixup is sampling-rate agnostic: it generates time-domain signals by mixing subsequences, so the sampling rate does not affect the generation process.
It works directly on the time-domain data, and the generated signal's properties (e.g., mean, std) are determined by the mixed subsequences and their weights, rather than any frequency-domain characteristics.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt


class TSMixupGenerator:
    """Chronos-style TSMixup: mix k mean-scaled subsequences with Dirichlet weights."""

    def __init__(self, K=3, 
                 alpha=1.5, 
                 l_min=128, 
                 l_max=2048, 
                 t_lengths=None,
                 
                 data_mode="synthetic",
                 data_dir=None,
                 seed=3, output_dir="./signals",

                 inject=None, 
                 P=16,
                 fs=1.0,

                 name=None,
                 ):

        self.K = K # K is the maximum number of subsequences to mix, randomly chosen between 1 and K for each signal. 
        # Higher K means more complex mixtures.
        self.alpha = alpha # alpha is the concentration parameter for the Dirichlet distribution, controlling the variability of the mixing weights. 
        # Higher alpha means more uniform weights, while lower alpha leads to more sparse mixtures.
        self.l_min, self.l_max = l_min, l_max # l_min and l_max define the range of lengths for the subsequences to be mixed.
        # Longer subsequences (higher l_max) can capture more complex patterns, while shorter ones (lower l_min) lead to more local mixing.
        self.data_mode = data_mode
        self.t_lengths = t_lengths or [500, 600, 700, 400, 550] # Default lengths for synthetic datasets if not provided. 
        # Differently from l_min/l_max, these are the lengths of the base datasets from which subsequences will be sampled.
        # we can have t_lenght different from l_min/l_max, for example we can have t_lenghts=[500, 600, 700] and l_min=128, l_max=512, which means that we will sample subsequences of length between 128 and 512 from the base datasets of length 500, 600 and 700.
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.rng = np.random.default_rng(seed) # Random number generator initialized with the given seed for reproducibility.
        self.datasets = self._build_datasets()
        self.inject = inject # inject is a dictionary specifying the parameters for the optional sinusoidal injection
        # It has the format {"mode": "cpp" or "hz", "value": float, "amplitude": float, "phase": float}
        # Easily, it is used to add a controlled sinusoidal component to the generated signal, either at a specific frequency in hertz (hz) or at a specific number of cycles per patch (cpp), which can be useful for testing the effect of tokenisation and sampling on the generated signals.
        # It allows us to isolate the effect of adding a known frequency component to the signal, and see how it interacts with the TSMixup process and the resulting signal properties.
        self.P = P # P is the patch size, needed for cpp-mode injection to define the frequency of the injected component in terms of cycles per patch.
        # It is relevant for the cpp injection mode, where the frequency of the injected sinusoidal component is defined in terms of how many cycles fit into a patch of length P.
        self.fs = fs
        self.name = name


    def _build_datasets(self) -> List[np.ndarray]:
        """Build the base datasets for mixing, either synthetic or loaded from files."""
        if self.data_mode == "synthetic":
            return [
                np.sin(np.linspace(0, 5 * np.pi * (i + 1), T))
                for i, T in enumerate(self.t_lengths)
                # for each length T in t_lengths, we create a sine wave that completes (i+1)*5 cycles over T samples, where i is the index of the dataset.
            ]

        if self.data_mode == "files":
            files = sorted(Path(self.data_dir).glob("*.npy"))
            if not files:
                raise FileNotFoundError(f"No .npy files in {self.data_dir}")
            return [np.load(f).astype(float) for f in files]

        raise ValueError(f"Unknown data_mode: {self.data_mode}")


    def _sample_subsequence(self, dataset, l) -> np.ndarray:
        """Sample a subsequence of length l from the given dataset, with wrapping if necessary."""
        T = dataset.shape[0]

        if T >= l:
            start = self.rng.integers(0, T - l + 1)
            return dataset[start:start + l].copy()

        repeats = (l // T) + 1
        return np.tile(dataset, repeats)[:l].copy()


    def generate(self) -> np.ndarray:
        """Generate a synthetic signal by mixing mean-scaled subsequences with Dirichlet weights."""
        k = self.rng.integers(1, self.K + 1) # Randomly choose how many subsequences to mix (between 1 and K).
        l = self.rng.integers(self.l_min, self.l_max + 1) # Randomly choose the length of the subsequences to mix (between l_min and l_max).
        t = np.arange(l) / self.fs

        scaled = []
        for _ in range(k):
            n = self.rng.integers(0, len(self.datasets)) # Randomly select which dataset to sample from for this subsequence.
            x = self._sample_subsequence(self.datasets[n], l) # Sample a subsequence of length l from the selected dataset.
            x = x / max(float(np.mean(np.abs(x))), 1e-8) # Mean-scale the subsequence to ensure that the mixing process is not dominated by any particular subsequence's amplitude.
            scaled.append(x) # Collect the scaled subsequences for mixing.

        lambdas = self.rng.dirichlet([self.alpha] * k) # Sample mixing weights from a Dirichlet distribution, which ensures that they sum to 1 and controls the variability of the weights based on alpha.
        signal = sum(lam * s for lam, s in zip(lambdas, scaled)) # Mix the scaled subsequences using the sampled weights to create the final synthetic signal.

        signal = self._apply_injection(signal, t)
        return signal

    

    # ---------------- controlled injection ----------------
    def _apply_injection(self, signal, t):
        """
        Differently from the base verson of TSMixup, we add an optional injection of a sinusoidal component with controlled amplitude and phase,
        either at a fixed cpp (cycles per patch) or a fixed hz frequency. This allows us to test the effect of tokenisation (cpp) and sampling (hz) on the generated signals, 
        while keeping the rest of the TSMixup process unchanged. The injected component is added to the mixed signal before saving, and its parameters are included in the filename for traceability.
        t is needed for hz-mode injection to compute the sinusoid in seconds (t = np.arange(len(signal)) / fs).
        """
        if not self.inject:
            return signal # If no injection is specified, return the original signal.
        
        # inject has the format {"mode": "cpp" or "hz", "value": float, "amplitude": float, "phase": float}
        spec = self.inject
        amp   = float(spec.get("amplitude", 1.0))
        phase = float(spec.get("phase", 0.0))

        n = np.arange(len(signal)) # Sample indices for the signal, used to compute the sinusoidal component.
        std = signal.std() # Standard deviation of the signal, used to normalize it before adding the sinusoidal component to ensure that the injected component has a controlled effect on the overall signal amplitude.
        signal = signal / std if std > 1e-8 else signal # Normalize the signal to have a standard deviation of 1, unless the std is very small, in which case we leave it unchanged to avoid numerical issues.

        if spec["mode"] == "cpp": # cpp = cycles per patch, which defines the frequency of the sinusoidal component in terms of how many cycles fit into the length of the signal (P).
            # use cpp when you want to inject a component that is related to the length of the signal, which can be relevant for testing tokenisation effects.
            cpp = float(spec["value"])
            period_samples = self.P / cpp
            comp = amp * np.sin(2 * np.pi * n / period_samples + phase) # Compute the sinusoidal component based on the specified cpp, amplitude, and phase.
        elif spec["mode"] == "hz": # hz = cycles per second, which defines the frequency of the sinusoidal component in terms of hertz, independent of the signal length, and is relevant for testing sampling effects.
            # use hz when you want to inject a component at a specific frequency regardless of the signal length
            f_hz = float(spec["value"])
            comp = amp * np.sin(2 * np.pi * f_hz * t + phase) # Compute the sinusoidal component based on the specified frequency in hertz, amplitude, and phase. Note that t is in seconds, so this is fs-dependent.
        else:
            raise ValueError(f"Unknown inject mode: {spec['mode']}")
        return signal + comp
    


    # ---------------- deterministic naming ----------------
    def _base_path(self) -> Path:
        t_lengths = ",".join(str(t) for t in self.t_lengths)
        return self.output_dir / (
            f"TSMixup_K{self.K}_alpha{self.alpha}"
            f"_lmin{self.l_min}_lmax{self.l_max}"
            f"_fs{self.fs}"
            f"_tl{t_lengths}"
            f"{self._build_tag()}.txt"
        )

    def _build_tag(self):
        if not self.inject:
            return ""
        spec = self.inject
        parts = []
        if "value" in spec:
            if spec["mode"] == "cpp":
                parts.append(f"cpp{spec['value']}")
            elif spec["mode"] == "hz":
                parts.append(f"hz{spec['value']}")
        if "amplitude" in spec:
            parts.append(f"amp{spec.get('amplitude', 1.0)}")
        if "phase" in spec:
            parts.append(f"ph{spec.get('phase', 0.0)}")
        return "__" + "_".join(parts) if parts else ""

    # ---------------- IO ----------------
    def save(self, signal):
        path = self.path().with_suffix(".npy")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, signal)

    def plot(self, signal):
        path = self.path().with_suffix(".png")
        plt.figure(figsize=(12, 4))
        plt.plot(signal, linewidth=0.8)
        plt.title("Generated TSMixup Signal")
        plt.grid()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def path(self) -> str:
        if self.name:
            return self.output_dir / f"{self.name}"
        else: 
            return self._base_path()
        

def retrieveDataFromPath(name:str) -> dict:
    # return a json/dict of the parameters and injection configuration encoded in the filename, 
    # for reference, from #TSMixup_K3_alpha1.5_lmin512_lmax512_fs256_tl256,384,512__cpp6.5_amp0.3_ph1.5707963
    
    parameters = name.split("__")[0].split("_") # = ['TSMixup', 'K3', 'alpha1.5', 'lmin512', 'lmax512', 'fs256', 'tl256,384,512'], 
    inject_spec = name.split("__")[1].split("_") if "__" in name else [] # = ['cpp6.5', 'amp0.3', 'ph1.5707963']
    
    K = int(parameters[1][1:]) # 3
    alpha = float(parameters[2][5:]) # 1.5
    l_min = int(parameters[3][4:]) # 512
    l_max = int(parameters[4][4:]) # 512
    fs = float(parameters[5][2:]) # 256
    t_lengths = [int(t) for t in parameters[6][2:].split(",")] # [256, 384, 512]

    inject = {}
    for spec in inject_spec:
        if spec.startswith("cpp"):
            inject["mode"] = "cpp"
            inject["value"] = float(spec[3:]) # 6.5
        elif spec.startswith("hz"):
            inject["mode"] = "hz"
            inject["value"] = float(spec[2:]) # 6.5
        elif spec.startswith("amp"):
            inject["amplitude"] = float(spec[3:]) # 0.3
        elif spec.startswith("ph"):
            inject["phase"] = float(spec[2:]) # 1.5707963
    
    return {
        "generator": "TSMixup",
        "K": K,
        "alpha": alpha,
        "l_min": l_min,
        "l_max": l_max,
        "fs": fs,
        "t_lengths": t_lengths,
        "inject": inject
    }