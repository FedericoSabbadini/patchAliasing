"""
tsmixup_generator.py — TSMixup augmentation as a self-contained class.
This module defines the TSMixupGenerator class, which generates synthetic time series data by mixing mean-scaled subsequences from a dataset with Dirichlet weights. 
The generator can work with either synthetic sine wave data or real data loaded from .npy files. 
The generated signals are saved as .npy files along with corresponding plots for visualization.

N.B. TSMixup generated index-domain signals, so the frequency fs does not affect the generation process but is included in the filename for clarity.
The time axis is simply the sample index divided by fs, which allows for consistent naming and potential future use of fs in downstream tasks without affecting the generation process.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt


class TSMixupGenerator:
    """Chronos-style TSMixup: mix k mean-scaled subsequences with Dirichlet weights."""

    def __init__(self, K=3, alpha=1.5, l_min=128, l_max=2048,
                 data_mode="synthetic", t_lengths=None,
                 data_dir="./input_datasets", fs=1.0,
                 seed=3, output_dir="./signals"
                 ):

        self.K, self.alpha = K, alpha
        self.l_min, self.l_max = l_min, l_max
        self.data_mode = data_mode
        self.t_lengths = t_lengths or [500, 600, 700, 400, 550]
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.rng = np.random.default_rng(seed)
        self.fs = fs

        self.datasets = self._build_datasets()

    def _build_datasets(self) -> List[np.ndarray]:
        if self.data_mode == "synthetic":
            return [
                np.sin(np.linspace(0, 5 * np.pi * (i + 1), T))
                for i, T in enumerate(self.t_lengths)
            ]

        if self.data_mode == "files":
            files = sorted(Path(self.data_dir).glob("*.npy"))
            if not files:
                raise FileNotFoundError(f"No .npy files in {self.data_dir}")
            return [np.load(f).astype(float) for f in files]

        raise ValueError(f"Unknown data_mode: {self.data_mode}")

    def _sample_subsequence(self, dataset, l):
        T = dataset.shape[0]

        if T >= l:
            start = self.rng.integers(0, T - l + 1)
            return dataset[start:start + l].copy()

        repeats = (l // T) + 1
        return np.tile(dataset, repeats)[:l].copy()

    def generate(self):
        k = self.rng.integers(1, self.K + 1)
        l = self.rng.integers(self.l_min, self.l_max + 1)

        # time axis in seconds
        t = np.arange(l) / self.fs

        scaled = []
        for _ in range(k):
            n = self.rng.integers(0, len(self.datasets))
            x = self._sample_subsequence(self.datasets[n], l)
            x = x / max(float(np.mean(np.abs(x))), 1e-8)
            scaled.append(x)

        lambdas = self.rng.dirichlet([self.alpha] * k)
        signal = sum(lam * s for lam, s in zip(lambdas, scaled))

        return signal, t

    def save(self, signal) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        path = self.output_dir / (
            f"syntheticTSMixup_K{self.K}_alpha{self.alpha}"
            f"_lmin{self.l_min}_lmax{self.l_max}_fs{self.fs}.npy"
        )

        np.save(path, signal)
        self._plot(signal, path)
        return path

    def _plot(self, signal, path):
        plt.figure(figsize=(12, 4))
        plt.plot(signal)
        plt.title("Generated TSMixup Signal")
        plt.xlabel("Samples")
        plt.ylabel("Value")
        plt.grid()
        plt.savefig(path.with_suffix(".png"))
        plt.close()