"""
tsmixup_generator.py — TSMixup augmentation as a self-contained class.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt


class TSMixupGenerator:
    """Chronos-style TSMixup: mix k mean-scaled subsequences with Dirichlet weights."""

    def __init__(self, K=3, alpha=1.5, l_min=128, l_max=2048,
                 data_mode="synthetic", t_lengths=None, data_dir="./input_datasets",
                 seed=3, output_dir="./signals"):
        self.K, self.alpha = K, alpha
        self.l_min, self.l_max = l_min, l_max
        self.data_mode = data_mode
        self.t_lengths = t_lengths or [500, 600, 700, 400, 550]
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.rng = np.random.default_rng(seed)
        self.datasets = self._build_datasets()

    def _build_datasets(self) -> List[np.ndarray]:
        if self.data_mode == "synthetic":
            return [np.sin(np.linspace(0, 5 * np.pi * (i + 1), T))
                    for i, T in enumerate(self.t_lengths)]
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
        return np.tile(dataset, (repeats,) + (1,) * (dataset.ndim - 1))[:l].copy()

    def generate(self) -> np.ndarray:
        k = self.rng.integers(1, self.K + 1)
        l = self.rng.integers(self.l_min, self.l_max + 1)
        scaled = []
        for _ in range(k):
            n = self.rng.integers(0, len(self.datasets))
            x = self._sample_subsequence(self.datasets[n], l)
            scaled.append(x / max(float(np.mean(np.abs(x))), 1e-8))
        lambdas = self.rng.dirichlet([self.alpha] * k)
        return sum(lam * s for lam, s in zip(lambdas, scaled))

    def save(self, signal) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fname = (f"syntheticTSMixup_K{self.K}_alpha{self.alpha}"
                 f"_lmin{self.l_min}_lmax{self.l_max}.npy")
        path = self.output_dir / fname
        np.save(path, signal)
        self.print(path, signal)
        return path
    
    def print(self, path, signal):
        plt.figure(figsize=(12, 4))
        plt.plot(signal)
        plt.title("Generated TSMixup Signal")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.grid()
        plt.savefig(path.with_suffix(".png"))

