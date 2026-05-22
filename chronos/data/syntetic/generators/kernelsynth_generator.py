"""
kernelsynth_generator.py — KernelSynth GP generation as a self-contained class.
This module defines the KernelSynthGenerator class, which generates synthetic time series data by sampling from a composite Gaussian Process (GP) prior defined by a random combination of kernels.
The generator constructs a kernel by randomly selecting and combining base kernels (e.g., RBF, Linear, Periodic) with random parameters and operations (addition or multiplication). 
The generated signals are saved as .npy files along with corresponding plots for visualization.

N.B. KernelSynth generated time-domain signals, so the frequency fs does affect the generation process by defining the time axis. The time axis is defined as t = np.arange(l_syn) / fs, which allows for consistent naming and potential future use of fs in downstream tasks.
This means that changing fs will change the time axis and thus affect the generated signal, making it more realistic for different sampling rates.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


# ---------------- Kernels ----------------

def _k_constant(t, tp, C=1.0):
    return np.full((len(t), len(tp)), C, dtype=float)

def _k_white(t, tp, noise=1.0):
    T, Tp = np.meshgrid(t, tp, indexing="ij")
    return noise * (T == Tp).astype(float)

def _k_linear(t, tp, s0=1.0):
    T, Tp = np.meshgrid(t, tp, indexing="ij")
    return s0 ** 2 + T * Tp

def _k_rbf(t, tp, ls=1.0):
    T, Tp = np.meshgrid(t, tp, indexing="ij")
    return np.exp(-0.5 * ((T - Tp) / ls) ** 2)

def _k_rq(t, tp, alpha=1.0, ls=1.0):
    T, Tp = np.meshgrid(t, tp, indexing="ij")
    return (1.0 + (T - Tp) ** 2 / (2.0 * alpha * ls ** 2)) ** (-alpha)

def _k_periodic(t, tp, periodicity=1.0, ls=1.0):
    T, Tp = np.meshgrid(t, tp, indexing="ij")
    return np.exp(-2.0 * np.sin(np.pi * np.abs(T - Tp) / periodicity) ** 2 / ls ** 2)


AWS_PERIODS = [
    24, 48, 96, 24*7, 48*7, 96*7, 7, 14, 30, 60, 365, 365*2,
    4, 26, 52, 4, 6, 12, 4, 4*10, 10
]


class KernelSynthGenerator:
    """KernelSynth: sample from a composite GP prior."""

    def __init__(self, J=5, l_syn=1024, jitter=0.0, fs=1.0,
                 seed=3, output_dir="./signals"):

        self.J, self.l_syn, self.jitter = J, l_syn, jitter
        self.output_dir = Path(output_dir)
        self.rng = np.random.default_rng(seed)
        self.fs = fs

        self.kernel_bank = self._build_kernel_bank()
        self.last_kernels = []

    def _build_kernel_bank(self):
        periodic = [
            (
                f"Periodic(period={p})",
                (lambda p=p: lambda t, tp: _k_periodic(
                    t, tp,
                    periodicity=p
                ))()
            )
            for p in AWS_PERIODS
        ]

        return [
            *periodic,
            ("Linear", lambda t, tp: _k_linear(t, tp, s0=0.0)),
            ("Linear", lambda t, tp: _k_linear(t, tp, s0=1.0)),
            ("Linear", lambda t, tp: _k_linear(t, tp, s0=10.0)),
            ("RBF", lambda t, tp: _k_rbf(t, tp, ls=0.1)),
            ("RBF", lambda t, tp: _k_rbf(t, tp, ls=1.0)),
            ("RBF", lambda t, tp: _k_rbf(t, tp, ls=10.0)),
            ("RQ", lambda t, tp: _k_rq(t, tp, alpha=0.1)),
            ("RQ", lambda t, tp: _k_rq(t, tp, alpha=1.0)),
            ("RQ", lambda t, tp: _k_rq(t, tp, alpha=10.0)),
            ("White", lambda t, tp: _k_white(t, tp, noise=0.1)),
            ("White", lambda t, tp: _k_white(t, tp, noise=1.0)),
            ("Constant", lambda t, tp: _k_constant(t, tp, C=1.0)),
        ]

    def generate(self):
        while True:
            # time axis in seconds
            t = np.arange(self.l_syn) / self.fs

            j = self.rng.integers(1, self.J + 1)

            chosen = [
                self.kernel_bank[i]
                for i in self.rng.choice(len(self.kernel_bank), size=j, replace=True)
            ]

            K_star = chosen[0][1](t, t)
            names = [chosen[0][0]]

            for name, kfn in chosen[1:]:
                op = self.rng.choice(["+", "*"])
                K_star = K_star + kfn(t, t) if op == "+" else K_star * kfn(t, t)
                names.append(f"{op} {name}")

            if self.jitter > 0.0:
                K_star = K_star + self.jitter * np.eye(self.l_syn)

            try:
                x = self.rng.multivariate_normal(np.zeros(self.l_syn), K_star)
            except np.linalg.LinAlgError:
                continue

            self.last_kernels = names
            return x, t

    def save(self, signal) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        path = self.output_dir / (
            f"syntheticKernelSynth_J{self.J}_l{self.l_syn}_fs{self.fs}.npy"
        )

        np.save(path, signal)
        self._plot(signal, path)
        return path

    def _plot(self, x, path):
        plt.figure(figsize=(12, 4))
        plt.plot(x, linewidth=0.8, color="crimson")
        plt.title("Generated KernelSynth Signal")
        plt.xlabel("Samples")
        plt.ylabel("Value")
        plt.grid()
        plt.tight_layout()
        plt.savefig(path.with_suffix(".png"))
        plt.close()