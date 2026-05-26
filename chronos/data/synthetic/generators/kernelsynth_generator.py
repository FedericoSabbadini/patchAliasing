"""
kernelsynth_generator.py — KernelSynth GP generation as a self-contained class.
This module defines the KernelSynthGenerator class, which generates synthetic time series data by sampling from a composite Gaussian Process (GP) prior defined by a random combination of kernels.
The generator constructs a kernel by randomly selecting and combining base kernels (e.g., RBF, Linear, Periodic) with random parameters and operations (addition or multiplication). 
The generated signals are saved as .npy files along with corresponding plots for visualization.

N.B. KernelSynth is sampling-rate dependent: the generated signal's properties (e.g., mean, std) and the time axis are influenced by the specified sampling frequency (fs). 
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


# ---------------- Kernels ----------------
# A kernel is a function k(t, tp) that takes two time vectors and returns the covariance matrix between them (corresponding to the dot product of their features). 
# We will define several base kernels and then combine them randomly to create a composite kernel for sampling.

def _k_constant(t, tp, C=1.0):
    """Constant kernel"""
    # From t=[0, 1, 2] and tp=[0, 1], we want a matrix of shape (3, 2) filled with C=1.0
    return np.full((len(t), len(tp)), C, dtype=float)

def _k_white(t, tp, noise=1.0):
    """White noise kernel"""
    # From t=[0, 1, 2] and tp=[0, 1], we want a matrix of shape (3, 2) where the diagonal (t==tp) is noise=1.0 and the rest is 0.0
    T, Tp = np.meshgrid(t, tp, indexing="ij")
    return noise * (T == Tp).astype(float)

def _k_linear(t, tp, s0=1.0):
    """Linear kernel"""
    T, Tp = np.meshgrid(t, tp, indexing="ij")
    return s0 ** 2 + T * Tp

def _k_rbf(t, tp, ls=1.0):
    """RBF kernel"""
    T, Tp = np.meshgrid(t, tp, indexing="ij")
    return np.exp(-0.5 * ((T - Tp) / ls) ** 2)

def _k_rq(t, tp, alpha=1.0, ls=1.0):
    """RQ kernel"""
    T, Tp = np.meshgrid(t, tp, indexing="ij")
    return (1.0 + (T - Tp) ** 2 / (2.0 * alpha * ls ** 2)) ** (-alpha)

def _k_periodic(t, tp, periodicity=1.0, ls=1.0):
    """Periodic kernel"""
    T, Tp = np.meshgrid(t, tp, indexing="ij")
    return np.exp(-2.0 * np.sin(np.pi * np.abs(T - Tp) / periodicity) ** 2 / ls ** 2)


# AWS_PERIODS are common periods observed in real-world time series (e.g., daily, weekly, monthly, yearly).
# We include these as potential periodicities for the periodic kernel to make the generated signals more realistic and relevant for time series analysis.
AWS_PERIODS = [
    24, 48, 96, 24*7, 48*7, 96*7, 7, 14, 30, 60, 365, 365*2,
    4, 26, 52, 4, 6, 12, 4, 4*10, 10
]


class KernelSynthGenerator:
    """KernelSynth: sample from a composite GP prior."""

    def __init__(self, 
                 J=5, 
                 l_syn=1024, 
                 jitter=0.0, 
                 fs=1.0,

                 seed=3, 
                 output_dir="./signals",
                 
                 inject=None, 
                 P=16,

                name=None
                 ):


        self.J = J # number of kernels to combine
        # J is typically set to a small integer (e.g., 3-10) to create a rich but not overly complex composite kernel. Too many kernels can lead to overfitting and numerical instability when sampling.
        self.l_syn = l_syn # length of the generated signal (number of samples)
        # l_syn is typically set to a power of 2 (e.g., 1024, 2048) to facilitate efficient sampling from the GP using methods like the Cholesky decomposition or FFT-based approaches.
        self.jitter = jitter # jitter to add to the kernel for numerical stability (if > 0). It adds noise to the diagonal of the kernel matrix, which can help ensure that it is positive definite and can be sampled from without numerical issues.
        # jitter is typically set to a small value (e.g., 1e-6) to prevent numerical instability when sampling from the GP, especially if the kernel matrix is close to singular.
        self.output_dir = Path(output_dir)
        self.rng = np.random.default_rng(seed)
        self.fs = fs # sampling frequency (Hz) - this will affect the time axis and the properties of the generated signal, making it more realistic for time series analysis.
        # fs is typically set to a value that reflects the desired temporal resolution of the generated signal (e.g., 1.0 for 1 sample per second, 24.0 for hourly data, 365.0 for daily data) to ensure that the generated signals have realistic time scales and properties.
        self.kernel_bank = self._build_kernel_bank()
        self.last_kernels = [] # store the last chosen kernels for reference

        self.inject = inject # optional injection configuration (None or dict with mode, value, amplitude, phase)
        self.P = P # patch size, needed for cpp-mode injection (if inject is not None and inject.mode == "cpp")

        self.name = name # optional name for the generated signal (if None, a deterministic name will be built based on the parameters and injection configuration)

    def _build_kernel_bank(self) -> list[tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]]:
        """Build a bank of base kernels with different parameters."""
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


    def generate(self) -> np.ndarray:
        while True:
            # time axis in seconds
            t = np.arange(self.l_syn) / self.fs # from 0 to (l_syn-1)/fs seconds
            j = self.rng.integers(1, self.J + 1) # number of kernels to combine (at least 1, at most J)

            chosen = [
                self.kernel_bank[i]
                for i in self.rng.choice(len(self.kernel_bank), size=j, replace=True)
            ] # randomly choose j kernels from the kernel bank (with replacement)

            K_star = chosen[0][1](t, t) # start with the first chosen kernel's covariance matrix (shape: (l_syn, l_syn))
            names = [chosen[0][0]] # store the name of the first chosen kernel for reference

            for name, kfn in chosen[1:]: # combine the remaining chosen kernels with random operations (addition or multiplication)
                op = self.rng.choice(["+", "*"])
                K_star = K_star + kfn(t, t) if op == "+" else K_star * kfn(t, t) # combine the kernels using the chosen operation (shape: (l_syn, l_syn))
                names.append(f"{op} {name}")

            if self.jitter > 0.0: # add jitter to the diagonal for numerical stability
                K_star = K_star + self.jitter * np.eye(self.l_syn) # add jitter to the diagonal

            try:
                signal = self.rng.multivariate_normal(np.zeros(self.l_syn), K_star) # sample from the GP with mean 0 and covariance K_star (shape: (l_syn,))
            except np.linalg.LinAlgError:
                continue

            self.last_kernels = names # store the names of the kernels used to generate the signal for reference
            signal = self._apply_injection(signal, t) # apply optional injection of a deterministic periodic component and get the corresponding tag for naming
            
            return signal

    

    # ---------------- controlled injection ----------------
    def _apply_injection(self, x, t):
        """Add a deterministic periodic component at a target cpp or Hz. Returns (x, tag)."""
        if not self.inject:
            return x, ""
        spec = self.inject
        amp   = float(spec.get("amplitude", 1.0))   # amplitude relative to unit-std background
        phase = float(spec.get("phase", 0.0))
        n = np.arange(len(x))

        # normalise GP background to unit std so the injection ratio is meaningful
        std = x.std()
        x = x / std if std > 1e-8 else x

        if spec["mode"] == "cpp":
            cpp = float(spec["value"])
            period_samples = self.P / cpp                  # cpp = P / period_samples
            comp = amp * np.sin(2 * np.pi * n / period_samples + phase)
        elif spec["mode"] == "hz":
            f_hz = float(spec["value"])
            comp = amp * np.sin(2 * np.pi * f_hz * t + phase)   # t in seconds → fs-dependent
        else:
            raise ValueError(f"Unknown inject mode: {spec.get('mode')}")

        return x + comp



    # ---------------- deterministic naming ----------------
    def _build_tag(self) -> str:
        if not self.inject:
            return ""
        spec = self.inject
        parts = []
        if "value" in spec:
            if spec["mode"] == "cpp":
                parts.append(f"cpp{float(spec['value']):.4f}".rstrip("0").rstrip("."))
            elif spec["mode"] == "hz":
                parts.append(f"hz{float(spec['value']):.4f}".rstrip("0").rstrip("."))
        if "amplitude" in spec:
            parts.append(f"amp{float(spec.get('amplitude', 1.0)):.4f}".rstrip("0").rstrip("."))
        if "phase" in spec:
            parts.append(f"ph{float(spec.get('phase', 0.0)):.4f}".rstrip("0").rstrip("."))
        return "__" + "_".join(parts) if parts else ""

    def _base_path(self) -> Path:
        return self.output_dir / (
            f"KernelSynth_J{self.J}_l_syn{self.l_syn}_fs{self.fs}_jitter{self.jitter}"
            f"{self._build_tag()}.txt"
        )
    

    # ---------------- I/O ----------------
    def save(self, signal):
        path = self.path().with_suffix(".npy")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, signal)

    def plot(self, signal):

        path = self.path().with_suffix(".png")
        plt.figure(figsize=(12, 4))
        plt.plot(signal, linewidth=0.8, color="crimson")
        plt.title("Generated KernelSynth Signal")
        plt.grid()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


    def path(self) -> Path:
        if self.name:
            return self.output_dir / f"{self.name}"
        else:
            return self._base_path()