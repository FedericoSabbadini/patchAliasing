"""
kernelsynth_generator.py — KernelSynth GP generation as a self-contained class.

inject: list of sinusoidal components, each a dict with:
    - freq_hz   : float  — frequency in Hz
    - amplitude : float  — absolute amplitude
    - phase     : float  — phase offset in radians (default 0.0)

Example:
    inject = [
        {"freq_hz": 64.0,  "amplitude": 0.05, "phase": 0.0},
        {"freq_hz": 120.0, "amplitude": 0.02, "phase": 1.5707963},
    ]

cpp is a derived read-only quantity:  cpp = freq_hz * P / fs

Amplitudes are absolute (no background normalisation) so that:
    background = signal - sum(tones)   holds bit-exactly.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
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
    """KernelSynth: sample from a composite GP prior, with optional multi-tone injection."""

    def __init__(self,
                 J: int = 5,
                 l_syn: int = 1024,
                 jitter: float = 0.0,
                 fs: float = 1.0,
                 seed: int = 3,
                 output_dir: str = "./signals",
                 inject: Optional[List[dict]] = None,
                 P: int = 16):

        self.J          = J
        self.l_syn      = l_syn
        self.jitter     = jitter
        self.fs         = fs
        self.output_dir = Path(output_dir)
        self.rng        = np.random.default_rng(seed)
        self.P          = P
        self.inject     = self._parse_inject(inject)
        self.kernel_bank  = self._build_kernel_bank()
        self.last_kernels = []

    # ------------------------------------------------------------------ #
    #  Injection                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_inject(inject) -> Optional[List[dict]]:
        """Validate and store inject list. Each component must have freq_hz, amplitude, phase."""
        if inject is None:
            return None
        items = [inject] if isinstance(inject, dict) else list(inject)
        out = []
        for c in items:
            if "freq_hz" not in c:
                raise ValueError(
                    f"inject component missing 'freq_hz': {c}\n"
                    "Each component must be a dict with keys: freq_hz, amplitude, phase."
                )
            out.append({
                "freq_hz":   float(c["freq_hz"]),
                "amplitude": float(c.get("amplitude", 1.0)),
                "phase":     float(c.get("phase", 0.0)),
            })
        return out

    def _apply_injection(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Add sinusoidal tones to x. Amplitudes are absolute so that:
            background = signal - sum(tones)
        holds exactly, which is required for causal SCPA probing.
        """
        if not self.inject:
            return x
        out = x.copy()
        for c in self.inject:
            out += c["amplitude"] * np.sin(2 * np.pi * c["freq_hz"] * t + c["phase"])
        return out

    def cpp_of(self, freq_hz: float) -> float:
        """Cycles per patch for a given frequency in Hz (derived, read-only)."""
        return freq_hz * self.P / self.fs

    # ------------------------------------------------------------------ #
    #  Kernel bank                                                         #
    # ------------------------------------------------------------------ #
    def _build_kernel_bank(self) -> list:
        periodic = [
            (f"Periodic(period={p})",
             (lambda p=p: lambda t, tp: _k_periodic(t, tp, periodicity=p))())
            for p in AWS_PERIODS
        ]
        return [
            *periodic,
            ("Linear",   lambda t, tp: _k_linear(t, tp, s0=0.0)),
            ("Linear",   lambda t, tp: _k_linear(t, tp, s0=1.0)),
            ("Linear",   lambda t, tp: _k_linear(t, tp, s0=10.0)),
            ("RBF",      lambda t, tp: _k_rbf(t, tp, ls=0.1)),
            ("RBF",      lambda t, tp: _k_rbf(t, tp, ls=1.0)),
            ("RBF",      lambda t, tp: _k_rbf(t, tp, ls=10.0)),
            ("RQ",       lambda t, tp: _k_rq(t, tp, alpha=0.1)),
            ("RQ",       lambda t, tp: _k_rq(t, tp, alpha=1.0)),
            ("RQ",       lambda t, tp: _k_rq(t, tp, alpha=10.0)),
            ("White",    lambda t, tp: _k_white(t, tp, noise=0.1)),
            ("White",    lambda t, tp: _k_white(t, tp, noise=1.0)),
            ("Constant", lambda t, tp: _k_constant(t, tp, C=1.0)),
        ]

    # ------------------------------------------------------------------ #
    #  Generation                                                          #
    # ------------------------------------------------------------------ #
    def generate(self) -> np.ndarray:
        while True:
            t = np.arange(self.l_syn) / self.fs
            j = self.rng.integers(1, self.J + 1)
            chosen = [
                self.kernel_bank[i]
                for i in self.rng.choice(len(self.kernel_bank), size=j, replace=True)
            ]

            K_star = chosen[0][1](t, t)
            names  = [chosen[0][0]]
            for name, kfn in chosen[1:]:
                op     = self.rng.choice(["+", "*"])
                K_star = K_star + kfn(t, t) if op == "+" else K_star * kfn(t, t)
                names.append(f"{op} {name}")

            if self.jitter > 0.0:
                K_star += self.jitter * np.eye(self.l_syn)

            try:
                signal = self.rng.multivariate_normal(np.zeros(self.l_syn), K_star)
            except np.linalg.LinAlgError:
                continue

            self.last_kernels = names
            return self._apply_injection(signal, t)

    # ------------------------------------------------------------------ #
    #  Naming — 12-significant-digit precision to preserve round-trip      #
    # ------------------------------------------------------------------ #
    def _build_tag(self) -> str:
        if not self.inject:
            return ""
        parts = [
            f"hz{c['freq_hz']:.12g}_amp{c['amplitude']:.12g}_ph{c['phase']:.12g}"
            for c in self.inject
        ]
        return "__" + "__".join(parts)

    def _base_path(self) -> Path:
        return self.output_dir / (
            f"KernelSynth_J{self.J}_l_syn{self.l_syn}_fs{self.fs:.12g}"
            f"_jitter{self.jitter:.12g}{self._build_tag()}.npy"
        )

    def path(self) -> Path:
        return self._base_path()

    # ------------------------------------------------------------------ #
    #  I/O                                                                 #
    # ------------------------------------------------------------------ #
    def getParameters(self) -> dict:
        inj_out = None
        if self.inject:
            inj_out = [{**c, "cpp": self.cpp_of(c["freq_hz"])} for c in self.inject]
        return {
            "generator": "KernelSynth",
            "J":       self.J,
            "l_syn":   self.l_syn,
            "fs":      self.fs,
            "jitter":  self.jitter,
            "P":       self.P,
            "inject":  inj_out,
        }

    def save(self, signal: np.ndarray):
        path = self.path().with_suffix(".npy")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, signal)

    def plot(self, signal: np.ndarray):
        path = self.path().with_suffix(".png")
        t    = np.arange(len(signal)) / self.fs
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        axes[0].plot(t, signal, linewidth=0.8, color="crimson")
        axes[0].set_title("Generated KernelSynth Signal")
        axes[0].set_xlabel("Time (s)"); axes[0].grid()

        freqs = np.fft.rfftfreq(len(signal), d=1.0 / self.fs)
        mag   = np.abs(np.fft.rfft(signal - signal.mean()))
        axes[1].plot(freqs, mag, linewidth=0.8, color="steelblue")
        axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("|FFT|")
        axes[1].set_title("Spectrum")
        if self.inject:
            for c in self.inject:
                axes[1].axvline(
                    c["freq_hz"], color="red", ls="--", alpha=0.7,
                    label=f"{c['freq_hz']} Hz  (cpp={self.cpp_of(c['freq_hz']):.4f})"
                )
            axes[1].legend()
        axes[1].grid()

        plt.tight_layout()
        plt.savefig(path)
        plt.close()
