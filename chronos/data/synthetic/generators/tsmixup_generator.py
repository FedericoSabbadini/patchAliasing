"""
tsmixup_generator.py — TSMixup augmentation as a self-contained class.

inject: list of sinusoidal components, each a dict with:
    - freq_hz   : float  — frequency in Hz
    - amplitude : float  — absolute amplitude
    - phase     : float  — phase offset in radians (default 0.0)

Example:
    inject = [
        {"freq_hz": 120.0, "amplitude": 0.03, "phase": 1.5707963},
        {"freq_hz":  40.0, "amplitude": 0.01, "phase": 0.0},
    ]

cpp is a derived read-only quantity:  cpp = freq_hz * P / fs

Amplitudes are absolute (no background normalisation) so that:
    background = signal - sum(tones)   holds bit-exactly.

N.B. TSMixup background generation is sampling-rate agnostic; injection is fs-dependent.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make synthetic/ importable
from tones import tone_on_grid, cpp as _cpp   # canonical injection convention + cpp (single source of truth)


class TSMixupGenerator:
    """Chronos-style TSMixup: mix k mean-scaled subsequences, with optional multi-tone injection."""

    def __init__(self,
                 K: int = 3,
                 alpha: float = 1.5,
                 l_min: int = 128,
                 l_max: int = 2048,
                 t_lengths=None,
                 data_mode: str = "synthetic",
                 data_dir=None,
                 seed: int = 3,
                 output_dir: str = "./signals",
                 inject: Optional[List[dict]] = None,
                 P: int = 16,
                 fs: float = 1.0):

        self.K          = K
        self.alpha      = alpha
        self.l_min      = l_min
        self.l_max      = l_max
        self.t_lengths  = t_lengths or [500, 600, 700, 400, 550]
        self.data_mode  = data_mode
        self.data_dir   = data_dir
        self.output_dir = Path(output_dir)
        self.rng        = np.random.default_rng(seed)
        self.P          = P
        self.fs         = fs
        self.inject     = self._parse_inject(inject)
        self.datasets   = self._build_datasets()

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

    def _apply_injection(self, signal: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Add sinusoidal tones to signal. Amplitudes are absolute so that:
            background = signal - sum(tones)
        holds exactly, which is required for causal SCPA probing.
        """
        if not self.inject:
            return signal
        out = signal.copy()
        for c in self.inject:
            out += tone_on_grid(t, c["freq_hz"], c["amplitude"], c["phase"])  # canonical tone (tones.py)
        return out

    def cpp_of(self, freq_hz: float) -> float:
        """Cycles per patch for a given frequency in Hz (derived, read-only)."""
        return _cpp(freq_hz, self.P, self.fs)

    # ------------------------------------------------------------------ #
    #  Dataset construction                                                #
    # ------------------------------------------------------------------ #
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

    def _sample_subsequence(self, dataset: np.ndarray, l: int) -> np.ndarray:
        T = dataset.shape[0]
        if T >= l:
            start = self.rng.integers(0, T - l + 1)
            return dataset[start:start + l].copy()
        repeats = (l // T) + 1
        return np.tile(dataset, repeats)[:l].copy()

    # ------------------------------------------------------------------ #
    #  Generation                                                          #
    # ------------------------------------------------------------------ #
    def generate(self) -> np.ndarray:
        k = self.rng.integers(1, self.K + 1)
        l = self.rng.integers(self.l_min, self.l_max + 1)
        t = np.arange(l) / self.fs

        scaled = []
        for _ in range(k):
            n = self.rng.integers(0, len(self.datasets))
            x = self._sample_subsequence(self.datasets[n], l)
            x = x / max(float(np.mean(np.abs(x))), 1e-8)
            scaled.append(x)

        lambdas = self.rng.dirichlet([self.alpha] * k)
        signal  = sum(lam * s for lam, s in zip(lambdas, scaled))
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
        tl = ",".join(str(t) for t in self.t_lengths)
        return self.output_dir / (
            f"TSMixup_K{self.K}_alpha{self.alpha:.12g}"
            f"_lmin{self.l_min}_lmax{self.l_max}"
            f"_fs{self.fs:.12g}_tl{tl}{self._build_tag()}.npy"
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
            "generator": "TSMixup",
            "K":         self.K,
            "alpha":     self.alpha,
            "l_min":     self.l_min,
            "l_max":     self.l_max,
            "t_lengths": self.t_lengths,
            "fs":        self.fs,
            "P":         self.P,
            "inject":    inj_out,
        }

    def save(self, signal: np.ndarray):
        path = self.path().with_suffix(".npy")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, signal)

    def plot(self, signal: np.ndarray):
        path = self.path().with_suffix(".png")
        t    = np.arange(len(signal)) / self.fs
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        axes[0].plot(t, signal, linewidth=0.8)
        axes[0].set_title("Generated TSMixup Signal")
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


# ------------------------------------------------------------------ #
#  Standalone utility: parse parameters from filename                  #
# ------------------------------------------------------------------ #
def retrieveDataFromPath(name: str) -> dict:
    """
    Parse a TSMixup filename back to parameters.
    Example:
        TSMixup_K3_alpha1.5_lmin512_lmax512_fs256_tl256,384,512__hz120_amp0.03_ph1.5707963
    """
    # Strip extension if present
    name = name.rsplit(".", 1)[0]

    base, *tag_parts = name.split("__")
    parameters = base.split("_")

    K         = int(parameters[1][1:])
    alpha     = float(parameters[2][5:])
    l_min     = int(parameters[3][4:])
    l_max     = int(parameters[4][4:])
    # parameters[5] = "fs256" — could parse if needed
    t_lengths = [int(x) for x in parameters[6][2:].split(",")]

    inject = []
    for tag in tag_parts:
        comp = {}
        for item in tag.split("_"):
            if item.startswith("hz"):
                comp["freq_hz"]   = float(item[2:])
            elif item.startswith("amp"):
                comp["amplitude"] = float(item[3:])
            elif item.startswith("ph"):
                comp["phase"]     = float(item[2:])
        if comp:
            inject.append(comp)

    return {
        "generator": "TSMixup",
        "K":         K,
        "alpha":     alpha,
        "l_min":     l_min,
        "l_max":     l_max,
        "t_lengths": t_lengths,
        "inject":    inject or None,
    }
