"""
signalModifier.py — Injects sinusoidal components into existing time-series signals.

All injection is in Hz (absolute frequency).
cpp is a derived read-only quantity:  cpp = freq_hz * P / fs

Amplitudes are absolute: background = signal - sum(tones) holds bit-exactly.
"""
import numpy as np
from typing import List


class SignalModifier:
    def __init__(self, fs: float = 1.0, P: int = 16):
        """
        Parameters
        ----------
        fs : float  Sampling frequency in Hz.
        P  : int    Patch size (used only for cpp reporting).
        """
        if fs <= 0:
            raise ValueError(f"fs must be positive, got {fs}")
        if P <= 0:
            raise ValueError(f"P must be positive, got {P}")
        self.fs = float(fs)
        self.P  = int(P)

    # ------------------------------------------------------------------ #
    #  Single-component injection                                          #
    # ------------------------------------------------------------------ #
    def addComponent(self,
                     signal: np.ndarray,
                     freq_hz: float,
                     amplitude: float,
                     phase: float = 0.0) -> np.ndarray:
        """
        Inject a single sinusoidal tone into signal.

        Returns a copy with the tone added.  Original signal is not modified.
        """
        t = np.arange(len(signal)) / self.fs
        return signal.copy() + amplitude * np.sin(2 * np.pi * freq_hz * t + phase)

    # ------------------------------------------------------------------ #
    #  Multi-component injection                                           #
    # ------------------------------------------------------------------ #
    def addComponents(self,
                      signal: np.ndarray,
                      components: List[dict]) -> np.ndarray:
        """
        Inject multiple sinusoidal tones in a single call.

        Each component dict must have:
            - freq_hz   : float
            - amplitude : float
            - phase     : float  (optional, default 0.0)

        Returns a copy with all tones summed in.  Original signal not modified.
        """
        t   = np.arange(len(signal)) / self.fs
        out = signal.copy()
        for c in components:
            out += float(c["amplitude"]) * np.sin(
                2 * np.pi * float(c["freq_hz"]) * t + float(c.get("phase", 0.0))
            )
        return out

    # ------------------------------------------------------------------ #
    #  Background recovery                                                 #
    # ------------------------------------------------------------------ #
    def recover_background(self,
                           signal: np.ndarray,
                           components: List[dict]) -> np.ndarray:
        """
        Exactly subtract all injected tones to recover the original background.
        Requires the same components list used during injection.
        """
        t   = np.arange(len(signal)) / self.fs
        out = signal.copy()
        for c in components:
            out -= float(c["amplitude"]) * np.sin(
                2 * np.pi * float(c["freq_hz"]) * t + float(c.get("phase", 0.0))
            )
        return out

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def cpp_of(self, freq_hz: float) -> float:
        """Cycles per patch for a given Hz frequency (derived, read-only)."""
        return freq_hz * self.P / self.fs

    def is_integer_cpp(self, freq_hz: float, tol: float = 1e-9) -> bool:
        """True iff freq_hz hits an exact integer-cpp blind spot (within tol)."""
        cpp = self.cpp_of(freq_hz)
        return abs(cpp - round(cpp)) < tol
