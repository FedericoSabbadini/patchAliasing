"""
tones.py — canonical sinusoidal-tone helpers shared by the synthetic signal generators
(TSMixup / KernelSynth injection) AND the model evaluation (frequency sweep / inspector).

Single source of truth for two things:

1. The injection convention (must stay bit-identical to what the generators emit):
       tone(t) = amplitude * sin(2*pi*freq_hz*t + phase),   t = arange(n) / fs

2. The cycles-per-patch aliasing coordinate:
       cpp = freq_hz * P / fs        # cycles of the tone inside one patch of P samples

Why cpp is a useful axis: it expresses a tone's frequency in units of one patch, so
patch-relative structure lines up at the same integer ticks whatever the model's P.

NOTE (empirical): the original hypothesis — that integer cpp (freq_hz = k*fs/P) produces
forecast-recovery NULLS because a whole number of cycles integrates to ~0 inside a patch —
was tested and REFUTED. Integer-cpp tones are in fact among the BEST recovered (see
testing/README.md, H1). What is real and geometric is a token-level collapse on the STRIDE
grid freq_hz = c*fs/S (H3), and that collapse does NOT null the forecast. cpp remains a
convenient patch-relative coordinate; it is not, on its own, evidence of an aliasing null.
"""
from __future__ import annotations

import numpy as np


def tone_on_grid(t: np.ndarray, freq_hz: float, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """A pure sinusoid sampled on an existing time grid `t` (seconds)."""
    return amplitude * np.sin(2 * np.pi * freq_hz * t + phase)


def make_tone(freq_hz: float, fs: float, n: int, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """A pure sinusoid of `n` samples at sampling rate `fs` (Hz)."""
    return tone_on_grid(np.arange(n) / fs, freq_hz, amplitude, phase)


def apply_injection(signal: np.ndarray, inject, fs: float) -> np.ndarray:
    """Add a list of {freq_hz, amplitude, phase} tones onto `signal` (absolute amplitudes).

    Mirrors the generators' `_apply_injection` so background = signal - sum(tones) holds.
    """
    if not inject:
        return signal
    t = np.arange(signal.shape[0]) / fs
    out = signal.copy()
    for c in inject:
        out += tone_on_grid(t, c["freq_hz"], c.get("amplitude", 1.0), c.get("phase", 0.0))
    return out


def cpp(freq_hz: float, P: int, fs: float) -> float:
    """Cycles per patch: how many tone cycles fit in one patch of P samples."""
    return freq_hz * P / fs


def cpp_to_freq(cpp_value: float, P: int, fs: float) -> float:
    """Inverse of cpp(): the frequency [Hz] that gives `cpp_value` cycles per patch."""
    return cpp_value * fs / P
