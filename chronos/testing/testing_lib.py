"""
testing_lib.py — model loader and shared helpers for the testing workflow.

(16,16) loads the official amazon/chronos-bolt-tiny; any other (P,S) loads the
retrained variant from the local checkpoint or Hugging Face.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ALL_MODELS: list[tuple[int, int]] = [
    (16, 16), (16, 12), (16, 8), (8, 8), (24, 24),
]

OFFICIAL_MODEL = "amazon/chronos-bolt-tiny"
SWEEP_REPO = "federicosabbadini/chronos-bolt-patch-sweep"
_SUBFOLDER = {
    (16, 12): "p16-s12-seed42", (16, 8): "p16-s8-seed42",
    (8, 8): "p8-s8-seed42", (24, 24): "p24-s24-seed42",
}
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "models" / "weights"


def model_tag(P: int, S: int) -> str:
    """Short, filesystem-safe id for a model (used as the per-model output folder name)."""
    return f"p{P}-s{S}"


def _resolve_local_ckpt(P: int, S: int) -> Path | None:
    """Latest local checkpoint dir for (P, S), or None if there is no local copy."""
    d = _WEIGHTS_DIR / f"p{P}-s{S}-seed42"
    if not d.exists():
        return None
    if (d / "model.safetensors").exists():                       # a finished (final) run
        return d
    cks = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return cks[-1] if cks else None                              # else the highest checkpoint


def load_pipeline(P: int, S: int, device: str = "cpu", pipeline_cls=None,
                   offline: bool = False):
    """Load the frozen Chronos-Bolt pipeline for geometry (P, S).

    Returns (pipeline, label). (16, 16) -> official; otherwise the local checkpoint if present,
    else the Hugging Face sweep variant. `pipeline_cls` defaults to chronos.BaseChronosPipeline.

    When *offline* is True (or the HF_HUB_OFFLINE env var is set), every ``from_pretrained``
    call uses ``local_files_only=True`` so no network request is attempted — the model must
    already be in the HuggingFace cache or at a local path.
    """
    import os
    if pipeline_cls is None:
        from chronos import BaseChronosPipeline as pipeline_cls
    local_only = offline or os.environ.get("HF_HUB_OFFLINE", "") == "1"
    if (P, S) == (16, 16):
        ck = _resolve_local_ckpt(P, S)
        if ck is not None:
            return pipeline_cls.from_pretrained(str(ck), device_map=device), "p16-s16 (local)"
        return (pipeline_cls.from_pretrained(OFFICIAL_MODEL, device_map=device,
                                             local_files_only=local_only),
                f"p16-s16 (official{', offline' if local_only else ''})")
    ck = _resolve_local_ckpt(P, S)
    if ck is not None:
        return pipeline_cls.from_pretrained(str(ck), device_map=device), f"p{P}-s{S} (local {ck.name})"
    if (P, S) in _SUBFOLDER:
        return (pipeline_cls.from_pretrained(SWEEP_REPO, subfolder=_SUBFOLDER[(P, S)],
                                             device_map=device, local_files_only=local_only),
                f"p{P}-s{S} (HF {_SUBFOLDER[(P, S)]}{', offline' if local_only else ''})")
    raise ValueError(f"No model available for (P={P}, S={S}). "
                     f"Supported: (16,16) + {sorted(_SUBFOLDER)}")


def forecast_median(pipe, context, pred_len: int, qi: int, device: str = "cpu") -> np.ndarray:
    """Median (q0.5) point forecast for one context window."""
    import torch
    x = torch.tensor(np.asarray(context, np.float32), device=device).unsqueeze(0)
    return pipe.predict(x, prediction_length=pred_len)[0, qi].float().cpu().numpy()


def fit_amp_phase(y, t, f) -> tuple[float, float]:
    """Least-squares amplitude & phase of a tone at frequency `f` Hz over time grid `t` (s)."""
    X = np.stack([np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t), np.ones_like(t)], 1)
    a, b, _ = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.hypot(a, b)), float(np.arctan2(b, a))
