"""
testing_lib.py — shared helpers for the structural-aliasing testing notebooks.

Its ONE job is to let both notebooks (`chronosBolt_layer_probing.ipynb`,
`contamination.ipynb`) select and load a model in a single, explicit place, plus a couple of
small forecast/fit helpers the contamination notebook reuses. The notebooks keep producing
their own figures — this module never plots anything.

Model selection is by patch/stride geometry (P, S):
    (16, 16)         -> the official amazon/chronos-bolt-tiny
    any other (P, S) -> the retrained variant, loaded PREFERABLY from the local checkpoint
                        (../models/weights/p{P}-s{S}-seed42, latest) and otherwise pulled from
                        the Hugging Face sweep repo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------- #
#  Model registry + loader (local checkpoint preferred, else Hugging Face)
# ---------------------------------------------------------------------------- #
ALL_MODELS: list[tuple[int, int]] = [   # every (P, S) in the sweep, in a fixed display order
    (16, 16),   # official chronos-bolt-tiny (the p16-s16 data point)
    (16, 12),   # stride axis  (overlap 0.25)
    (16, 8),    # stride axis  (overlap 0.50)
    (16, 4),    # stride axis  (overlap 0.75)
    (8, 8),     # patch axis   (contiguous)
    (24, 24),   # patch axis   (contiguous)
]

OFFICIAL_MODEL = "amazon/chronos-bolt-tiny"
SWEEP_REPO = "federicosabbadini/chronos-bolt-patch-sweep"
_SUBFOLDER = {
    (16, 12): "p16-s12-seed42", (16, 8): "p16-s8-seed42", (16, 4): "p16-s4-seed42",
    (8, 8): "p8-s8-seed42", (24, 24): "p24-s24-seed42",
}
# this file lives in chronos/testing/, so parent.parent is chronos/
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


def load_pipeline(P: int, S: int, device: str = "cpu", pipeline_cls=None):
    """Load the frozen Chronos-Bolt pipeline for geometry (P, S).

    Returns (pipeline, label). (16, 16) -> official; otherwise the local checkpoint if present,
    else the Hugging Face sweep variant. `pipeline_cls` defaults to chronos.BaseChronosPipeline.
    """
    if pipeline_cls is None:
        from chronos import BaseChronosPipeline as pipeline_cls
    if (P, S) == (16, 16):
        return pipeline_cls.from_pretrained(OFFICIAL_MODEL, device_map=device), "p16-s16 (official)"
    ck = _resolve_local_ckpt(P, S)
    if ck is not None:
        return pipeline_cls.from_pretrained(str(ck), device_map=device), f"p{P}-s{S} (local {ck.name})"
    if (P, S) in _SUBFOLDER:
        return (pipeline_cls.from_pretrained(SWEEP_REPO, subfolder=_SUBFOLDER[(P, S)], device_map=device),
                f"p{P}-s{S} (HF {_SUBFOLDER[(P, S)]})")
    raise ValueError(f"No model available for (P={P}, S={S}). "
                     f"Supported: (16,16) + {sorted(_SUBFOLDER)}")


def geometry(pipe) -> tuple[int, int, int, int]:
    """(P, S, prediction_length, median-quantile-index) read from the loaded model config."""
    cfg = pipe.model.config.chronos_config
    qs = list(cfg["quantiles"])
    qi = qs.index(0.5) if 0.5 in qs else len(qs) // 2
    return cfg["input_patch_size"], cfg["input_patch_stride"], cfg["prediction_length"], qi


# ---------------------------------------------------------------------------- #
#  Small forecast + sinusoid-fitting helpers (reused by the contamination notebook)
# ---------------------------------------------------------------------------- #
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
