"""
model_loader.py — locate and load retrained Chronos-Bolt checkpoints.

Every (P, S) geometry in the sweep has a checkpoint directory named ``p{P}-s{S}-seed42``
under ``chronos/models/weights/``.  This module resolves that directory, preferring the
local copy when one exists and falling back to the Hugging Face sweep repository.

This is the only place checkpoint resolution lives for the Bayesian analysis.  The older
``chronos/testing/testing_lib.py`` provides the same service for the Deliverable 1 testing
notebooks; this copy exists so that ``chronos/bayesian/`` is self-contained.
"""
from __future__ import annotations

from pathlib import Path

# ---- HuggingFace repository where the retrained sweep checkpoints are stored ----------
SWEEP_REPO: str = "federicosabbadini/chronos-bolt-patch-sweep"

# ---- local directory that mirrors (or replaces) the HF repo --------------------------
# ``chronos/models/weights/`` sits two levels above this file.
_WEIGHTS_DIR: Path = Path(__file__).resolve().parent.parent / "models" / "weights"


def resolve_local_checkpoint(P: int, S: int) -> Path | None:
    """Find the local checkpoint directory for geometry (P, S), or return None.

    A checkpoint is valid when it contains ``model.safetensors`` (a finished training run).
    If only numbered ``checkpoint-*`` sub-directories exist (an interrupted run), the highest
    numbered one is returned as a best-effort fallback.
    """
    d = _WEIGHTS_DIR / f"p{P}-s{S}-seed42"
    if not d.exists():
        return None
    if (d / "model.safetensors").exists():
        return d
    # Interrupted run: pick the highest checkpoint
    cks = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return cks[-1] if cks else None
