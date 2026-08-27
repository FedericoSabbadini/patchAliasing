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

import hashlib
import json
from pathlib import Path
from typing import Any

# ---- HuggingFace repository where the retrained sweep checkpoints are stored ----------
SWEEP_REPO: str = "federicosabbadini/chronos-bolt-patch-sweep"
# Immutable commit used by the Deliverable 3 analysis.  A branch or the repository default would
# make two resumed sessions capable of loading different weights under the same model tag.
SWEEP_REVISION: str = "230ea28278a3c60621964b920e9778c0ba73337e"

# ---- local directory that mirrors (or replaces) the HF repo --------------------------
# ``chronos/models/weights/`` sits two levels above this file.
_WEIGHTS_DIR: Path = Path(__file__).resolve().parent.parent / "models" / "weights"

REQUIRED_CHECKPOINT_FILES = ("model.safetensors", "config.json", "run_config.json")
EXPECTED_TRAINING_STEPS = 100_000
EXPECTED_SEED = 42


def _all_values(obj: Any, key: str) -> list[Any]:
    """Return values for ``key`` anywhere in a JSON tree (run configs changed nesting once)."""
    values: list[Any] = []
    if isinstance(obj, dict):
        for k, value in obj.items():
            if k == key:
                values.append(value)
            values.extend(_all_values(value, key))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(_all_values(value, key))
    return values


def _first_value(obj: Any, *keys: str) -> Any | None:
    for key in keys:
        values = _all_values(obj, key)
        if values:
            return values[0]
    return None


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_dir(path: Path, P: int, S: int) -> dict[str, Any]:
    """Validate a finished local checkpoint and return its parsed provenance.

    Numbered trainer checkpoints are deliberately not accepted: the experiment compares models
    after the common 100k-step budget, so an interrupted state is a different treatment.
    """
    path = Path(path)
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (path / name).is_file()]
    if missing:
        numbered = sorted(p.name for p in path.glob("checkpoint-*") if p.is_dir())
        detail = f"; partial trainer states present: {numbered}" if numbered else ""
        raise ValueError(f"incomplete checkpoint {path}: missing {missing}{detail}")

    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        run_config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checkpoint metadata in {path}: {exc}") from exc

    got_P = _first_value(config, "input_patch_size", "patch_size")
    got_S = _first_value(config, "input_patch_stride", "patch_stride")
    got_pred = _first_value(config, "prediction_length")
    if got_P is None or got_S is None:
        raise ValueError(f"checkpoint {path} does not declare patch size and stride")
    if (int(got_P), int(got_S)) != (int(P), int(S)):
        raise ValueError(
            f"checkpoint geometry mismatch: requested p{P}-s{S}, metadata says p{got_P}-s{got_S}")
    if got_pred is not None and int(got_pred) != 64:
        raise ValueError(f"checkpoint {path} has prediction_length={got_pred}, expected 64")

    status = _first_value(run_config, "status", "training_status")
    if status is not None and str(status).lower() not in {"done", "complete", "completed", "finished"}:
        raise ValueError(f"checkpoint {path} is not complete (status={status!r})")
    steps = _first_value(run_config, "steps_completed", "completed_steps", "max_steps")
    if steps is not None and int(steps) != EXPECTED_TRAINING_STEPS:
        raise ValueError(
            f"checkpoint {path} used {steps} training steps, expected {EXPECTED_TRAINING_STEPS}")
    seed = _first_value(run_config, "seed", "random_seed")
    if seed is not None and int(seed) != EXPECTED_SEED:
        raise ValueError(f"checkpoint {path} used seed={seed}, expected {EXPECTED_SEED}")

    return {
        "geometry": {"P": int(got_P), "S": int(got_S)},
        "prediction_length": None if got_pred is None else int(got_pred),
        "status": status,
        "steps": None if steps is None else int(steps),
        "seed": None if seed is None else int(seed),
    }


def resolve_local_checkpoint(P: int, S: int) -> Path | None:
    """Find the local checkpoint directory for geometry (P, S), or return None.

    A local directory is accepted only when it is a finished, metadata-consistent 100k-step run.
    Interrupted ``checkpoint-*`` directories are rejected rather than substituted for the final
    state, because doing so would break the common-budget comparison in Deliverable 3.
    """
    d = _WEIGHTS_DIR / f"p{P}-s{S}-seed42"
    if not d.exists():
        return None
    validate_checkpoint_dir(d, P, S)
    return d


def checkpoint_identity(P: int, S: int) -> dict[str, Any]:
    """Return an immutable, JSON-serialisable identity for the checkpoint selected for pP-sS."""
    local = resolve_local_checkpoint(P, S)
    if local is None:
        payload: dict[str, Any] = {
            "source": "huggingface",
            "repo": SWEEP_REPO,
            "revision": SWEEP_REVISION,
            "subfolder": f"p{P}-s{S}-seed42",
            "geometry": {"P": int(P), "S": int(S)},
        }
    else:
        metadata = validate_checkpoint_dir(local, P, S)
        files = {
            name: {"bytes": (local / name).stat().st_size, "sha256": _sha256(local / name)}
            for name in REQUIRED_CHECKPOINT_FILES
        }
        payload = {
            "source": "local",
            "path": str(local.resolve()),
            "metadata": metadata,
            "files": files,
        }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["identity_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload
