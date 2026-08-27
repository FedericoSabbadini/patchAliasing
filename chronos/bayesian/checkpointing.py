"""Small atomic-I/O and fingerprint primitives for the Deliverable 3 Bayesian run."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_target(path: Path) -> Path:
    path = Path(path)
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")


def atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    """Write beside ``path``, flush via the writer, then atomically replace the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_target(path)
    try:
        writer(temporary)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise OSError(f"writer produced no data for {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    def _write(target: Path) -> None:
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    atomic_write(path, _write)


def atomic_parquet(path: Path, frame: Any) -> None:
    atomic_write(path, lambda target: frame.to_parquet(target, index=False))


def atomic_npy(path: Path, array: Any) -> None:
    import numpy as np

    def _write(target: Path) -> None:
        with target.open("wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())

    atomic_write(path, _write)


def atomic_netcdf(path: Path, idata: Any) -> None:
    atomic_write(path, lambda target: idata.to_netcdf(target))

