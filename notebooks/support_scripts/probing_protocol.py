"""Leakage-safe, frequency-grouped prequential MDL for the two probing notebooks.

The probe is StandardScaler -> PCA (at most 20 components) -> logistic regression.
PCA uses deterministic randomized SVD (seed 42), fitted only on preceding chunks;
the same unsupervised projection serves the real-label and permuted-label probes.
This is the repository's five-block MDL protocol, not the online replay-stream
estimator of Pagani et al.  With the first block coded uniformly, its maximum SV
is 1 - first_chunk_n / n rather than one.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

FS = 512
BAND = (2, 250)
STAGES = ("dec_0", "dec_1", "dec_2", "dec_3", "output_head")
# name, inclusive band lower/upper bounds, class boundary (class 1 means f > boundary)
TASKS = (
    ("LL", 2, 64, 33),
    ("L", 2, 126, 64),
    ("LH", 65, 126, 95),
    ("Mid", 2, 250, 126),
    ("HL", 127, 188, 157),
    ("H", 127, 250, 188),
    ("HH", 189, 250, 219),
)


@dataclass(frozen=True)
class ProbeConfig:
    mode: str = "full"
    seed: int = 42
    n_backgrounds: int = 100
    n_phases: int = 10
    frequency_step: int = 1
    n_chunks: int = 5
    n_pca: int = 20
    max_iter: int = 2000
    batch_size: int = 64
    threads: int = 4

    def __post_init__(self) -> None:
        if self.mode not in {"full", "smoke"}:
            raise ValueError("mode must be 'full' or 'smoke'")
        fields = ("seed", "n_backgrounds", "n_phases", "frequency_step", "n_chunks",
                  "n_pca", "max_iter", "batch_size", "threads")
        for name in fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} is outside its valid range")
        if self.n_chunks < 2 or self.n_phases < 2 or self.n_pca > 20:
            raise ValueError("Need >=2 chunks/phases and at most 20 PCA components")
        if self.mode == "full":
            required = {"seed": 42, "n_backgrounds": 100, "n_phases": 10,
                        "frequency_step": 1, "n_chunks": 5, "n_pca": 20}
            if any(getattr(self, key) != value for key, value in required.items()):
                raise ValueError(f"Full mode requires the complete registered design: {required}")

    @classmethod
    def smoke(cls, **overrides: Any) -> "ProbeConfig":
        values = dict(mode="smoke", n_backgrounds=2, n_phases=2, frequency_step=4)
        values.update(overrides)
        return cls(**values)


def context_length(P: int, S: int) -> int:
    """Longest <=512 context with no masked patch padding or uncovered suffix."""
    if (not isinstance(P, Integral) or not isinstance(S, Integral)
            or isinstance(P, bool) or isinstance(S, bool) or not 1 <= S <= P <= 512):
        raise ValueError("Expected integer geometry 1 <= S <= P <= 512")
    for length in range(FS, P - 1, -1):
        if length % P == 0 and (length - P) % S == 0:
            return length
    raise ValueError(f"No complete context exists for P={P}, S={S}")


def make_design(cfg: ProbeConfig) -> pd.DataFrame:
    """Frequency-major contexts with genuinely distinct [0, 2pi) phases."""
    frequencies = np.arange(BAND[0], BAND[1] + 1, cfg.frequency_step, dtype=float)
    phase_indices = np.arange(cfg.n_phases, dtype=int)
    return pd.DataFrame({
        "frequency": np.repeat(frequencies, cfg.n_phases),
        "phase": np.tile(2 * np.pi * phase_indices / cfg.n_phases, len(frequencies)),
        "phase_index": np.tile(phase_indices, len(frequencies)),
    })


def grouped_chunks(frequencies: np.ndarray, labels: np.ndarray, n_chunks: int = 5,
                   seed: int = 42) -> list[np.ndarray]:
    """Stratify unique frequencies by class, keeping every phase in its chunk.

    These are ordered prequential chunks, not cross-validation folds: block k
    can be encoded only from blocks 0..k-1.  Every chunk has both true classes.
    """
    freq = np.asarray(frequencies, dtype=float)
    y = np.asarray(labels)
    if (freq.ndim != 1 or y.ndim != 1 or len(freq) != len(y)
            or len(freq) == 0 or not np.isfinite(freq).all()):
        raise ValueError("Expected matching finite, nonempty frequency/label vectors")
    if not isinstance(n_chunks, Integral) or n_chunks < 2:
        raise ValueError("n_chunks must be an integer >=2")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("The task must contain both binary classes")
    unique_frequencies = np.unique(freq)
    frequency_labels = []
    for frequency in unique_frequencies:
        group_labels = np.unique(y[freq == frequency])
        if len(group_labels) != 1:
            raise ValueError("Every frequency must have one consistent task label")
        frequency_labels.append(group_labels[0])
    frequency_labels = np.asarray(frequency_labels)
    rng = np.random.default_rng(seed)
    class_parts = []
    for label in (0, 1):
        groups = unique_frequencies[frequency_labels == label]
        if len(groups) < n_chunks:
            raise ValueError(f"Class {label} has fewer than {n_chunks} frequency groups")
        class_parts.append(np.array_split(rng.permutation(groups), n_chunks))
    chunks = []
    for k in range(n_chunks):
        selected = np.concatenate([parts[k] for parts in class_parts])
        chunks.append(np.flatnonzero(np.isin(freq, selected)))
    return chunks


def _validate_chunks(chunks: Sequence[np.ndarray], n: int) -> list[np.ndarray]:
    parts = [np.asarray(chunk) for chunk in chunks]
    if len(parts) < 2 or any(c.ndim != 1 or c.size == 0 or c.dtype.kind not in "iu" for c in parts):
        raise ValueError("Need >=2 nonempty chunks containing integer row indices")
    flattened = np.concatenate(parts)
    if len(flattened) != n or not np.array_equal(np.sort(flattened), np.arange(n)):
        raise ValueError("Chunks must partition every row once, without overlap")
    return parts


def prequential_mdl(features: np.ndarray, labels: np.ndarray,
                    chunks: Sequence[np.ndarray], *, n_pca: int = 20,
                    max_iter: int = 2000, seed: int = 42,
                    control_labels: np.ndarray | None = None) -> dict[str, float | int]:
    """Compute true and shuffled codelengths, fitting transforms on past rows only.

    A caller may supply control labels for a reproducibility test; normally one
    fixed seeded global permutation is used.  Blocks remain defined by the true
    task.  Any numerical or convergence failure is surfaced, never discarded.
    """
    from sklearn.decomposition import PCA
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    if (X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or y.ndim != 1
            or len(X) != len(y) or not np.isfinite(X).all()):
        raise ValueError("Expected nonempty finite [n,d] features and matching labels")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("The task must contain both binary classes")
    if not isinstance(n_pca, Integral) or not 1 <= n_pca <= 20 or max_iter < 1:
        raise ValueError("Expected 1..20 PCA components and positive max_iter")
    parts = _validate_chunks(chunks, len(y))
    shuffled = (np.random.default_rng(seed).permutation(y) if control_labels is None
                else np.asarray(control_labels))
    if shuffled.shape != y.shape or not np.array_equal(np.sort(shuffled), np.sort(y)):
        raise ValueError("Control labels must be a permutation of the true labels")
    lengths = np.full(2, float(len(parts[0])))

    for k in range(1, len(parts)):
        train, test = np.concatenate(parts[:k]), parts[k]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train])
        X_test = scaler.transform(X[test])
        n_components = min(n_pca, X.shape[1], len(train) - 1)
        if n_components < 1:
            raise ValueError("Too few preceding examples for PCA")
        projection = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
        # Constant representations legitimately have no signal; PCA's explained
        # variance ratio is undefined but its projected coordinates remain zero.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
            Z_train = projection.fit_transform(X_train)
        Z_test = projection.transform(X_test)
        for target_index, target in enumerate((y, shuffled)):
            if len(np.unique(target[train])) < 2:
                lengths[target_index] += len(test)
                continue
            classifier = LogisticRegression(C=1.0, max_iter=max_iter, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                classifier.fit(Z_train, target[train])
            probability = classifier.predict_proba(Z_test)
            classes = list(classifier.classes_)
            p = probability[np.arange(len(test)), [classes.index(value) for value in target[test]]]
            lengths[target_index] += float(-np.log2(np.clip(p, 1e-6, 1.0)).sum())

    if not np.isfinite(lengths).all():
        raise ValueError("Prequential coding produced nonfinite lengths")
    n = len(y)
    return {"n": n, "L_bits": float(lengths[0]), "L_uniform": float(n),
            "SV": float(1 - lengths[0] / n), "L_shuffled_bits": float(lengths[1]),
            "SV_shuffled": float(1 - lengths[1] / n), "first_chunk_n": len(parts[0])}


def score_features(features: Mapping[str, np.ndarray], design: pd.DataFrame,
                   cfg: ProbeConfig, metadata: Mapping[str, Any]) -> pd.DataFrame:
    """Return seven task scores at all five stages for one base background."""
    required = {"frequency", "phase", "phase_index"}
    if not required.issubset(design.columns) or design.empty:
        raise ValueError(f"Design must contain nonempty columns {sorted(required)}")
    reserved = {"stage", "task", "n", "L_bits", "L_uniform", "SV", "L_shuffled_bits",
                "SV_shuffled", "first_chunk_n"}
    if reserved.intersection(metadata):
        raise ValueError("Metadata must not overwrite numerical score fields")
    for stage in STAGES:
        if stage not in features:
            raise ValueError(f"Missing required stage {stage}")
        array = np.asarray(features[stage])
        if array.ndim != 2 or len(array) != len(design) or not np.isfinite(array).all():
            raise ValueError(f"Invalid feature rows at stage {stage}")
    frequencies = design["frequency"].to_numpy(dtype=float)
    if not np.isfinite(frequencies).all():
        raise ValueError("Design frequencies must be finite")
    rows = []
    for task, lower, upper, boundary in TASKS:
        selected = (frequencies >= lower) & (frequencies <= upper)
        frequency = frequencies[selected]
        labels = (frequency > boundary).astype(int)
        chunks = grouped_chunks(frequency, labels, cfg.n_chunks, cfg.seed)
        for stage in STAGES:
            result = prequential_mdl(np.asarray(features[stage])[selected], labels, chunks,
                                     n_pca=cfg.n_pca, max_iter=cfg.max_iter, seed=cfg.seed)
            rows.append({**metadata, "stage": stage, "task": task, **result})
    return pd.DataFrame(rows)
