"""Leakage-safe, frequency-grouped prequential MDL for the two probing notebooks.

The probe is StandardScaler -> PCA (at most 20 components) -> logistic regression.
PCA uses deterministic randomized SVD (seed 42), fitted only on preceding chunks;
the same unsupervised projection serves the real-label and permuted-label probes.
This is the repository's five-block MDL protocol, not the online replay-stream
estimator of Pagani et al.  With the first block coded uniformly, its maximum SV
is 1 - first_chunk_n / n rather than one.

Ten probe points are read (Deliverable 1 v1): the [REG] vector after each of the
four encoder and four decoder blocks, the input of the output projection and its
output, the quantile head.  Two controls accompany every value: permuted labels
and the same architecture with untrained (randomly initialised) weights.

A second, frequentist reading mirrors the lock/control recovery count: a tone at
a candidate site and a tone beside it are located by probes that never saw their
neighbourhood, and the located rates are compared stage by stage.
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
ENCODER_STAGES = ("enc_0", "enc_1", "enc_2", "enc_3")
DECODER_STAGES = ("dec_0", "dec_1", "dec_2", "dec_3")
OUTPUT_STAGES = ("output_reg", "output_head")
STAGES = ENCODER_STAGES + DECODER_STAGES + OUTPUT_STAGES
NETWORKS = ("trained", "random_init")
ARMS = ("lock", "control")
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
    # Untrained-network control (same architecture, random weights, this seed).
    random_init: bool = True
    # Lock/control reading: sweep frequencies within this distance of a trial tone
    # are withheld from the probe that locates it; controls keep the recovery
    # count's guard of more than 2 Hz from every candidate site.
    trial_exclusion_hz: float = 1.0
    control_guard_hz: float = 2.000001

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
        if not isinstance(self.random_init, bool):
            raise ValueError("random_init must be a boolean")
        if not (0 < float(self.trial_exclusion_hz) < float(self.control_guard_hz)):
            raise ValueError("Need 0 < trial_exclusion_hz < control_guard_hz")
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
                   cfg: ProbeConfig, metadata: Mapping[str, Any],
                   control_features: Mapping[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Return seven task scores at all ten stages for one base background.

    `control_features` are the same contexts read from the untrained network; when
    given, every row also carries L_random_init_bits and SV_random_init, coded with
    the same blocks and the same probe as the trained network.
    """
    required = {"frequency", "phase", "phase_index"}
    if not required.issubset(design.columns) or design.empty:
        raise ValueError(f"Design must contain nonempty columns {sorted(required)}")
    reserved = {"stage", "task", "n", "L_bits", "L_uniform", "SV", "L_shuffled_bits",
                "SV_shuffled", "first_chunk_n", "L_random_init_bits", "SV_random_init"}
    if reserved.intersection(metadata):
        raise ValueError("Metadata must not overwrite numerical score fields")
    for source in (features,) if control_features is None else (features, control_features):
        for stage in STAGES:
            if stage not in source:
                raise ValueError(f"Missing required stage {stage}")
            array = np.asarray(source[stage])
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
            if control_features is not None:
                control = prequential_mdl(np.asarray(control_features[stage])[selected], labels,
                                          chunks, n_pca=cfg.n_pca, max_iter=cfg.max_iter,
                                          seed=cfg.seed)
                result["L_random_init_bits"] = control["L_bits"]
                result["SV_random_init"] = control["SV"]
            rows.append({**metadata, "stage": stage, "task": task, **result})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Lock/control reading
# --------------------------------------------------------------------------- #
def lock_frequencies(P: int, S: int) -> np.ndarray:
    """In-band candidate sites k*fs/P and c*fs/S (same rule as comparison_lib)."""
    return np.array(sorted(
        set(round(k * FS / P, 6) for k in range(1, P // 2 + 1) if BAND[0] <= k * FS / P <= BAND[1])
        | set(round(k * FS / S, 6) for k in range(1, S // 2 + 1) if BAND[0] <= k * FS / S <= BAND[1])))


def control_delta(P: int, S: int, frequency: float, guard: float = 2.000001) -> float:
    """Largest offset below fs/(4S) whose two controls are in band and off every site."""
    locks = lock_frequencies(P, S)
    for d in np.arange(.25 * FS / S, 1. - 1e-9, -.05):
        controls = np.array([frequency - d, frequency + d])
        if (np.all((controls >= BAND[0]) & (controls <= BAND[1]))
                and np.min(abs(controls[:, None] - locks)) >= guard):
            return float(d)
    raise ValueError(f"No disjoint controls at p{P}-s{S}, f={frequency}")


def site_branch(P: int, S: int, frequency: float) -> str:
    patch = abs(frequency * P / FS - round(frequency * P / FS)) < 1e-4
    stride = abs(frequency * S / FS - round(frequency * S / FS)) < 1e-4
    return "both" if patch and stride else "patch" if patch else "stride" if stride else "none"


def task_path(frequency: float) -> list[tuple[str, int]]:
    """The three band tasks that place a tone in an eighth of the band, with labels.

    Level one halves the band (Mid), level two halves that half (L or H) and level
    three halves the quarter (LL, LH, HL or HH).  A tone is *located* at a stage
    when the three probes all return its label.
    """
    f = float(frequency)
    if not BAND[0] <= f <= BAND[1]:
        raise ValueError("Trial tone outside the analysed band")
    boundary = {name: cut for name, _low, _high, cut in TASKS}
    second = "H" if f > boundary["Mid"] else "L"
    third = (("HH" if f > boundary["H"] else "HL") if second == "H"
             else ("LH" if f > boundary["L"] else "LL"))
    return [(name, int(f > boundary[name])) for name in ("Mid", second, third)]


def boundary_distance(frequency: float) -> float:
    """Distance in Hz from the nearest boundary of the tasks on the tone's path."""
    boundary = {name: cut for name, _low, _high, cut in TASKS}
    return float(min(abs(float(frequency) - boundary[name]) for name, _label in task_path(frequency)))


def pure_trials(P: int, S: int, cfg: ProbeConfig) -> pd.DataFrame:
    """Every candidate site of the geometry with one control on each side.

    Sites keep their exact, possibly non-integer, value.  Each tone is presented at
    every phase of the design, so lock and control arms are paired by phase.
    """
    rows = []
    for pair, lock in enumerate(lock_frequencies(P, S)):
        delta = control_delta(P, S, lock, cfg.control_guard_hz)
        for arm, frequency in (("lock", lock), ("control", lock - delta), ("control", lock + delta)):
            rows.append(dict(pair=pair, arm=arm, frequency=float(frequency), lock_hz=float(lock),
                             branch=site_branch(P, S, lock)))
    return _expand_phases(pd.DataFrame(rows), cfg)


def background_trials(P: int, S: int, n_backgrounds: int, cfg: ProbeConfig,
                      first_seed: int = 42001) -> pd.DataFrame:
    """One (site, control) pair per background, drawn as in the recovery count.

    The generator and the order of its draws repeat comparison_lib.injection_trials,
    so background i meets the same site and the same control side in both analyses.
    Row `pair` i belongs to background i only.
    """
    rng = np.random.default_rng(np.random.SeedSequence([first_seed, P, S, 131]))
    locks = lock_frequencies(P, S)
    schedule = np.resize(rng.permutation(locks), n_backgrounds)
    rng.shuffle(schedule)
    signs = np.resize(np.array([-1., 1.]), n_backgrounds)
    rng.shuffle(signs)
    rows = []
    for pair, (lock, sign) in enumerate(zip(schedule, signs)):
        control = lock + sign * control_delta(P, S, lock, cfg.control_guard_hz)
        for arm, frequency in (("lock", lock), ("control", control)):
            rows.append(dict(pair=pair, arm=arm, frequency=float(frequency), lock_hz=float(lock),
                             branch=site_branch(P, S, lock)))
    return _expand_phases(pd.DataFrame(rows), cfg)


def _expand_phases(tones: pd.DataFrame, cfg: ProbeConfig) -> pd.DataFrame:
    tones = tones.assign(boundary_distance=[boundary_distance(f) for f in tones.frequency])
    phases = pd.DataFrame({"phase_index": np.arange(cfg.n_phases, dtype=int)})
    phases["phase"] = 2 * np.pi * phases.phase_index / cfg.n_phases
    return tones.merge(phases, how="cross").reset_index(drop=True)


def score_trials(features: Mapping[str, np.ndarray], design: pd.DataFrame,
                 trial_features: Mapping[str, np.ndarray], trials: pd.DataFrame,
                 cfg: ProbeConfig, metadata: Mapping[str, Any],
                 network: str = "trained") -> pd.DataFrame:
    """Locate each trial tone with probes that never saw its neighbourhood.

    For every pair, the sweep frequencies within cfg.trial_exclusion_hz of the lock
    or of a control are withheld; the usual probe (standardisation, PCA, logistic
    regression) is fitted on the remaining rows of each task on the tones' paths and
    then reads the trial contexts.  One row per pair, tone, stage: contexts read,
    contexts located (three levels right), per-level hits, and the mean cost in bits
    of coding the three labels with the probe.
    """
    from sklearn.decomposition import PCA
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if network not in NETWORKS:
        raise ValueError(f"Unknown network {network}")
    needed = {"pair", "arm", "frequency", "lock_hz", "branch", "boundary_distance", "phase_index"}
    if not needed.issubset(trials.columns) or trials.empty:
        raise ValueError("Invalid trial design")
    if set(trials.arm) != set(ARMS):
        raise ValueError("Every trial design needs lock and control tones")
    sweep = design["frequency"].to_numpy(dtype=float)
    for stage in STAGES:
        a, b = np.asarray(features.get(stage)), np.asarray(trial_features.get(stage))
        if (a.ndim != 2 or b.ndim != 2 or len(a) != len(design) or len(b) != len(trials)
                or a.shape[1] != b.shape[1] or not np.isfinite(a).all() or not np.isfinite(b).all()):
            raise ValueError(f"Invalid sweep or trial features at stage {stage}")
    bands = {name: (low, high, cut) for name, low, high, cut in TASKS}
    rows = []
    for pair, group in trials.groupby("pair", sort=True):
        tones = group.frequency.unique()
        if np.min(np.abs(sweep[:, None] - tones[None, :]), axis=1).min() > cfg.trial_exclusion_hz + 1:
            raise ValueError("Trial tones lie outside the swept grid")
        kept = np.min(np.abs(sweep[:, None] - tones[None, :]), axis=1) > cfg.trial_exclusion_hz
        paths = {float(f): task_path(f) for f in tones}
        for stage in STAGES:
            X, T = np.asarray(features[stage], dtype=float), np.asarray(trial_features[stage], dtype=float)
            probes = {}
            for task in sorted({name for path in paths.values() for name, _ in path}):
                low, high, cut = bands[task]
                train = kept & (sweep >= low) & (sweep <= high)
                labels = (sweep[train] > cut).astype(int)
                if len(np.unique(labels)) < 2:
                    raise ValueError(f"Task {task} lost a class after withholding the trial tones")
                scaler = StandardScaler().fit(X[train])
                n_components = min(cfg.n_pca, X.shape[1], int(train.sum()) - 1)
                projection = PCA(n_components=n_components, svd_solver="randomized", random_state=cfg.seed)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="invalid value encountered in divide",
                                            category=RuntimeWarning)
                    Z = projection.fit_transform(scaler.transform(X[train]))
                classifier = LogisticRegression(C=1.0, max_iter=cfg.max_iter, random_state=cfg.seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    classifier.fit(Z, labels)
                probes[task] = (scaler, projection, classifier)
            for (arm, frequency), tone in group.groupby(["arm", "frequency"], sort=False):
                index = tone.index.to_numpy()
                hits, bits = [], np.zeros(len(index))
                for task, label in paths[float(frequency)]:
                    scaler, projection, classifier = probes[task]
                    probability = classifier.predict_proba(projection.transform(scaler.transform(T[index])))
                    p = probability[:, list(classifier.classes_).index(label)]
                    hits.append(p > .5)
                    bits += -np.log2(np.clip(p, 1e-6, 1.0))
                hits = np.array(hits)
                first = tone.iloc[0]
                rows.append({**metadata, "network": network, "stage": stage, "pair": int(pair),
                             "arm": arm, "frequency": float(frequency), "lock_hz": float(first.lock_hz),
                             "branch": first.branch, "boundary_distance": float(first.boundary_distance),
                             "n_contexts": int(len(index)), "located": int(hits.all(axis=0).sum()),
                             "level1": int(hits[0].sum()), "level2": int(hits[1].sum()),
                             "level3": int(hits[2].sum()), "bits_per_context": float(bits.mean()),
                             "n_withheld": int((~kept).sum())})
    return pd.DataFrame(rows)
