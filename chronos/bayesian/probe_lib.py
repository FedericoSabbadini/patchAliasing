"""
probe_lib.py, the *light*, notebook-free probing core for the Bayesian analysis.

Everything the Bayesian notebook needs from Chronos lives here. The two existing testing
notebooks (`chronosBolt_layer_probing.ipynb`, `contamination.ipynb`) stay untouched: this module
distils the parts of them that produce NUMBERS (MDL codelengths, forecast recovery, token
collapse) and drops everything that produces FIGURES, so the Bayesian notebook never has to
execute another notebook.

Provenance of each piece (so the port can be audited against the originals):

    forecast / recovery / collapse ....... originally from hypotheses.py (D1)
    [REG] capture across enc/dec/output ... originally from probing notebook (D1) §3
    prequential MDL codelength ........... originally from probing notebook (D1) §5
    lock geometry (k*fs/P, c*fs/S) ....... deliverable Eq. 7
    model registry / loader .............. model_loader.py  (in this directory)

The one substantive change is **batching**: `hypotheses.py` runs one context per forward pass,
which is fine for a handful of figures but not for the ~37k forward passes the Bayesian design
needs. Every method here takes a stack of contexts `[B, CTX]` and returns `[B, ...]`.

Conventions are kept bit-identical to the rest of the project so the numbers remain comparable:
fs = 512 Hz, context = 480 samples, horizon = 64, band = [2, 250] Hz, and a tone injected on a
unit-variance generator background at SNR = 4.
"""
from __future__ import annotations

import sys
from math import gcd
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------- #
#  Repo wiring: this file lives in chronos/bayesian/
# --------------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent
_CHRONOS = _HERE.parent
_SYNTHETIC = _CHRONOS / "data" / "synthetic"
for _p in (_HERE, _SYNTHETIC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# --------------------------------------------------------------------------------- #
#  Fixed experimental setup (deliverable convention, do not change without changing the .tex)
# --------------------------------------------------------------------------------- #
FS = 512                    # sampling frequency [Hz]; Nyquist = 256 Hz
CTX = 480                   # context length. Divisible by every stride used by the Bayesian design
                            # (8/12/15/16/20/24/32) so the patch grid is exact and no internal
                            # padding fakes a collapse. S=28 is the one exception: see BAYES_MODELS.
PRED = 64                   # forecast horizon (Chronos-Bolt's native prediction_length)
BAND = (2.0, 250.0)         # analysis band, strictly inside Nyquist
TONE_SNR = 4.0              # tone amplitude over a unit-variance background (probing convention)
SEED = 42

# The geometries of tab:patchStride: every (P, S) on the sweep repository with S >= 8. All are
# retrained from scratch under the same budget; the published amazon/chronos-bolt-tiny checkpoint
# is NOT part of the design (deliverable2.tex, "Models and signals"). The S <= 5 runs are excluded,
# matching deliverable1.tex: their stride-lock class has too few in-band members to carry a local
# H1 contrast.
#
# S | P  decides whether a geometry can separate the two branches at all: when the stride divides
# the patch, every stride lock c*fs/S is also a patch null k*fs/P with k = c*P/S, so the geometry
# yields no stride-only site. Ten of the nineteen have S ∤ P; between them they carry 68 sites.
MODELS_ALL: list[tuple[int, int]] = [
    (8, 4),     # O=0.500   S|P    included on user request (S=4; deliverable excludes it by design)
    (8, 5),     # O=0.375   S∤P    included on user request (S=5; deliverable excludes it by design)
    (8, 8),     # O=0.000   S|P
    (16, 4),    # O=0.750   S|P    included on user request (S=4; deliverable excludes it by design)
    (16, 8),    # O=0.500   S|P
    (16, 12),   # O=0.250   S∤P    ( 4 stride-only)
    (16, 15),   # O=0.062   S∤P    ( 7 stride-only)
    (16, 16),   # O=0.000   S|P    budget-matched baseline
    (24, 8),    # O=0.667   S|P
    (24, 12),   # O=0.500   S|P
    (24, 15),   # O=0.375   S∤P    ( 6 stride-only)
    (24, 16),   # O=0.333   S∤P    ( 4 stride-only)
    (24, 20),   # O=0.167   S∤P    ( 8 stride-only)
    (24, 24),   # O=0.000   S|P
    (32, 8),    # O=0.750   S|P
    (32, 12),   # O=0.625   S∤P    ( 4 stride-only)
    (32, 15),   # O=0.531   S∤P    ( 7 stride-only)
    (32, 16),   # O=0.500   S|P
    (32, 20),   # O=0.375   S∤P    ( 8 stride-only)
    (32, 24),   # O=0.250   S∤P    ( 8 stride-only)
    (32, 28),   # O=0.125   S∤P    (12 stride-only)  -- CTX % 28 != 0, see BAYES_MODELS
    (32, 32),   # O=0.000   S|P
]

# Exclude certain stride values: S5, S15, S28 are not used
EXCLUDE_S: set[int] = {5, 15, 28}
MODELS: list[tuple[int, int]] = [(P, S) for P, S in MODELS_ALL if S not in EXCLUDE_S]

# The subset the Bayesian analysis fits. The context has to close on a whole number of strides,
# otherwise the last patch is internally padded and the padding, not the geometry, produces the
# collapse. CTX = 480 satisfies this for every stride except 28 (480 = 17*28 + 4); the smallest
# context divisible by 28 as well is 1680, which no longer fits the probing convention. p32-s28 is
# therefore collected descriptively but not fitted, and the deliverable records the exclusion.
BAYES_MODELS: list[tuple[int, int]] = [(P, S) for (P, S) in MODELS if CTX % S == 0]

# Optional runtime override, so a local-only subset can be run without editing this file or the
# notebook: PATCHALIASING_MODELS="p16-s12,p16-s8,p8-s8,p24-s24" restricts MODELS to those tags.
# Combined with HF_HUB_OFFLINE=1 (which model_loader honours) this drives a fully
# offline run on the local checkpoints alone.
import os as _os
_only = _os.environ.get("PATCHALIASING_MODELS", "").strip()
if _only:
    _tags = {t.strip() for t in _only.split(",") if t.strip()}
    MODELS = [(P, S) for (P, S) in MODELS_ALL if f"p{P}-s{S}" in _tags and S not in EXCLUDE_S]
BAYES_MODELS = [(P, S) for (P, S) in MODELS if CTX % S == 0]


def design_gaps(models: list[tuple[int, int]] | None = None) -> list[str]:
    """What the current subset of MODELS cannot identify, as plain sentences.

    Worth printing rather than discovering from a degenerate posterior. A geometry supplies
    stride-only sites only when S does not divide P; without them theta_S and kappa_S have no
    frequency at which a loss is attributable to the stride branch alone, and the H3a estimand is
    fitted on whatever happens to be left.
    """
    models = models or MODELS
    out = []
    so = [(P, S) for (P, S) in models if P % S != 0]
    if not so:
        out.append("no geometry has S not dividing P: no stride-only site exists, so theta_S and "
                   "kappa_S (H3a) are not identified at all.")
    elif len(so) == 1:
        P, S = so[0]
        out.append(f"only p{P}-s{S} supplies stride-only sites, so kappa_S (H3a) is fitted at a "
                   f"single value of fs/S and its slope rests on one geometry.")
    n_P = len({P for P, _ in models})
    if n_P < 2:
        out.append("the patch size does not vary: delta_P and kappa_P (H3b) are not identified.")
    ov = {round((P - S) / P, 3) for P, S in models}
    if len(ov) < 3:
        out.append(f"only {len(ov)} distinct overlap values: delta_O (M1) rests on a short axis.")
    pend = [m for m in MODELS_ALL if m not in models]
    if pend:
        out.append("not yet trained: " + ", ".join(model_tag(P, S) for P, S in pend))
    return out

GENERATORS = ("tsmixup", "kernelsynth")   # the deliverable's two synthetic corpora


# --------------------------------------------------------------------------------- #
#  The Light TSMixup sinusoidal pool
# --------------------------------------------------------------------------------- #
TSMIXUP_POOL_N_FREE = 150      # non-integer, non-candidate frequencies added to the pool
TSMIXUP_POOL_GUARD = 0.2       # [Hz] minimum distance of a "free" frequency from any candidate


def tsmixup_pool(n_free: int = TSMIXUP_POOL_N_FREE, guard: float = TSMIXUP_POOL_GUARD,
                 models=None) -> list[float]:
    """The frequencies of the Light TSMixup source pool, in Hz.

    Three groups:
      1. every integer frequency in BAND, so the pool covers the analysed band uniformly and a
         candidate site is not represented differently from the controls that flank it;
      2. the members of F_lock that are NOT integers, which group 1 cannot reach: they come from
         the geometries whose S or P is 12, 20 or 24, where fs/D is not a whole number of Hz;
      3. the control frequencies f_k +/- delta of every candidate of every geometry, so that the
         two arms of a contrast triplet are represented in the background identically: without
         them every locked arm would sit on a pool frequency and fewer than half the control arms
         would, which is an asymmetry the paired contrast assumes is absent;
      4. `n_free` non-integer frequencies spread across the band and kept at least `guard` Hz away
         from every candidate, so the pool also carries content no geometry predicts.

    Built from MODELS rather than hard-coded, so adding a geometry adds its own non-integer
    candidates automatically. Deterministic, hence reproducible without a seed.
    """
    models = models or MODELS
    lo, hi = BAND

    lock = set()
    for P, S in models:
        lock |= {round(f, 9) for f in f_lock(P, S, fmax=hi, fmin=lo)}
    lock_sorted = sorted(lock)

    integers = [float(f) for f in range(int(np.ceil(lo)), int(np.floor(hi)) + 1)]
    non_integer_locks = [f for f in lock_sorted if abs(f - round(f)) > 1e-9]

    controls = set()
    for P, S in models:
        for fk in f_lock(P, S, fmax=hi, fmin=lo):
            d = control_offset(P, S, fk)
            if not np.isfinite(d):
                continue
            for f in (fk - d, fk + d):
                if lo <= f <= hi:
                    controls.add(round(f, 9))

    def _clear(f: float) -> bool:
        return (min(abs(f - l) for l in lock_sorted) >= guard) and (abs(f - round(f)) >= guard)

    free = []
    for f0 in np.linspace(lo + 0.5, hi - 0.5, n_free):
        f, k = float(f0), 0
        while not _clear(f) and k < 40:          # nudge off a collision, alternating either way
            k += 1
            f = float(f0) + 0.05 * ((k + 1) // 2) * (1 if k % 2 else -1)
        free.append(round(f, 4))

    return sorted(set(integers) | {round(f, 9) for f in non_integer_locks} | controls | set(free))


def model_tag(P: int, S: int) -> str:
    """Filesystem-safe id for a geometry (e.g. 'p16-s12')."""
    return f"p{P}-s{S}"


# --------------------------------------------------------------------------------- #
#  1. Signals: generator backgrounds + injected tone
# --------------------------------------------------------------------------------- #
_bg_cache: dict[tuple[str, int], np.ndarray] = {}

# Every background is drawn ONCE at this length and sliced to whatever a caller asks for.
# Drawing separately per length would not give nested signals: length is a generator parameter,
# not a window. TSMixup lays a fixed number of cycles over the requested length, so a 480-sample
# draw and a 544-sample draw from the same seed have different spectral content (measured
# correlation -0.15), and "background 0" would mean a different signal in different tables.
CANON_LEN = CTX + PRED


def make_tone(f: float, phase: float = 0.0, n: int = CTX, amp: float = 1.0) -> np.ndarray:
    """A pure sinusoid, in the project's canonical injection convention (see data/synthetic/tones.py)."""
    t = np.arange(n) / FS
    return (amp * np.sin(2 * np.pi * f * t + phase)).astype(np.float32)


def background(generator: str, n: int, seed: int) -> np.ndarray:
    """The first `n` samples of one unit-variance background realisation.

    `generator` is "tsmixup" or "kernelsynth", the two corpora named in the deliverable's Data
    section. Draw `seed` is generated once at `CANON_LEN` and sliced, so every caller sees the same
    realisation regardless of the length it needs: the 480 samples an MDL cell uses are the first
    480 of the 544 a contrast triplet uses. Results are cached on (generator, seed) for the life of
    the process; KernelSynth draws from a GP prior, which costs an O(n^3) factorisation.

    The design reuses a fixed pool of backgrounds as a grouping factor rather than drawing a fresh
    one per observation, so this cache is hit far more often than it is missed.
    """
    key = (generator, int(seed))
    if key not in _bg_cache:
        import signalGenerator as sg                  # chronos/data/synthetic/signalGenerator.py

        m = max(int(n), CANON_LEN)
        tmp = _HERE / "_gen_tmp"                      # generators want an output dir; nothing is saved
        tmp.mkdir(parents=True, exist_ok=True)
        if generator == "kernelsynth":
            # KernelSynth spec of the deliverable's signal set (J=5 composite GP kernels)
            params = {"J": 5, "l_syn": m, "fs": FS, "jitter": 1e-4, "P": 16}
            gen = sg.runKernelSynth(params, seed, tmp)
        elif generator == "tsmixup":
            # light-TSMixup spec. The source pool is the frequency pool of `tsmixup_pool`, so the
            # background covers the analysed band instead of the three low-frequency sources the
            # earlier `synthetic` mode supplied.
            params = {"K": 10, "alpha": 1.5, "l_min": m, "l_max": m, "fs": FS, "P": 16,
                      "data_mode": "frequencies", "pool_freqs": tsmixup_pool()}
            gen = sg.runTSMixup(params, seed, tmp)
        else:
            raise ValueError(f"unknown generator {generator!r}; expected one of {GENERATORS}")

        x = np.asarray(gen.generate(), float).ravel()[:m]
        sd = x.std()
        x = (x / sd) if sd > 1e-8 else x              # unit variance, so TONE_SNR is a real SNR
        _bg_cache[key] = x.astype(np.float32)

    out = _bg_cache[key]
    if n > len(out):
        raise ValueError(f"background of length {n} requested but only {len(out)} drawn")
    return out[:n]


def background_pool(generator: str, n_bg: int, length: int = CTX + PRED,
                    seed0: int = 10_000) -> list[np.ndarray]:
    """A fixed, reproducible pool of `n_bg` backgrounds, the design's `u_background` levels.

    Generated once at the FULL length (context + horizon) so that slicing gives a context and its
    genuine continuation; the forecast target is then the real future of the input, not a
    separately drawn signal.
    """
    return [background(generator, length, seed0 + i) for i in range(n_bg)]

def save_signal_pool(out_dir, generators=GENERATORS, n_bg: int = 6,
                     seed0: int = 10_000) -> "object":
    """Write the generated backgrounds -- and a few complete model inputs -- to disk.

    The backgrounds are the actual signals every posterior in this analysis is computed from, so
    they are archived rather than left as an in-memory cache: without them a reader can read the
    method but cannot reproduce a single number. `examples` is a list of (frequency, phase) pairs
    saved as full contexts (background + injected tone), i.e. exactly what Chronos is fed.

    One file per draw. Because every background is drawn at `CANON_LEN` and sliced, the archived
    signal is the whole realisation: the shorter versions used by the MDL cells, the band tasks and
    the collapse sweep are its leading samples, so nothing further needs storing.

    Returns a dataframe of the metadata, and writes `<out_dir>/signals/`.
    """
    import pandas as pd

    out = Path(out_dir) / "signals"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for gen in generators:
        for i, x in enumerate(background_pool(gen, n_bg, CANON_LEN, seed0)):
            name = f"background_{gen}_bg{i}.npy"
            np.save(out / name, x)
            rows.append(dict(kind="background", generator=gen, bg_id=i, seed=seed0 + i,
                             n=len(x), std=float(x.std()), file=name))

    meta = pd.DataFrame(rows)
    meta.to_parquet(out / "signals_index.parquet", index=False)
    return meta


def build_context(bg: np.ndarray | None, f: float, phase: float, n: int = CTX) -> np.ndarray:
    """One model input: unit-variance background + tone at (f, phase), or the pure tone if bg is None.

    `bg is None` reproduces the clean-sinusoid mode of hypotheses.py, where the token collapse at a
    stride lock is EXACTLY zero. With a background the collapse becomes a deep dip instead, both
    modes are collected, because H3's comb model is fitted on each.
    """
    tone = make_tone(f, phase, n, TONE_SNR if bg is not None else 1.0)
    return tone if bg is None else (bg[:n] + tone).astype(np.float32)


def phases_Sf(f: float, n_max: int = 10) -> np.ndarray:
    """Non-redundant phase offsets S_f = min(fs/gcd(f, fs) - 1, n_max)  (Pagani et al., Eq. 6).

    A tone at a patch-aligned frequency admits only a few genuinely distinct alignments with the
    patch grid; drawing independent uniform phases would oversample duplicates. Spreading `n_ph`
    offsets over one period is what the probing notebook and hypotheses.py both do.
    """
    period = FS // gcd(int(round(f)), FS)
    n_ph = int(np.clip(period - 1, 1, n_max))
    ks = np.linspace(0, period, n_ph, endpoint=False)
    return 2 * np.pi * f * ks / FS


def fit_amp_phase(y: np.ndarray, t: np.ndarray, f: float) -> tuple[float, float]:
    """Least-squares amplitude and phase of a tone at `f` Hz over the time grid `t` (seconds).

    Regressing y on [cos, sin, 1] and taking the hypotenuse is the estimator prescribed by the
    deliverable's Bayesian section for the recovery ratio R = A_pred / A_true.
    """
    X = np.stack([np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t), np.ones_like(t)], 1)
    a, b, _ = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.hypot(a, b)), float(np.arctan2(b, a))


# --------------------------------------------------------------------------------- #
#  2. Lock geometry ,  F_lock = {c*fs/S} union {k*fs/P}   (deliverable Eq. 7)
# --------------------------------------------------------------------------------- #
def patch_nulls(P: int, fmax: float = BAND[1], fmin: float = BAND[0]) -> list[float]:
    """Patch-integration nulls k*fs/P: a tone completing an integer number of cycles inside one patch."""
    out = [k * FS / P for k in range(1, int(fmax * P / FS) + 2)]
    return [f for f in out if fmin <= f <= fmax]


def stride_locks(S: int, fmax: float = BAND[1], fmin: float = BAND[0]) -> list[float]:
    """Structural stride locks c*fs/S: x[n+S] = x[n], so consecutive patches see identical samples."""
    out = [c * FS / S for c in range(1, int(fmax * S / FS) + 2)]
    return [f for f in out if fmin <= f <= fmax]


def f_lock(P: int, S: int, fmax: float = BAND[1], fmin: float = BAND[0]) -> list[float]:
    """The deliverable's F_lock for one geometry: the UNION of the two families (Eq. 7 uses `or`)."""
    both = set(np.round(patch_nulls(P, fmax, fmin), 6)) | set(np.round(stride_locks(S, fmax, fmin), 6))
    return sorted(float(f) for f in both)


def lock_family(f: float, P: int, S: int, tol: float = 1e-6) -> str:
    """Label a lock frequency: 'stride', 'patch' or 'both'."""
    is_s = any(abs(f - x) < tol for x in stride_locks(S))
    is_p = any(abs(f - x) < tol for x in patch_nulls(P))
    return "both" if (is_s and is_p) else ("stride" if is_s else ("patch" if is_p else "none"))


def union_grid(models: list[tuple[int, int]] = None, step: float = 1.0,
               band: tuple[float, float] = BAND) -> np.ndarray:
    """The H3 sweep grid: a uniform `step`-Hz grid UNION the exact predicted sites of EVERY geometry.

    Why the union and not each model's own grid: scoring a geometry only where it predicts a dip
    would make every model trivially "pass". Evaluating all of them on one grid that contains all
    competing predictions is what makes the comb-model comparison (M_S vs M_P vs M_0) falsifiable,
    and it puts the non-integer sites (42.66... Hz for S=12, 21.33... Hz for S=24) on the grid,
    which a plain 1 Hz sweep misses entirely.
    """
    models = MODELS if models is None else models
    lo, hi = band
    grid = set(np.round(np.arange(lo, hi + 1e-9, step), 6))
    for (P, S) in models:
        grid |= set(np.round(patch_nulls(P, hi, lo), 6))
        grid |= set(np.round(stride_locks(S, hi, lo), 6))
    return np.array(sorted(grid), dtype=float)


def controls_are_clean(fk: float, delta: float, P: int, S: int, guard: float = 1.0) -> bool:
    """True if neither control f_k +/- delta lands on another member of F_lock.

    The paired contrast is only interpretable when the controls are genuinely non-locked. For
    example at P=16, S=12 the patch null 32 Hz has its upper control at 42.67 Hz, which is the
    first STRIDE lock, that site is dropped rather than silently compared against a lock.
    """
    others = [x for x in f_lock(P, S) if abs(x - fk) > 1e-6]
    for c in (fk - delta, fk + delta):
        if c < BAND[0] or c > BAND[1]:
            return False
        if any(abs(c - o) < guard for o in others):
            return False
    return True


def control_offset(P: int, S: int, fk: float | None = None, guard: float = 1.0,
                   step: float = 0.05) -> float:
    """Control distance for a lock site, per deliverable2.tex, "What enters the inference".

    Deliverable 1 fixed delta = 0.25 * fs / S, a quarter of the stride-lock spacing. That rule is
    defined from the stride alone and therefore cannot see the patch grid, and at P=16, S=12 it is
    exactly one third of the patch-null spacing fs/P: every stride-only site then has one control
    sitting on a patch null, and the site is discarded. All four of them are, which is the whole of
    the stride-only evidence that geometry can supply.

    The rule adopted here is the largest offset not exceeding 0.25 * fs / S that keeps BOTH
    controls at least `guard` away from every member of F_lock. It changes nothing anywhere else in
    the design (at every other geometry the default offset is already clean) and recovers the four
    sites at p16-s12.

    `fk=None` returns the Deliverable 1 default, for the sensitivity comparison.
    """
    d0 = 0.25 * FS / S
    if fk is None:
        return d0
    d = d0
    while d >= 1.0:
        if controls_are_clean(fk, d, P, S, guard=guard):
            return round(d, 6)
        d -= step
    return float("nan")          # no clean offset exists: the caller drops the site


# --------------------------------------------------------------------------------- #
#  3. The model wrapper, batched forecast, token collapse, [REG] capture
# --------------------------------------------------------------------------------- #
def load_checkpoint(P: int, S: int, device: str = "cpu", pipeline_cls=None):
    """Resolve (P, S) to a retrained checkpoint. Returns (pipeline, label).

    Every geometry is ``p{P}-s{S}-seed42``, resolved by ``model_loader.py`` in this directory.
    The local checkpoint is preferred; if absent, the Hugging Face sweep repository is tried.
    The published ``amazon/chronos-bolt-tiny`` is never used: even (16, 16) loads the retrained
    variant so the baseline is budget-matched like the rest of the design.
    """
    import model_loader as ml

    if pipeline_cls is None:
        from chronos import BaseChronosPipeline as pipeline_cls
    ck = ml.resolve_local_checkpoint(P, S)
    if ck is not None:
        return pipeline_cls.from_pretrained(str(ck), device_map=device), f"p{P}-s{S} (local {ck.name})"
    sub = f"p{P}-s{S}-seed42"
    try:
        pipe = pipeline_cls.from_pretrained(ml.SWEEP_REPO, subfolder=sub, device_map=device)
    except Exception as e:
        pend = "" if (P, S) in MODELS_ALL else " It is not one of probe_lib.MODELS_ALL."
        raise RuntimeError(
            f"No checkpoint for (P={P}, S={S}). Expected subfolder '{sub}' in {ml.SWEEP_REPO}, or a "
            f"local copy under {ml._WEIGHTS_DIR}.{pend}\n  original error: {e}") from e
    return pipe, f"p{P}-s{S} (HF {sub})"


class Probe:
    """A loaded Chronos-Bolt checkpoint, wrapped for batched measurement.

    Selection goes through `load_checkpoint` above: every geometry in MODELS is a retrained variant
    under the uniform 100k-step budget, including (16, 16), and the published
    `amazon/chronos-bolt-tiny` is not used by this analysis.
    """

    def __init__(self, P: int, S: int, device: str | None = None, batch_size: int = 64):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.pipe, self.label = load_checkpoint(P, S, device=self.device)

        cfg = self.pipe.model.config.chronos_config
        self.P = int(cfg["input_patch_size"])
        self.S = int(cfg["input_patch_stride"])
        self.pred_len = int(cfg["prediction_length"])
        quantiles = list(cfg["quantiles"])
        # the median forecast is the point prediction the deliverable's recovery ratio is fitted to
        self.qi = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
        self.tag = model_tag(self.P, self.S)

        mdl = self.pipe.model
        self.n_enc = len(mdl.encoder.block)
        self.n_dec = len(mdl.decoder.block)
        # the deliverable's probe points: [REG] after every encoder and decoder block, plus the
        # 256-d pre-projection [REG] and the projected quantile head (Pagani et al.'s h4).
        self.stages = ([f"enc_{k}" for k in range(self.n_enc)]
                       + [f"dec_{k}" for k in range(self.n_dec)]
                       + ["output_reg", "output_head"])

        if (self.P, self.S) != (P, S):
            print(f"WARNING: requested (P={P}, S={S}) but the loaded checkpoint is "
                  f"(P={self.P}, S={self.S})")

    # ---------------------------------------------------------------------- #
    def close(self) -> None:
        """Drop the model and free GPU memory so the next geometry can be loaded."""
        del self.pipe
        self.pipe = None
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    # ---------------------------------------------------------------------- #
    def _batches(self, contexts: np.ndarray):
        """Yield (start, tensor) slices of a [N, CTX] stack, sized to `batch_size`."""
        X = np.asarray(contexts, np.float32)
        if X.ndim == 1:
            X = X[None, :]
        for i in range(0, len(X), self.batch_size):
            chunk = X[i:i + self.batch_size]
            yield i, self.torch.tensor(chunk, device=self.device)

    def forecast(self, contexts: np.ndarray) -> np.ndarray:
        """Median (q=0.5) point forecasts for a stack of contexts. [N, CTX] -> [N, PRED]."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for _, xb in self._batches(contexts):
                # pipe.predict returns [batch, n_quantiles, prediction_length]
                y = self.pipe.predict(xb, prediction_length=PRED)[:, self.qi, :]
                out.append(y.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    # ---------------------------------------------------------------------- #
    def recovery(self, contexts: np.ndarray, futures: np.ndarray, freqs: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Forecast amplitude recovery R = A_pred / A_true, and the phase error in degrees.

        `futures[i]` must be the TRUE continuation of `contexts[i]` (same background, same tone),
        and `freqs[i]` the frequency whose amplitude is being read back. Both amplitudes are fitted
        with the same least-squares estimator over the same 64-step horizon, so R is a ratio of
        like for like and is not inflated by the background's own energy at that frequency.
        """
        preds = self.forecast(contexts)
        t_fut = np.arange(CTX, CTX + preds.shape[1]) / FS
        R = np.empty(len(preds))
        dphase = np.empty(len(preds))
        for i, f in enumerate(np.asarray(freqs, float)):
            a_hat, ph_hat = fit_amp_phase(preds[i], t_fut, f)
            a_true, ph_true = fit_amp_phase(np.asarray(futures[i], float), t_fut, f)
            R[i] = a_hat / max(a_true, 1e-9)
            dphase[i] = np.degrees(abs(np.angle(np.exp(1j * (ph_hat - ph_true)))))
        return R, dphase

    # ---------------------------------------------------------------------- #
    def collapse(self, contexts: np.ndarray) -> np.ndarray:
        """Across-patch dispersion of the input-patch-embedding tokens. [N, CTX] -> [N].

        This is the quantity the deliverable calls the token collapse: 0 means consecutive patches
        embed to the SAME vector (t_k = t_{k+1}), which is exactly the degeneracy the phase-lock
        condition predicts. We hook `input_patch_embedding` and take, per item, the mean over the
        256 embedding dimensions of the standard deviation across patches.
        """
        torch = self.torch
        emb = self.pipe.model.input_patch_embedding
        captured: dict[str, np.ndarray] = {}

        def hook(_m, _i, o):
            a = o[0] if isinstance(o, tuple) else o
            captured["t"] = a.detach().float().cpu().numpy()   # [batch, n_patches, d_model]

        out = []
        with torch.no_grad():
            for _, xb in self._batches(contexts):
                h = emb.register_forward_hook(hook)
                try:
                    self.pipe.predict(xb, prediction_length=PRED)
                finally:
                    h.remove()
                tok = captured["t"]
                out.append(tok.std(axis=1).mean(axis=1))       # std over patches, mean over dims
        return np.concatenate(out, axis=0)

    # ---------------------------------------------------------------------- #
    @staticmethod
    def _reg(a, which: str) -> np.ndarray:
        """Pull the single [REG] vector out of a block output. [B, T, D] -> [B, D].

        Encoder blocks append the [REG] token, so it is the LAST position; decoder blocks emit a
        single query token, so position 0 IS the [REG] vector.
        """
        a = a[0] if isinstance(a, tuple) else a
        a = a.detach().float().cpu().numpy()
        if a.ndim == 2:                       # already [B, D]
            return a
        return a[:, -1, :] if which == "last" else a[:, 0, :]

    def capture_reg(self, contexts: np.ndarray, pipe=None) -> dict[str, np.ndarray]:
        """{stage: [N, D]} representations for a stack of contexts.

        `pipe` lets the caller pass a random-init clone for the architectural control; it defaults
        to the trained model. Stages are the deliverable's probe points (see `self.stages`).
        """
        torch = self.torch
        pipe = pipe or self.pipe
        mdl = pipe.model
        cap: dict[str, np.ndarray] = {}
        acc: dict[str, list] = {s: [] for s in self.stages}

        def save(key, which):
            def hook(_m, _i, o):
                cap[key] = self._reg(o, which)
            return hook

        def save_input(key):
            def hook(_m, i, _o):
                cap[key] = self._reg(i[0], "single")
            return hook

        with torch.no_grad():
            for _, xb in self._batches(contexts):
                handles = []
                for k, b in enumerate(mdl.encoder.block):
                    handles.append(b.register_forward_hook(save(f"enc_{k}", "last")))
                for k, b in enumerate(mdl.decoder.block):
                    handles.append(b.register_forward_hook(save(f"dec_{k}", "single")))
                if hasattr(mdl, "output_patch_embedding"):
                    handles.append(mdl.output_patch_embedding.register_forward_hook(save_input("output_reg")))
                    handles.append(mdl.output_patch_embedding.register_forward_hook(save("output_head", "single")))
                try:
                    pipe.predict(xb, prediction_length=PRED)
                finally:
                    for h in handles:
                        h.remove()
                for s in self.stages:
                    acc[s].append(cap[s])
        return {s: np.concatenate(v, axis=0) for s, v in acc.items()}

    def random_init_clone(self):
        """An untrained copy of the same architecture, the deliverable's random-init control.

        A high MDL compression on an untrained network means the band is exposed by the
        architecture (random projections over a patch), not learned; only the gap between the two
        is evidence about training.
        """
        import copy
        clone = copy.deepcopy(self.pipe)
        clone.model = type(self.pipe.model)(copy.deepcopy(self.pipe.model.config)).eval()
        if self.device == "cuda":
            clone.model = clone.model.to("cuda")
        return clone


# --------------------------------------------------------------------------------- #
#  4. Prequential MDL codelength  (Voita & Titov 2020; probing notebook §5)
# --------------------------------------------------------------------------------- #
def mdl_codelength(Xf: np.ndarray, y: np.ndarray, K: int = 5, n_pca: int = 20,
                   seed: int = SEED) -> float:
    """Online (prequential) description length in BITS of the labels y given the features Xf.

    The data are shuffled once and split into K chunks. The first chunk is charged at the uniform
    rate (1 bit/example, since the tasks are binary); every later chunk is encoded with a probe
    trained ONLY on the chunks before it, and charged -log2 p(true label). Summing gives the cost
    of transmitting the labels together with the probe that predicts them, a probe that memorises
    noise pays for itself and gains nothing, which is precisely why MDL is preferred to accuracy.

    The probe pipeline (StandardScaler -> PCA(20) -> LogisticRegression) and K = 5 are kept
    identical to the probing notebook, so codelengths from the two analyses are comparable.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xf = np.asarray(Xf, float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    Xf, y = Xf[idx], y[idx]

    chunks = np.array_split(np.arange(len(y)), K)
    L = len(chunks[0]) * 1.0                      # first chunk: no probe yet -> 1 bit/example
    for k in range(1, K):
        tr = np.concatenate(chunks[:k])
        te = chunks[k]
        if len(te) == 0:
            continue
        if len(np.unique(y[tr])) < 2:             # cannot fit a probe yet -> uniform code
            L += len(te) * 1.0
            continue
        n_comp = max(2, min(n_pca, Xf.shape[1], len(tr) - 1))
        clf = make_pipeline(StandardScaler(), PCA(n_components=n_comp),
                            LogisticRegression(max_iter=2000))
        clf.fit(Xf[tr], y[tr])                    # scaler and PCA are fitted INSIDE the fold: no leakage
        proba = clf.predict_proba(Xf[te])
        classes = list(clf.named_steps["logisticregression"].classes_)
        p = np.array([proba[i, classes.index(y[te][i])] if y[te][i] in classes else 1e-6
                      for i in range(len(te))])
        L += float(-np.log2(np.clip(p, 1e-6, 1)).sum())
    return float(L)


def space_saving(L: float, n: int) -> float:
    """SV = 1 - L(D)/L_uniform(D), with L_uniform = n bits for a balanced binary task."""
    return 1.0 - L / max(n, 1)


# The seven hierarchical band-classification tasks (Pagani et al. Fig. 1), kept for the
# descriptive cross-check. Each is (name, band_lo, band_hi, decision boundary).
BAND_TASKS = [
    ("LL", 2, 64, 33),
    ("L", 2, 126, 64),
    ("LH", 65, 126, 95),
    ("Mid", 2, 250, 126),
    ("HL", 127, 188, 157),
    ("H", 127, 250, 188),
    ("HH", 189, 250, 219),
]


# --------------------------------------------------------------------------------- #
#  5. Collapse-site detection and the derived per-geometry summaries used by H3
# --------------------------------------------------------------------------------- #
def detect_collapse_sites(freqs: np.ndarray, z: np.ndarray, pure: bool,
                          rel_thr: float = 0.02, dip_ratio: float = 0.75,
                          window_hz: float = 6.0) -> list[float]:
    """Frequencies at which the across-patch token dispersion collapses.

    Two regimes, exactly as in hypotheses.py:
      * pure sinusoid , the degeneracy is EXACT, so a site is any frequency whose dispersion is
        within `rel_thr` of zero relative to the curve's own maximum;
      * generator background, the background breaks patch identity, so the dispersion never
        reaches zero; a site is a prominent local minimum dipping to <= `dip_ratio` of its local
        baseline. The baseline window is expressed in Hz (not in samples) because the H3 sweep grid
        is non-uniform.
    """
    freqs = np.asarray(freqs, float)
    z = np.asarray(z, float)
    if pure:
        thr = rel_thr * float(np.nanmax(z))
        return [float(f) for f, v in zip(freqs, z) if v <= thr]

    sites = []
    for i in range(1, len(z) - 1):
        near = (np.abs(freqs - freqs[i]) <= window_hz) & (np.arange(len(freqs)) != i)
        if not near.any():
            continue
        local = float(np.median(z[near]))
        if z[i] <= z[i - 1] and z[i] <= z[i + 1] and local > 0 and z[i] <= dip_ratio * local:
            sites.append(float(freqs[i]))
    return sites


def merge_adjacent(sites: list[float], tol: float = 1.5) -> list[float]:
    """Collapse runs of neighbouring detections into one site (their mean).

    On the union grid a single dip can straddle two nearby grid points (e.g. 42.5 and 42.67 Hz);
    counting both would inflate the site count and corrupt the spacing estimate feeding Eq. (13).
    """
    if not sites:
        return []
    sites = sorted(sites)
    groups, cur = [], [sites[0]]
    for f in sites[1:]:
        if f - cur[-1] <= tol:
            cur.append(f)
        else:
            groups.append(cur)
            cur = [f]
    groups.append(cur)
    return [float(np.mean(g)) for g in groups]


def site_summary(sites: list[float]) -> dict:
    """The two derived statistics the H3 movement model (Eq. 14) regresses on.

    `f1` is the lowest detected site (the comb's fundamental) and `delta_hat` the median successive
    difference (its spacing). Both are read off the data WITHOUT assuming which parameter generates
    the comb, that is the point: the regression then estimates whether the spacing moves as 1/S.
    """
    sites = sorted(sites)
    if not sites:
        return {"n_sites": 0, "f1": np.nan, "delta_hat": np.nan}
    if len(sites) == 1:
        # a single in-band site is its own fundamental; the spacing is unidentified from differences
        return {"n_sites": 1, "f1": float(sites[0]), "delta_hat": float(sites[0])}
    diffs = np.diff(sites)
    return {"n_sites": len(sites), "f1": float(sites[0]), "delta_hat": float(np.median(diffs))}


def comb_distance(freqs: np.ndarray, spacing: float, band: tuple[float, float] = BAND) -> np.ndarray:
    """rho_Delta(f): distance from each frequency to the nearest member of a comb of given spacing.

    Used to label each swept frequency as on- or off-grid for the H3 site model (Eq. 12): a
    frequency counts as on the grid when this distance is within the tolerance. Written here rather
    than in the notebook so the comparison M_S / M_P / M_0 uses one definition of membership.
    """
    freqs = np.asarray(freqs, float)
    grid = np.arange(spacing, band[1] + spacing, spacing)
    grid = grid[grid >= band[0] - spacing]
    return np.min(np.abs(freqs[:, None] - grid[None, :]), axis=1)
