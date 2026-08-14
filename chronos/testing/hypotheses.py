"""
hypotheses.py — H1/H2/H3 hypothesis tests.

    python hypotheses.py                     # default (16,16), pure sinusoid
    python hypotheses.py --P 16 --S 8        # retrained variant
    python hypotheses.py --cross             # H3 collapse table + H1 spectrum, ALL models
    python hypotheses.py --background-tsm    # TSMixup background + tone
    python hypotheses.py --background-ks     # KernelSynth background + tone
"""
from __future__ import annotations

import argparse
import sys
from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import testing_lib as tl

FS = 512
CTX = 480
PRED = 64
BAND = (2, 250)
H3_MIN_FRAC = 0.6          # H3: at least this fraction of predicted stride-locks must collapse
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_BG = False
BG_GEN = "tsmixup"
TONE_SNR = 4.0
_GEN_P = 16
_bg_cache: dict = {}


def _bg_suffix() -> str:
    if not USE_BG:
        return ""
    return "_ks" if BG_GEN == "kernelsynth" else "_tsm"


def _background(n: int, seed: int) -> np.ndarray:
    p = str(Path(__file__).resolve().parent.parent / "data" / "synthetic")
    if p not in sys.path:
        sys.path.insert(0, p)
    import signalGenerator as sg
    tmp = Path(__file__).resolve().parent / "outputs" / "_gen_tmp"
    if BG_GEN == "kernelsynth":
        params = {"J": 5, "l_syn": n, "fs": FS, "jitter": 1e-4, "P": _GEN_P}
        return np.asarray(sg.runKernelSynth(params, seed, tmp).generate(), float).ravel()[:n]
    params = {"K": 10, "alpha": 1.5, "l_min": n, "l_max": n, "fs": FS, "P": _GEN_P,
              "t_lengths": [n // 2, n, n]}
    return np.asarray(sg.runTSMixup(params, seed, tmp).generate(), float).ravel()[:n]


def make_tone(f: float, phase: float = 0.0, n: int = CTX, amp: float = 1.0) -> np.ndarray:
    t = np.arange(n) / FS
    return (amp * np.sin(2 * np.pi * f * t + phase)).astype(np.float32)


def signal(f: float, phase: float = 0.0, n: int = CTX) -> np.ndarray:
    """Pure tone (default) or background + tone (USE_BG).

    The background is keyed on the phase alone, so a lock and its f_k+/-delta controls
    evaluated at the same phase share one background realisation (the paired contrast the
    deliverable requires). Model.__init__ clears the cache so it never bleeds across models."""
    if not USE_BG:
        return make_tone(f, phase, n)
    key = round(float(phase), 4)
    if key not in _bg_cache:
        b = _background(CTX + PRED, (abs(hash(key)) % (2 ** 31)) + 1)
        s = b.std()
        _bg_cache[key] = (b / s if s > 1e-8 else b).astype(np.float32)
    return (_bg_cache[key][:n] + make_tone(f, phase, n, TONE_SNR)).astype(np.float32)


def phases_Sf(f: float, n_max: int = 8) -> np.ndarray:
    """Non-redundant phase offsets (paper Eq. 6)."""
    period = FS // gcd(int(round(f)), FS)
    n_ph = int(np.clip(period - 1, 1, n_max))
    ks = np.linspace(0, period, n_ph, endpoint=False)
    return 2 * np.pi * f * ks / FS


class Model:
    """Loads a (P,S) model and exposes forecast-recovery + token-collapse."""

    def __init__(self, P: int, S: int):
        global _GEN_P
        _bg_cache.clear()                    # per-model backgrounds (generator depends on _GEN_P)
        self.pipe, self.label = tl.load_pipeline(P, S, device=DEVICE)
        cfg = self.pipe.model.config.chronos_config
        self.P = cfg["input_patch_size"]
        self.S = cfg["input_patch_stride"]
        _GEN_P = self.P
        qs = list(cfg["quantiles"])
        self.qi = qs.index(0.5) if 0.5 in qs else len(qs) // 2
        self._emb = self.pipe.model.input_patch_embedding

    @torch.no_grad()
    def forecast(self, ctx: np.ndarray) -> np.ndarray:
        x = torch.tensor(np.asarray(ctx, np.float32), device=DEVICE).unsqueeze(0)
        return self.pipe.predict(x, prediction_length=PRED)[0, self.qi].float().cpu().numpy()

    def recovery(self, f: float, phase: float = 0.0) -> tuple[float, float]:
        """Forecast amplitude recovery R and phase error [deg]."""
        ctx = signal(f, phase)
        fc = self.forecast(ctx)
        t_fut = (np.arange(CTX, CTX + len(fc))) / FS
        a_hat, ph_hat = tl.fit_amp_phase(fc, t_fut, f)
        a_true, ph_true = tl.fit_amp_phase(signal(f, phase, CTX + len(fc))[CTX:], t_fut, f)
        R = a_hat / max(a_true, 1e-9)
        dphase = np.degrees(np.abs(np.angle(np.exp(1j * (ph_hat - ph_true)))))
        return float(R), float(dphase)

    @torch.no_grad()
    def collapse(self, f: float, phase: float = 0.0) -> float:
        """Across-patch std of input-patch-embedding tokens (0 = patches identical)."""
        cap: dict = {}
        h = self._emb.register_forward_hook(lambda m, i, o: cap.__setitem__("t", o.detach().float().cpu().numpy()))
        try:
            self.forecast(signal(f, phase))
        finally:
            h.remove()
        tok = cap["t"][0]
        return float(tok.std(axis=0).mean())


# --- lock geometry ---

def stride_locks(S: int, fmax: float) -> list[float]:
    return [c * FS / S for c in range(1, int(fmax * S / FS) + 1) if c * FS / S >= BAND[0]]


def patch_nulls(P: int, fmax: float) -> list[float]:
    return [k * FS / P for k in range(1, int(fmax * P / FS) + 1)]


def half_stride_locks(S: int, fmax: float) -> list[float]:
    """(c+1/2)*fs/S: anti-periodic lock (secondary collapse dip)."""
    out, c = [], 0
    while (c + 0.5) * FS / S <= fmax:
        f = (c + 0.5) * FS / S
        if f >= BAND[0]:
            out.append(round(f, 2))
        c += 1
    return out


def lock_frequencies(P: int, S: int, fmax: float) -> list[float]:
    """Union of stride + patch locks, merged when closer than 0.5 Hz."""
    fam = ([c * FS / S for c in range(1, int(fmax * S / FS) + 1)]
           + [k * FS / P for k in range(1, int(fmax * P / FS) + 1)])
    out: list[float] = []
    for f in sorted(f for f in fam if BAND[0] <= f <= fmax):
        if not out or f - out[-1] > 0.5:
            out.append(f)
    return out


def _controls_clean(fk: float, delta: float, locks: list[float], tol: float = 0.5) -> bool:
    """True if both controls f_k +/- delta lie in-band AND off every *other* F_lock member.

    H1's contrast compares recovery at a lock against its f_k+/-delta neighbours; if a
    neighbour is itself an F_lock member the clean baseline is contaminated (a lock-vs-lock
    contrast), so that lock is dropped. With delta = 0.25 f_s/S this bites only at p16-s12,
    where the patch grid (32 Hz) and stride grid (42.6..Hz) interleave so each control lands
    on the other family -> only {64,128,192} survive (its own stride locks are not H1-testable
    at this delta; a one-sided control would be needed to recover them)."""
    others = [l for l in locks if abs(l - fk) > tol]
    return all(BAND[0] <= c <= BAND[1] and not any(abs(c - l) <= tol for l in others)
               for c in (fk - delta, fk + delta))


def _fmt(f: float) -> str:
    return f"{f:.0f}" if abs(f - round(f)) < 0.05 else f"{f:.1f}"


def _sweep_grid(extra: list[float]) -> np.ndarray:
    """0.1 Hz grid with exact lock frequencies injected."""
    base = np.arange(BAND[0], BAND[1] + 0.05, 0.1)
    return np.union1d(base, [f for f in extra if BAND[0] <= f <= BAND[1]])


def _is_collapse(v: float, f: float, freqs: np.ndarray, coll: np.ndarray, tau: float = 0.75) -> bool:
    """True if the token collapse at f drops at least (1-tau) below the local baseline
    (median over +/-12 Hz, excluding the point). One criterion for pure and background:
    a pure lock reaches ~0 and passes trivially; a background lock must still dip >=25%."""
    i = int(np.argmin(np.abs(freqs - f)))
    hw = int(round(12 / 0.1))
    lo, hi = max(0, i - hw), min(len(coll), i + hw + 1)
    neigh = np.concatenate([coll[lo:i], coll[i + 1:hi]])
    if neigh.size == 0:
        return False
    local = float(np.median(neigh))
    return local > 1e-9 and v <= tau * local


# --- H3: do the collapse sites sit where the geometry predicts? ---

def test_H3(m: Model, out: Path, n_phase: int = 3) -> dict:
    all_stride = stride_locks(m.S, BAND[1])
    half = half_stride_locks(m.S, BAND[1])
    pred_patch = [f for f in patch_nulls(m.P, BAND[1]) if BAND[0] <= f <= BAND[1]]

    freqs = _sweep_grid(all_stride + half + pred_patch)
    coll = np.array([np.mean([m.collapse(f, ph) for ph in phases_Sf(f, n_phase)]) for f in freqs])

    def _coll_at(f: float) -> float:                     # reuse the swept curve (every lock f is on the grid)
        return float(coll[int(np.argmin(np.abs(freqs - f)))])
    lock_val = {round(f, 1): _coll_at(f) for f in all_stride}
    predicted = sorted(lock_val)
    measured = [f for f in predicted if _is_collapse(lock_val[f], f, freqs, coll)]
    frac = len(measured) / len(predicted) if predicted else 0.0
    passed = len(predicted) > 0 and frac >= H3_MIN_FRAC

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(freqs, coll, color="#c62828", lw=1.4, zorder=3)
    for j, f in enumerate(all_stride):
        ax.axvline(f, color="#1b5e20", ls="-", lw=1.2, alpha=.6, zorder=1,
                   label="c·fs/S" if j == 0 else None)
    for j, f in enumerate(half):
        ax.axvline(f, color="#ef6c00", ls=":", lw=1.2, alpha=.7, zorder=2,
                   label="(c+½)·fs/S" if j == 0 else None)
    for j, f in enumerate(pred_patch):
        ax.axvline(f, color="#1565c0", ls=(0, (4, 3)), lw=1.0, alpha=.9, zorder=5,
                   label="k·fs/P" if j == 0 else None)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("token collapse (std)", fontsize=9)
    ax.set_xticks(list(range(32, int(BAND[1]) + 1, 32)))
    ax.set_title(f"H3 — token collapse — {m.label}  [{len(measured)}/{len(predicted)} sites]", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", ncol=3); ax.margins(x=0.01); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / "H3_collapse_sites.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    return dict(predicted=predicted, measured=measured, patch_nulls=pred_patch,
                half_stride=half, passed=passed, frac=frac)


# --- H1: is the loss localized at the lock? ---

def test_H1(m: Model, out: Path, n_phase: int = 6) -> dict:
    delta = 0.25 * FS / m.S
    locks_all = lock_frequencies(m.P, m.S, BAND[1])
    locks = [f for f in locks_all
             if f - delta >= BAND[0] and _controls_clean(f, delta, locks_all)]
    n_dropped = sum(1 for f in locks_all
                    if f - delta >= BAND[0] and not _controls_clean(f, delta, locks_all))

    rows = []
    for fk in locks:
        phs = phases_Sf(fk, n_phase)                     # shared phase set: lock & controls paired
        def mean_R(f):
            return float(np.mean([m.recovery(f, ph)[0] for ph in phs]))
        Rk, Rlo, Rhi = mean_R(fk), mean_R(fk - delta), mean_R(fk + delta)
        d = np.log(Rk + 0.01) - 0.5 * (np.log(Rlo + 0.01) + np.log(Rhi + 0.01))
        live = max(Rk, Rlo, Rhi) > 0.05
        rows.append(dict(fk=fk, cpp=fk * m.P / FS, R_lock=Rk, R_ctrl=0.5 * (Rlo + Rhi),
                         collapse=m.collapse(fk), d=float(d), live=live))

    d_live = np.array([r["d"] for r in rows if r["live"]])
    frac_neg = float(np.mean(d_live < 0)) if len(d_live) else 0.0
    passed = len(d_live) > 0 and frac_neg >= 0.5 and d_live.mean() < 0

    sl = stride_locks(m.S, BAND[1])
    pn = [f for f in patch_nulls(m.P, BAND[1]) if BAND[0] <= f <= BAND[1]]
    sweep_freqs = _sweep_grid(sl + pn)
    R_curve = np.array([np.mean([m.recovery(f, ph)[0]
                                 for ph in phases_Sf(f, min(n_phase, 3))]) for f in sweep_freqs])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sweep_freqs, R_curve, color="#c62828", lw=1.5, zorder=3, label="forecast recovery")
    for j, f in enumerate(sl):
        ax.axvline(f, color="#1b5e20", lw=1.2, alpha=.5, zorder=1,
                   label="stride lock c·fs/S" if j == 0 else None)
    for j, f in enumerate(pn):
        ax.axvline(f, color="#1565c0", ls=(0, (4, 3)), lw=1.0, alpha=.85, zorder=4,
                   label="patch null k·fs/P" if j == 0 else None)
    ax.axhline(1, color="grey", ls="--", lw=.6)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("amplitude recovery")
    ax.set_ylim(0, 1.25)
    ax.set_xticks(list(range(32, int(BAND[1]) + 1, 32)))
    ax.set_title(f"H1 — forecast recovery — {m.label}", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.margins(x=0.01); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / "H1_local_contrast.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    return dict(rows=rows, frac_neg=frac_neg, mean_d=float(d_live.mean()) if len(d_live) else 0.0,
                n_live=int(len(d_live)), passed=passed, delta=delta, n_dropped=n_dropped)


# --- H2: is the lock deficit constant across the signal's phase? ---

def test_H2(m: Model, out: Path, n_phase: int = 10) -> dict:
    delta = 0.25 * FS / m.S
    locks_all = lock_frequencies(m.P, m.S, BAND[1])
    locks = [f for f in locks_all
             if f - delta >= BAND[0] and _controls_clean(f, delta, locks_all)]

    rows, spreads = [], []
    for fk in locks:
        phs = phases_Sf(fk, n_phase)
        lock_R = np.array([m.recovery(fk, p)[0] for p in phs])
        ctrl_R = np.array([0.5 * (m.recovery(fk - delta, p)[0] + m.recovery(fk + delta, p)[0]) for p in phs])
        if max(float(ctrl_R.mean()), float(lock_R.mean())) <= 0.05:
            continue
        deficits = ctrl_R - lock_R
        rows.append(dict(fk=fk, phs=phs, lock_R=lock_R, ctrl_R=ctrl_R, deficits=deficits))
        spreads.append(float(deficits.std()))
        if len(rows) >= 6:
            break

    mean_abs = np.mean([np.abs(r["deficits"]).mean() for r in rows]) if rows else 0.0
    mean_spread = float(np.mean(spreads)) if spreads else 0.0
    cv = mean_spread / max(mean_abs, 1e-9)
    passed = cv < 0.5

    ncols = min(len(rows), 3) if rows else 1
    nrows_fig = max(1, (len(rows) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows_fig, ncols, figsize=(4.5 * ncols, 3.6 * nrows_fig),
                             squeeze=False)
    for i, r in enumerate(rows):
        ax = axes[i // ncols][i % ncols]
        xdeg = np.degrees(r["phs"]) % 360
        o = np.argsort(xdeg)
        xdeg, lockR, ctrlR = xdeg[o], r["lock_R"][o], r["ctrl_R"][o]
        gap_mu, gap_sd = float(r["deficits"].mean()), float(r["deficits"].std())
        ax.plot(xdeg, lockR, "o", color="#c62828", ms=6, label="lock")
        ax.plot(xdeg, ctrlR, "o", color="#9e9e9e", ms=6, label="control f_k±δ")
        ax.axhline(lockR.mean(), color="#c62828", ls="--", lw=1.0)
        ax.axhline(ctrlR.mean(), color="#9e9e9e", ls="--", lw=1.0)
        ax.set_xlim(-15, 360)
        ax.set_title(f"{_fmt(r['fk'])} Hz   gap={gap_mu:+.3f}±{gap_sd:.3f}", fontsize=9)
        ax.set_xlabel("phase [deg]")
        ax.grid(alpha=.15)
    for i in range(len(rows), nrows_fig * ncols):
        axes[i // ncols][i % ncols].set_visible(False)
    if rows:
        axes[0][0].set_ylabel("recovery")
        axes[0][0].legend(fontsize=8, loc="upper right", ncol=2)
    verdict = "SUPPORTED" if passed else "REFUTED"
    fig.suptitle(f"H2 — phase invariance — {m.label}    CV={cv:.2f} → {verdict}", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "H2_phase_invariance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    return dict(rows=[{"fk": r["fk"], "spread": s} for r, s in zip(rows, spreads)],
                mean_spread=mean_spread, mean_abs=mean_abs, cv=cv, passed=passed)


# --- cross-model analyses ---

def cross_collapse_table(n_phase: int = 3) -> None:
    global USE_BG
    bg = USE_BG
    root = Path("outputs/hypotheses"); root.mkdir(parents=True, exist_ok=True)
    suf = _bg_suffix()
    lines = ["model,P,S,stride_locks,collapsed_freqs"]
    mode = "KernelSynth bg" if BG_GEN == "kernelsynth" and USE_BG else ("TSMixup bg" if USE_BG else "pure sinusoid")
    print(f"\n=== H3 across geometry [{mode}]: collapse sites follow c·fs/S ===")
    spectra = []
    for (P, S) in tl.ALL_MODELS:
        m = Model(P, S)
        all_stride = stride_locks(m.S, BAND[1])
        half = half_stride_locks(m.S, BAND[1])
        pred_patch = [f for f in patch_nulls(m.P, BAND[1]) if BAND[0] <= f <= BAND[1]]
        freqs = _sweep_grid(all_stride + half + pred_patch)
        coll = np.array([np.mean([m.collapse(f, ph) for ph in phases_Sf(f, n_phase)]) for f in freqs])
        ref_coll = None
        if bg:
            USE_BG = False
            ref_coll = np.array([np.mean([m.collapse(f, ph) for ph in phases_Sf(f, n_phase)]) for f in freqs])
            USE_BG = True
        pred = [round(f, 1) for f in all_stride]
        def _coll_at(f: float) -> float:                 # reuse the swept curve (every lock f is on the grid)
            return float(coll[int(np.argmin(np.abs(freqs - f)))])
        measured = [round(f, 1) for f in all_stride
                    if _is_collapse(_coll_at(f), f, freqs, coll)]
        print(f"  p{m.P}-s{m.S:<3} stride-locks {pred}  ->  collapsed {measured}  [{len(measured)}/{len(pred)}]")
        lines.append(f"p{m.P}-s{m.S},{m.P},{m.S},{' '.join(map(str,pred))},{' '.join(map(str,measured))}")
        spectra.append((f"p{m.P}-s{m.S}", m.S, freqs, coll, measured, ref_coll))
    (root / f"collapse_sites_all_models{suf}.csv").write_text("\n".join(lines), encoding="utf-8")

    fig, axes = plt.subplots(len(spectra), 1, figsize=(12, 1.8 * len(spectra)), sharex=True)
    for (ax, (label, S, fq, coll, meas, ref_coll)), (P, _S) in zip(zip(axes, spectra), tl.ALL_MODELS):
        if ref_coll is not None:
            ax.plot(fq, ref_coll, color="#9e9e9e", lw=1.0, alpha=.5, zorder=2)
        ax.plot(fq, coll, color="#c62828", lw=1.3, zorder=3)
        for f in stride_locks(S, BAND[1]):
            ax.axvline(f, color="#1b5e20", ls="-", lw=1.0, alpha=.5, zorder=1)
        for f in half_stride_locks(S, BAND[1]):
            ax.axvline(f, color="#ef6c00", ls=":", lw=1.1, alpha=.6, zorder=1)
        for f in patch_nulls(P, BAND[1]):
            ax.axvline(f, color="#1565c0", ls=(0, (4, 3)), lw=1.0, alpha=.8, zorder=4)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.01); ax.grid(alpha=.15)
    axes[0].plot([], [], color="#1b5e20", lw=1.2, label="c·fs/S")
    axes[0].plot([], [], color="#1565c0", ls=(0, (4, 3)), lw=1.2, label="k·fs/P")
    axes[0].plot([], [], color="#ef6c00", ls=":", lw=1.3, label="(c+½)·fs/S")
    if bg: axes[0].plot([], [], color="#9e9e9e", lw=1.2, label="pure (ref)")
    axes[0].legend(fontsize=8, loc="upper right", ncol=4)
    axes[0].set_title("H3 — token collapse", fontsize=11)
    axes[-1].set_xlabel("frequency [Hz]")
    axes[-1].set_xticks(list(range(32, int(BAND[1]) + 1, 32)))
    fig.tight_layout(); fig.savefig(root / f"H3_collapse_sites_all_models{suf}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {root/('collapse_sites_all_models'+suf+'.csv')}")
    print(f"  saved -> {root/('H3_collapse_sites_all_models'+suf+'.png')}")


def cross_H1(n_phase: int = 3) -> None:
    global USE_BG
    root = Path("outputs/hypotheses"); root.mkdir(parents=True, exist_ok=True)
    bg = USE_BG; suf = _bg_suffix()
    all_locks = []
    for P, S in tl.ALL_MODELS:
        all_locks += stride_locks(S, BAND[1])
        all_locks += [f for f in patch_nulls(P, BAND[1]) if BAND[0] <= f <= BAND[1]]
    freqs = _sweep_grid(all_locks)
    mode = "KernelSynth bg" if BG_GEN == "kernelsynth" and USE_BG else ("TSMixup bg" if USE_BG else "pure sinusoid")
    print(f"\n=== H1 recovery spectrum [{mode}] ===")
    rows = []
    for (P, S) in tl.ALL_MODELS:
        m = Model(P, S)
        def curve():
            return np.array([np.mean([m.recovery(f, ph)[0] for ph in phases_Sf(f, n_phase)]) for f in freqs])
        USE_BG = bg;    R = curve()
        ref = None
        if bg:
            USE_BG = False; ref = curve(); USE_BG = True
        sl = stride_locks(m.S, BAND[1])
        pn = [f for f in patch_nulls(m.P, BAND[1]) if BAND[0] <= f <= BAND[1]]
        rows.append((f"p{m.P}-s{m.S}", R, ref, sl, pn))
    USE_BG = bg

    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.7 * len(rows)), sharex=True)
    for ax, (label, R, ref, sl, pn) in zip(axes, rows):
        if ref is not None:
            ax.plot(freqs, ref, color="#9e9e9e", lw=1.0, alpha=.55, zorder=2)
        ax.plot(freqs, R, color="#c62828", lw=1.3, zorder=3)
        for f in sl: ax.axvline(f, color="#1b5e20", lw=1.0, alpha=.5, zorder=1)
        for f in pn: ax.axvline(f, color="#1565c0", ls=(0, (4, 3)), lw=1.0, alpha=.85, zorder=4)
        ax.axhline(1, color="grey", ls="--", lw=.6)
        ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
        ax.margins(x=0.01); ax.grid(alpha=.15)
    axes[0].plot([], [], color="#1b5e20", label="c·fs/S")
    axes[0].plot([], [], color="#1565c0", ls=(0, (4, 3)), label="k·fs/P")
    if bg: axes[0].plot([], [], color="#9e9e9e", label="pure (ref)")
    axes[0].legend(fontsize=8, loc="upper right", ncol=3)
    axes[0].set_title("H1 — forecast recovery", fontsize=11)
    axes[-1].set_xlabel("frequency [Hz]")
    axes[-1].set_xticks(list(range(32, int(BAND[1]) + 1, 32)))
    fig.tight_layout(); fig.savefig(root / f"H1_local_contrast_all_models{suf}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {root/('H1_local_contrast_all_models'+suf+'.png')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--cross", action="store_true")
    ap.add_argument("--background-tsm", action="store_true",
                    help="tone on TSMixup background; outputs to *_tsm paths")
    ap.add_argument("--background-ks", action="store_true",
                    help="tone on KernelSynth background; outputs to *_ks paths")
    ap.add_argument("--only-h2", action="store_true",
                    help="run only H2 (skip H1/H3)")
    args = ap.parse_args()

    global USE_BG, BG_GEN
    if args.background_ks:
        USE_BG, BG_GEN = True, "kernelsynth"
    elif args.background_tsm:
        USE_BG, BG_GEN = True, "tsmixup"

    mode = "KernelSynth bg" if BG_GEN == "kernelsynth" and USE_BG else ("TSMixup bg" if USE_BG else "pure sinusoid")

    if args.cross:
        cross_collapse_table()
        cross_H1()
        return

    m = Model(args.P, args.S)
    suf = _bg_suffix()
    out = Path(f"outputs/hypotheses/p{m.P}-s{m.S}{suf}"); out.mkdir(parents=True, exist_ok=True)
    print(f"loaded: {m.label}  (P={m.P}, S={m.S})  device={DEVICE}  | signal mode: {mode}")

    if args.only_h2:
        h2 = test_H2(m, out)
        print(f"H2 (phase-invariant deficit): {'SUPPORTED' if h2['passed'] else 'REFUTED'}")
        print(f"     CV={h2['cv']:.2f} (spread={h2['mean_spread']:.3f}, |deficit|={h2['mean_abs']:.3f})")
        return

    h3 = test_H3(m, out)
    h1 = test_H1(m, out)
    h2 = test_H2(m, out)

    print("\n" + "=" * 68)
    print(f"VERDICTS  —  {m.label}")
    print("=" * 68)
    print(f"H3 (sites match geometry): {'PASS' if h3['passed'] else 'FAIL'}"
          f"  [{len(h3['measured'])}/{len(h3['predicted'])} sites]")
    print(f"     predicted {h3['predicted']}")
    print(f"     measured  {h3['measured']}")
    print(f"H1 (localized loss at lock): {'SUPPORTED' if h1['passed'] else 'REFUTED'}")
    print(f"     among {h1['n_live']} live locks: d<0 at {h1['frac_neg']*100:.0f}%; "
          f"mean d={h1['mean_d']:+.2f} (delta={h1['delta']:.1f} Hz)"
          + (f"; {h1['n_dropped']} lock(s) dropped (control on F_lock)" if h1['n_dropped'] else ""))
    for r in h1["rows"]:
        tag = "dead-zone" if not r["live"] else ("loss" if r["d"] < 0 else "no-loss")
        print(f"     {r['fk']:>4} Hz cpp{r['cpp']:.2f}: R_lock={r['R_lock']:.2f} "
              f"R_ctrl={r['R_ctrl']:.2f} collapse={r['collapse']:.3f}  d={r['d']:+.2f} [{tag}]")
    print(f"H2 (phase-invariant deficit): {'SUPPORTED' if h2['passed'] else 'REFUTED'}")
    print(f"     phase-spread CV={h2['cv']:.2f} (mean spread={h2['mean_spread']:.3f}, "
          f"mean |deficit|={h2['mean_abs']:.3f})")
    print("=" * 68)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
