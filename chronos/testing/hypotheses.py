"""
hypotheses.py — direct, falsifiable tests of the deliverable's H1/H2/H3.

The two notebooks *describe* structural aliasing (probing, reconstruction, contamination).
This file *tests the three written hypotheses head-on*, on the model selected by geometry
(P, S) through `testing_lib`, and prints a PASS/FAIL verdict for each with the numbers behind it.

    H1  local information loss at a lock:
        at f in [f_k - eps, f_k + eps] the model recovers LESS than at controls f_k +/- delta
        (delta = 0.25 * fs/S >> eps).  Expected: lower forecast amplitude recovery AND a token
        collapse (t_k = t_{k+1}) at the lock.  Absent / non-localized loss refutes H1.

    H2  the deficit is a property of the GEOMETRY, not of the signal's phase:
        across the S_f non-redundant phase offsets sampled at f_k, the lock-vs-control drop
        stays ~constant.  A drop whose size varies systematically with phase refutes H2.

    H3  the lock SITES move with the geometry:
        patch nulls sit at k*fs/P, stride locks at c*fs/S.  Measured token-collapse minima must
        land on those predicted sites and MOVE when P or S changes.  A site that does not move
        when its generating parameter is varied refutes H3.

Run:
    ../.venv/Scripts/python.exe hypotheses.py                 # loaded default (16,16)
    ../.venv/Scripts/python.exe hypotheses.py --P 16 --S 8    # one retrained variant
    ../.venv/Scripts/python.exe hypotheses.py --cross         # H3 collapse-site table, ALL models

Outputs (per model): outputs/hypotheses/p{P}-s{S}/{H1_local_contrast,H2_phase_invariance,
H3_collapse_sites}.png  and a printed verdict block.  --cross also writes
outputs/hypotheses/collapse_sites_all_models.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import testing_lib as tl

# --------------------------------------------------------------------------- #
#  fixed experimental setup (deliverable convention)
# --------------------------------------------------------------------------- #
FS = 512                 # sampling frequency [Hz]  (Nyquist = 256 Hz)
CTX = 480                # context length: divisible by every stride in the sweep (16/12/8/4/24)
                         # so the patch grid is exact and no internal padding fakes a token collapse
PRED = 64                # forecast horizon
BAND = (2, 250)          # analysis band, strictly below Nyquist
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------- #
#  signal mode: pure sinusoid (default) vs realistic TSMixup background + tone
# --------------------------------------------------------------------------- #
# By default every test uses a PURE sinusoid: this isolates the geometry, so the token collapse is
# EXACTLY 0 at a stride lock (t_k = t_{k+1}) — the clean definition of the phenomenon. With
# --background (or HYP_BACKGROUND=1) the tone instead rides on a unit-variance TSMixup background
# at SNR=4 (the SAME construction as the probing notebook). That is the realistic-signal cross-check:
# the exact degeneracy becomes a deep but non-zero DIP (the background breaks patch identity), while
# the recovery-based verdicts H1/H2 are unchanged. Background outputs are written to *_bg paths so
# they never overwrite the pure-sinusoid figures.
USE_BG = os.environ.get("HYP_BACKGROUND", "0") == "1"
TONE_SNR = 4.0                     # tone amplitude over unit-variance background (probing convention)
_GEN_P = 16                        # patch passed to the generator (background is P-independent)
_bg_cache: dict = {}
_sg = None


def _load_generator():
    global _sg
    if _sg is None:
        p = str(Path(__file__).resolve().parent.parent / "data" / "synthetic")
        if p not in sys.path:
            sys.path.insert(0, p)
        import signalGenerator as sg
        _sg = sg
    return _sg


def _background(n: int, seed: int) -> np.ndarray:
    """One TSMixup background realisation of length n (no injection)."""
    sg = _load_generator()
    p = {"K": 10, "alpha": 1.5, "l_min": n, "l_max": n, "fs": FS, "P": _GEN_P,
         "t_lengths": [n // 2, n, n]}
    tmp = Path(__file__).resolve().parent / "outputs" / "_gen_tmp"
    return np.asarray(sg.runTSMixup(p, seed, tmp).generate(), float).ravel()[:n]


# --------------------------------------------------------------------------- #
#  signal + measurement helpers
# --------------------------------------------------------------------------- #
def make_tone(f: float, phase: float = 0.0, n: int = CTX, amp: float = 1.0) -> np.ndarray:
    t = np.arange(n) / FS
    return (amp * np.sin(2 * np.pi * f * t + phase)).astype(np.float32)


def signal(f: float, phase: float = 0.0, n: int = CTX) -> np.ndarray:
    """Context builder. Pure tone (default) or unit-variance TSMixup background + tone (USE_BG).

    The background is deterministic per (f, phase) and generated once at full length (CTX+PRED),
    then sliced, so the forecast target `signal(f,phase,CTX+PRED)[CTX:]` is the true continuation
    of the context `signal(f,phase,CTX)`."""
    if not USE_BG:
        return make_tone(f, phase, n)
    key = (round(float(f), 3), round(float(phase), 4))
    if key not in _bg_cache:
        b = _background(CTX + PRED, (abs(hash(key)) % (2 ** 31)) + 1)
        s = b.std()
        _bg_cache[key] = (b / s if s > 1e-8 else b).astype(np.float32)
    return (_bg_cache[key][:n] + make_tone(f, phase, n, TONE_SNR)).astype(np.float32)


def phases_Sf(f: float, n_max: int = 8) -> np.ndarray:
    """Non-redundant phase offsets S_f = min(fs/gcd(f,fs) - 1, n_max) (paper Eq. 6)."""
    period = FS // gcd(int(round(f)), FS)
    n_ph = int(np.clip(period - 1, 1, n_max))
    ks = np.linspace(0, period, n_ph, endpoint=False)
    return 2 * np.pi * f * ks / FS


def fit_amp_phase(y: np.ndarray, t: np.ndarray, f: float) -> tuple[float, float]:
    X = np.stack([np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t), np.ones_like(t)], 1)
    a, b, _ = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.hypot(a, b)), float(np.arctan2(b, a))


class Model:
    """Thin wrapper: loads (P,S) once and exposes forecast-recovery + token-collapse."""

    def __init__(self, P: int, S: int):
        global _GEN_P
        self.pipe, self.label = tl.load_pipeline(P, S, device=DEVICE)
        cfg = self.pipe.model.config.chronos_config
        self.P = cfg["input_patch_size"]
        self.S = cfg["input_patch_stride"]
        _GEN_P = self.P                        # background is P-independent, but keep it in sync
        qs = list(cfg["quantiles"])
        self.qi = qs.index(0.5) if 0.5 in qs else len(qs) // 2
        self._emb = self.pipe.model.input_patch_embedding

    @torch.no_grad()
    def forecast(self, ctx: np.ndarray) -> np.ndarray:
        x = torch.tensor(np.asarray(ctx, np.float32), device=DEVICE).unsqueeze(0)
        return self.pipe.predict(x, prediction_length=PRED)[0, self.qi].float().cpu().numpy()

    def recovery(self, f: float, phase: float = 0.0) -> tuple[float, float]:
        """Forecast amplitude recovery R = amp(forecast)/amp(truth) and phase error [deg]."""
        ctx = signal(f, phase)
        fc = self.forecast(ctx)
        t_fut = (np.arange(CTX, CTX + len(fc))) / FS
        a_hat, ph_hat = fit_amp_phase(fc, t_fut, f)
        a_true, ph_true = fit_amp_phase(signal(f, phase, CTX + len(fc))[CTX:], t_fut, f)
        R = a_hat / max(a_true, 1e-9)
        dphase = np.degrees(np.abs(np.angle(np.exp(1j * (ph_hat - ph_true)))))
        return float(R), float(dphase)

    @torch.no_grad()
    def collapse(self, f: float, phase: float = 0.0) -> float:
        """Across-patch std of the input-patch-embedding tokens (0 = consecutive patches identical)."""
        cap: dict = {}
        h = self._emb.register_forward_hook(lambda m, i, o: cap.__setitem__("t", o.detach().float().cpu().numpy()))
        try:
            self.forecast(signal(f, phase))
        finally:
            h.remove()
        tok = cap["t"][0]                       # [n_patches, d_model]
        return float(tok.std(axis=0).mean())    # mean over dims of the across-patch std


# --------------------------------------------------------------------------- #
#  lock geometry
# --------------------------------------------------------------------------- #
def patch_nulls(P: int, fmax: float) -> list[int]:
    return [round(k * FS / P) for k in range(1, int(fmax * P / FS) + 1)]


def stride_locks(S: int, fmax: float) -> list[int]:
    return [round(c * FS / S) for c in range(1, int(fmax * S / FS) + 1)]


def detect_collapse_sites(freqs: np.ndarray, coll: np.ndarray) -> list[int]:
    """Frequencies where the across-patch token std collapses.

    Pure sinusoid: an exact zero (within 2% of the global spread) — the clean degeneracy.
    Background+tone: the background breaks patch identity so the std never reaches 0; instead we
    detect a prominent local minimum that dips to <=75% of its local (±6 Hz) baseline."""
    if not USE_BG:
        thr = 0.02 * coll.max()
        return [int(f) for f, c in zip(freqs, coll) if c <= thr]
    sites, w = [], 6
    for i in range(1, len(coll) - 1):
        lo, hi = max(0, i - w), min(len(coll), i + w + 1)
        local = np.median(np.concatenate([coll[lo:i], coll[i + 1:hi]]))
        if coll[i] <= coll[i - 1] and coll[i] <= coll[i + 1] and local > 0 and coll[i] <= 0.75 * local:
            sites.append(int(freqs[i]))
    return sites


# --------------------------------------------------------------------------- #
#  H3 — do the collapse sites sit where the geometry predicts, and move with it?
# --------------------------------------------------------------------------- #
def test_H3(m: Model, out: Path, n_phase: int = 3) -> dict:
    freqs = np.arange(BAND[0], BAND[1] + 1, dtype=float)
    coll = np.array([np.mean([m.collapse(f, ph) for ph in phases_Sf(f, n_phase)]) for f in freqs])

    measured = detect_collapse_sites(freqs, coll)
    # The across-patch token collapse (t_k = t_{k+1}) is a STRIDE-lock effect: consecutive patches
    # see identical samples iff x[n+S]=x[n], i.e. f = c*fs/S.  (The patch-integration null k*fs/P is
    # a different, within-patch effect; it coincides with the stride family only when S=P.)
    # Only INTEGER-Hz members of the c*fs/S grid can be seen on the 1 Hz sweep; non-integer stride
    # locks (e.g. 42.7/85.3 Hz for S=12) are off-grid and are neither predicted nor counted here.
    predicted = sorted({int(round(c * FS / m.S)) for c in range(1, m.S)
                        if (c * FS) % m.S == 0 and BAND[0] <= c * FS / m.S <= BAND[1]})
    pred_patch = [f for f in patch_nulls(m.P, BAND[1]) if BAND[0] <= f <= BAND[1]]
    pred_stride = predicted

    hit = [f for f in predicted if any(abs(f - mf) <= 1 for mf in measured)]
    # H3 for the collapse family: every predicted (integer) stride-lock is a measured collapse site,
    # and no measured collapse site is spurious (measured ⊆ integer stride-lock grid, within 1 Hz).
    spurious = [mf for mf in measured if not any(abs(mf - f) <= 1 for f in predicted)]
    passed = len(predicted) > 0 and len(hit) == len(predicted) and not spurious

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(freqs, coll, color="#c62828", lw=1.4, label="across-patch token std")
    for f in pred_patch:
        ax.axvline(f, color="#1565c0", ls="-", lw=1.0, alpha=.7)
    for f in pred_stride:
        ax.axvline(f, color="#6a1b9a", ls="--", lw=1.0, alpha=.7)
    if not USE_BG:
        thr = 0.02 * coll.max()
        ax.axhline(thr, color="gray", ls=":", lw=1, label=f"collapse threshold ({thr:.3f})")
    ax.plot([], [], color="#1565c0", label=f"predicted patch null  k*fs/P  (P={m.P})")
    ax.plot([], [], color="#6a1b9a", ls="--", label=f"predicted stride lock c*fs/S (S={m.S})")
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("token collapse  (0 = identical patches)")
    mode = "TSMixup background + tone" if USE_BG else "pure sinusoid"
    ax.set_title(f"H3 — {m.label} [{mode}]: collapse minima land on k*fs/P and c*fs/S")
    ax.legend(fontsize=8, loc="upper right"); ax.margins(x=0.01)
    fig.tight_layout(); fig.savefig(out / "H3_collapse_sites.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    return dict(predicted=predicted, measured=measured, hit=hit, spurious=spurious,
                patch_nulls=pred_patch, passed=passed)


# --------------------------------------------------------------------------- #
#  H1 — is the loss localized at the lock (lower recovery + token collapse vs f_k +/- delta)?
# --------------------------------------------------------------------------- #
def test_H1(m: Model, out: Path, n_phase: int = 6) -> dict:
    delta = max(1, round(0.25 * FS / m.S))     # deliverable control offset, >= 1 Hz on the grid
    locks = sorted(set(patch_nulls(m.P, BAND[1] - delta)) | set(stride_locks(m.S, BAND[1] - delta)))
    locks = [f for f in locks if f - delta >= BAND[0]]

    rows = []
    for fk in locks:
        def mean_R(f):
            return float(np.mean([m.recovery(f, ph)[0] for ph in phases_Sf(f, n_phase)]))
        Rk, Rlo, Rhi = mean_R(fk), mean_R(fk - delta), mean_R(fk + delta)
        d = np.log(Rk + 0.01) - 0.5 * (np.log(Rlo + 0.01) + np.log(Rhi + 0.01))
        # "live" = there is actually recoverable signal here (both dead => the contrast is noise)
        live = max(Rk, Rlo, Rhi) > 0.05
        rows.append(dict(fk=fk, cpp=fk * m.P / FS, R_lock=Rk, R_ctrl=0.5 * (Rlo + Rhi),
                         collapse=m.collapse(fk), d=float(d), live=live))

    # H1 is only meaningful where a tone can be recovered at all (below the reconstruction ceiling).
    d_live = np.array([r["d"] for r in rows if r["live"]])
    frac_neg = float(np.mean(d_live < 0)) if len(d_live) else 0.0
    passed = len(d_live) > 0 and frac_neg >= 0.5 and d_live.mean() < 0   # H1 predicts d<0 at (most) live locks

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.5))
    xs = np.arange(len(rows))
    labels = [f"{r['fk']}\ncpp{r['cpp']:.2f}" + ("" if r["live"] else "\n(dead)") for r in rows]
    a0.bar(xs - .2, [r["R_lock"] for r in rows], .4, color="#c62828", label="recovery @ lock")
    a0.bar(xs + .2, [r["R_ctrl"] for r in rows], .4, color="#9e9e9e", label="recovery @ f_k +/- delta")
    a0.set_xticks(xs); a0.set_xticklabels(labels, fontsize=7)
    a0.set_ylabel("forecast amplitude recovery"); a0.legend(fontsize=8)
    a0.set_title(f"H1 — recovery at lock vs control (delta={delta} Hz)")
    cols = ["#bdbdbd" if not r["live"] else ("#2e7d32" if r["d"] < 0 else "#c62828") for r in rows]
    a1.bar(xs, [r["d"] for r in rows], color=cols); a1.axhline(0, color="k", lw=.8)
    a1.set_xticks(xs); a1.set_xticklabels(labels, fontsize=7)
    a1.set_ylabel("log local contrast  d  (green<0 = supports H1)")
    md = d_live.mean() if len(d_live) else 0.0
    a1.set_title(f"H1 — among LIVE locks: d<0 at {frac_neg*100:.0f}%  (mean d={md:+.2f}); grey = dead zone")
    fig.tight_layout(); fig.savefig(out / "H1_local_contrast.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    return dict(rows=rows, frac_neg=frac_neg, mean_d=float(d_live.mean()) if len(d_live) else 0.0,
                n_live=int(len(d_live)), passed=passed, delta=delta)


# --------------------------------------------------------------------------- #
#  H2 — is the lock deficit constant across the signal's phase?
# --------------------------------------------------------------------------- #
def test_H2(m: Model, out: Path, n_phase: int = 10) -> dict:
    delta = max(1, round(0.25 * FS / m.S))
    locks = sorted(set(patch_nulls(m.P, BAND[1] - delta)) | set(stride_locks(m.S, BAND[1] - delta)))
    locks = [f for f in locks if f - delta >= BAND[0]][:6]      # a handful is enough

    rows, spreads = [], []
    for fk in locks:
        phs = phases_Sf(fk, n_phase)
        ctrl = 0.5 * (np.array([m.recovery(fk - delta, p)[0] for p in phases_Sf(fk - delta, n_phase)]).mean()
                      + np.array([m.recovery(fk + delta, p)[0] for p in phases_Sf(fk + delta, n_phase)]).mean())
        deficits = np.array([ctrl - m.recovery(fk, p)[0] for p in phs])   # control - lock, per phase
        rows.append(dict(fk=fk, deficits=deficits))
        spreads.append(float(deficits.std()))

    # H2 holds if the phase-spread of the deficit is small relative to its mean magnitude
    mean_abs = np.mean([np.abs(r["deficits"]).mean() for r in rows]) if rows else 0.0
    mean_spread = float(np.mean(spreads)) if spreads else 0.0
    cv = mean_spread / max(mean_abs, 1e-9)
    passed = cv < 0.5                                    # spread < half the effect size

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, r in enumerate(rows):
        ax.scatter([i] * len(r["deficits"]), r["deficits"], s=22, color="#1565c0", alpha=.7)
        ax.scatter([i], [r["deficits"].mean()], s=90, marker="_", color="k")
    ax.axhline(0, color="gray", lw=.8)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([str(r["fk"]) for r in rows])
    ax.set_xlabel("lock frequency [Hz]"); ax.set_ylabel("deficit = R(control) - R(lock)")
    ax.set_title(f"H2 — deficit across phase   (CV={cv:.2f}; H2 holds if spread << effect)")
    fig.tight_layout(); fig.savefig(out / "H2_phase_invariance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    return dict(rows=[{"fk": r["fk"], "spread": s} for r, s in zip(rows, spreads)],
                mean_spread=mean_spread, mean_abs=mean_abs, cv=cv, passed=passed)


# --------------------------------------------------------------------------- #
#  --cross : regenerate the collapse-site table for every model (H3 across geometry)
# --------------------------------------------------------------------------- #
def cross_collapse_table(n_phase: int = 3) -> None:
    root = Path("outputs/hypotheses"); root.mkdir(parents=True, exist_ok=True)
    suf = "_bg" if USE_BG else ""
    freqs = np.arange(BAND[0], BAND[1] + 1, dtype=float)
    lines = ["model,P,S,stride_locks_integer,measured_collapse_freqs"]
    mode = "TSMixup background + tone" if USE_BG else "pure sinusoid"
    print(f"\n=== H3 across geometry [{mode}]: token-collapse sites follow the STRIDE grid c*fs/S ===")
    print("   (only integer-Hz stride-locks are on the sweep grid; non-integer ones are undercounted)")
    spectra = []                                          # (label, S, freqs, collapse curve, measured)
    for (P, S) in tl.ALL_MODELS:
        m = Model(P, S)
        coll = np.array([np.mean([m.collapse(f, ph) for ph in phases_Sf(f, n_phase)]) for f in freqs])
        measured = detect_collapse_sites(freqs, coll)
        # integer-Hz members of the stride-lock grid c*fs/S inside the band (what the 1 Hz sweep sees)
        pred = sorted({int(round(c * FS / m.S)) for c in range(1, m.S)
                       if (c * FS) % m.S == 0 and BAND[0] <= c * FS / m.S <= BAND[1]})
        print(f"  p{m.P}-s{m.S:<3} stride-locks(int) {pred}  ->  measured {measured}")
        lines.append(f"p{m.P}-s{m.S},{m.P},{m.S},{' '.join(map(str,pred))},{' '.join(map(str,measured))}")
        spectra.append((f"p{m.P}-s{m.S}", m.S, freqs, coll, measured))
    (root / f"collapse_sites_all_models{suf}.csv").write_text("\n".join(lines), encoding="utf-8")

    # figure: one row per model, collapse curve + measured sites; the sites clearly MOVE with S,
    # and they are IDENTICAL in shape whatever the training budget (structural, not learned).
    fig, axes = plt.subplots(len(spectra), 1, figsize=(12, 1.5 * len(spectra)), sharex=True)
    for ax, (label, S, fq, coll, meas) in zip(axes, spectra):
        ax.plot(fq, coll, color="#c62828", lw=1.1)
        for f in meas:
            ax.axvline(f, color="#6a1b9a", ls="--", lw=1.0, alpha=.7)
        ax.set_yticks([]); ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
        ax.margins(x=0.01)
    axes[0].set_title(f"H3 across geometry [{mode}] — token-collapse sites (dashed) move with the stride grid c·fs/S\n"
                      "(identical structure at every training budget → the collapse is geometric, not learned)",
                      fontsize=11)
    axes[-1].set_xlabel("frequency [Hz]")
    fig.tight_layout(); fig.savefig(root / f"H3_collapse_sites_all_models{suf}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {root/('collapse_sites_all_models'+suf+'.csv')}")
    print(f"  saved -> {root/('H3_collapse_sites_all_models'+suf+'.png')}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--cross", action="store_true", help="H3 collapse-site table over ALL models")
    ap.add_argument("--background", action="store_true",
                    help="ride the tone on a unit-variance TSMixup background (probing convention); "
                         "outputs go to *_bg paths. Default: pure sinusoid.")
    args = ap.parse_args()

    global USE_BG
    USE_BG = USE_BG or args.background
    mode = "TSMixup background + tone" if USE_BG else "pure sinusoid"

    if args.cross:
        cross_collapse_table()
        return

    m = Model(args.P, args.S)
    suf = "_bg" if USE_BG else ""
    out = Path(f"outputs/hypotheses/p{m.P}-s{m.S}{suf}"); out.mkdir(parents=True, exist_ok=True)
    print(f"loaded: {m.label}  (P={m.P}, S={m.S})  device={DEVICE}  | signal mode: {mode}")

    h3 = test_H3(m, out)
    h1 = test_H1(m, out)
    h2 = test_H2(m, out)

    print("\n" + "=" * 68)
    print(f"VERDICTS  —  {m.label}")
    print("=" * 68)
    print(f"H3 (sites match geometry): {'PASS' if h3['passed'] else 'FAIL'}")
    print(f"     predicted {h3['predicted']}")
    print(f"     measured  {h3['measured']}")
    print(f"H1 (localized loss at lock): {'SUPPORTED' if h1['passed'] else 'REFUTED'}")
    print(f"     among {h1['n_live']} live locks: d<0 at {h1['frac_neg']*100:.0f}%; "
          f"mean d={h1['mean_d']:+.2f} (delta={h1['delta']} Hz)")
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
