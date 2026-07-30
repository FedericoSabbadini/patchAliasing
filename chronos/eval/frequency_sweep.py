"""
frequency_sweep.py — systematic patch-aliasing evaluation for the Chronos-Bolt P/S sweep.

WHAT THIS DOES THAT THE NOTEBOOK DOES NOT
-----------------------------------------
The notebook (chronos_bolt_comparison.ipynb) probes ONE hand-picked frequency at a
time, plots it, and eyeballs a single MSE. That cannot demonstrate patch aliasing,
which is a *frequency-response* phenomenon. This script instead:

  1. Sweeps a whole grid of frequencies automatically in one run.
  2. Averages each measurement over several signal phases (a single phase can be
     lucky/unlucky), reporting mean +/- std.
  3. Measures AMPLITUDE RECOVERY and PHASE ERROR by least-squares-fitting a sinusoid
     at the known frequency to each model's median forecast — robust for short,
     non-integer-period horizons where a raw FFT is too coarse.
  4. The official amazon/chronos-bolt-tiny (P=16, S=16, 200k steps, full corpus) serves
     as the p16-s16 data point — the strongest possible S=16 anchor for the stride axis.
  5. Overlays the THEORETICAL aliasing nulls at f = k * fs / P (integer cpp) on the
     curves, so the plot either confirms patch aliasing or rules it out.
  6. Saves a tidy metrics.csv (one row per model x frequency) plus the figures.

Reuses the project's canonical tone + cpp helpers (chronos/data/synthetic/tones.py) so the
probe signal is generated with exactly the convention the synthetic generators inject.

THE ALIASING HYPOTHESIS
-----------------------
The input patch embedding is a linear map over P consecutive samples. When an integer
number of signal periods fits exactly inside one patch window, i.e.
    f = k * fs / P     <=>     cpp = k     (k = 1, 2, ...),
the patch integrates a whole period and the projection loses the oscillation -> the
model should fail to recover amplitude at those frequencies. Increasing patch overlap
(smaller stride S) raises the effective patch-sampling rate fs/S and should push the
first null higher / fill it in. This script is built to test exactly that.

Just run it:  python chronos/eval/frequency_sweep.py
Edit the CONFIG block to change the grid, context length, models, etc.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import torch
from chronos import ChronosBoltPipeline

# reuse the project's canonical tone + cpp helpers (single source of truth with the generators)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "synthetic"))
from tones import make_tone, cpp as cpp_of, cpp_to_freq

# ============================================================================ #
#  CONFIG — the only things to edit                                            #
# ============================================================================ #
REPO = "federicosabbadini/chronos-bolt-patch-sweep"   # where the retrained variants live
OFFICIAL_MODEL = "amazon/chronos-bolt-tiny"           # the official model IS p16-s16 (200k, full data)
OFFICIAL_LABEL = "p16-s16 (official)"                 # treated as the P=16 S=16 data point, not a separate ceiling
MODEL_NAMES = [                                        # the P/S variants (local-vs-local comparison)
    "p16-s12-seed42", "p16-s8-seed42",                 # S=4 dropped: its stride-lock (128 Hz) is
    "p8-s8-seed42", "p24-s24-seed42",                  # in the collapse band -> no informative test
]

FS = 512                       # sampling frequency [Hz] (matches the notebook probe)
AMPLITUDE = 5.0                # sinusoid amplitude (normalized away internally; used only for GT sanity)
FREQS = np.arange(2, 129, 2)   # frequency grid [Hz]; fine enough to resolve nulls at 32/64 Hz
PHASES = np.linspace(0, 2 * np.pi, 6, endpoint=False)  # average over these phases

CONTEXT_LENGTH = 512           # samples fed as history. NOTE: models were trained at 2048;
                               # 512 matches the notebook. Try 2048 to rule out the padding effect.
PREDICTION_LENGTH = 64         # forecast horizon [samples] (<= trained 64)
QUANTILE = 4                   # index into [0.1..0.9]; 4 = 0.5 median (the point forecast)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BATCH = 256                # chunk size for batched inference (avoid OOM)

# Load weights from the LOCAL training output when available (instant, no HF download);
# fall back to the HF repo only for variants missing locally.
LOCAL_MODELS = Path(__file__).resolve().parent.parent / "outputs" / "models"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def load_variant(name: str):
    """Load a P/S variant from local disk if its final weights exist, else from HF."""
    local = LOCAL_MODELS / name
    if (local / "model.safetensors").exists():
        print(f"  (local) {local}")
        return ChronosBoltPipeline.from_pretrained(str(local), device_map=DEVICE, torch_dtype=torch.float32)
    print(f"  (HF) {REPO}/{name}")
    return ChronosBoltPipeline.from_pretrained(REPO, subfolder=name, device_map=DEVICE, torch_dtype=torch.float32)


# ============================================================================ #
#  Helpers                                                                      #
# ============================================================================ #
def parse_ps(name: str) -> tuple[int, int]:
    """'p16-s8-seed42' -> (16, 8). Official model -> (16, 16) (its stock geometry)."""
    m = re.search(r"p(\d+)-s(\d+)", name)
    return (int(m.group(1)), int(m.group(2))) if m else (16, 16)


def make_windows(freq: float):
    """Build (contexts, gts, t_future) for every phase at one frequency.

    Reuses the project's canonical tone builder (tones.make_tone) so the probe signal is
    generated with exactly the same convention the synthetic generators inject. The first
    CONTEXT_LENGTH samples go to the model, the rest are the ground-truth continuation.
    Returns contexts [n_phase, CONTEXT_LENGTH], gts [n_phase, PREDICTION_LENGTH],
    and the absolute time axis of the horizon (shared across phases).
    """
    n = CONTEXT_LENGTH + PREDICTION_LENGTH
    contexts, gts = [], []
    for ph in PHASES:
        y = make_tone(freq, FS, n, amplitude=AMPLITUDE, phase=ph)  # canonical tone (tones.py)
        contexts.append(y[:CONTEXT_LENGTH])
        gts.append(y[CONTEXT_LENGTH:])
    t_future = (np.arange(n) / FS)[CONTEXT_LENGTH:]
    return np.asarray(contexts, np.float32), np.asarray(gts, np.float32), t_future


def fit_sinusoid(y: np.ndarray, t: np.ndarray, freq: float) -> tuple[float, float]:
    """Least-squares fit y ~ a*cos(2pi f t) + b*sin(2pi f t) + c.

    Returns (amplitude, phase) with amplitude = hypot(a, b) and phase = atan2(b, a),
    i.e. y ~ amp * cos(2pi f t - phase) + c. Robust for short / non-integer-period
    windows where an FFT bin would be too coarse.
    """
    X = np.stack([np.cos(2 * np.pi * freq * t), np.sin(2 * np.pi * freq * t), np.ones_like(t)], axis=1)
    (a, b, _c), *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(np.hypot(a, b)), float(np.arctan2(b, a))


def wrap(angle: float) -> float:
    """Wrap a phase error to [-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


@torch.no_grad()
def predict_median(pipeline, contexts: np.ndarray) -> np.ndarray:
    """Batched median forecast for a stack of contexts -> [n, PREDICTION_LENGTH]."""
    out = []
    for i in range(0, len(contexts), MAX_BATCH):
        chunk = torch.tensor(contexts[i : i + MAX_BATCH], dtype=torch.float32, device=DEVICE)
        pred = pipeline.predict(chunk, prediction_length=PREDICTION_LENGTH)  # [b, Q, H]
        out.append(pred[:, QUANTILE, :].float().cpu().numpy())
    return np.concatenate(out, axis=0)


# ============================================================================ #
#  Sweep                                                                        #
# ============================================================================ #
def evaluate_model(pipeline, label: str) -> list[dict]:
    """Run the full frequency sweep for one model; one aggregated row per frequency."""
    P = parse_ps(label)[0]                        # this model's patch size -> cpp coordinate
    rows = []
    for freq in FREQS:
        contexts, gts, t_future = make_windows(float(freq))
        preds = predict_median(pipeline, contexts)  # [n_phase, H]

        recoveries, phase_errs, mses = [], [], []
        for ph_i in range(len(PHASES)):
            amp_gt, phase_gt = fit_sinusoid(gts[ph_i], t_future, float(freq))
            amp_pr, phase_pr = fit_sinusoid(preds[ph_i], t_future, float(freq))
            if amp_gt > 1e-8:
                recoveries.append(amp_pr / amp_gt)      # 1.0 = perfect amplitude, 0 = flat
                phase_errs.append(abs(wrap(phase_pr - phase_gt)))
            mses.append(float(np.mean((preds[ph_i] - gts[ph_i]) ** 2)))

        rows.append({
            "model": label, "freq": float(freq),
            "cpp": cpp_of(float(freq), P, FS),        # cycles-per-patch: aliasing-native coordinate (nulls at integers)
            "recovery_mean": float(np.mean(recoveries)), "recovery_std": float(np.std(recoveries)),
            "phase_err_mean": float(np.mean(phase_errs)), "phase_err_std": float(np.std(phase_errs)),
            "mse_mean": float(np.mean(mses)),
        })
        print(f"  {label} @ {freq:6.1f} Hz (cpp={rows[-1]['cpp']:.2f})  recovery={rows[-1]['recovery_mean']:.3f}  "
              f"phase_err={np.degrees(rows[-1]['phase_err_mean']):5.1f} deg")
    return rows


def null_freqs(P: int, fmax: float) -> list[float]:
    """Theoretical patch-aliasing nulls: integer cpp -> f = k*fs/P below fmax (via tones.cpp_to_freq)."""
    return [cpp_to_freq(k, P, FS) for k in range(1, int(fmax * P / FS) + 1)]


# ============================================================================ #
#  Plots                                                                        #
# ============================================================================ #
def plot_recovery(all_rows: dict[str, list[dict]]):
    """Primary figure: amplitude recovery vs frequency — all models as equal participants."""
    fig, ax = plt.subplots(figsize=(13, 6))
    fmax = float(FREQS.max())

    # theoretical nulls for each patch size present (grey verticals, one style per P)
    all_labels = list(all_rows.keys())
    for P, ls in zip(sorted({parse_ps(n)[0] for n in all_labels}), [":", "--", "-."]):
        for j, f0 in enumerate(null_freqs(P, fmax)):
            ax.axvline(f0, color="grey", ls=ls, alpha=0.5, lw=1,
                       label=f"P={P} nulls (k*fs/P)" if j == 0 else None)

    for label, rows in all_rows.items():
        f = [r["freq"] for r in rows]
        rec = np.array([r["recovery_mean"] for r in rows])
        std = np.array([r["recovery_std"] for r in rows])
        line, = ax.plot(f, rec, lw=2, marker="o", ms=3, label=label)
        ax.fill_between(f, rec - std, rec + std, color=line.get_color(), alpha=0.15)

    ax.axhline(1.0, color="green", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("signal frequency [Hz]"); ax.set_ylabel("amplitude recovery (pred / ground truth)")
    ax.set_title("Patch-aliasing frequency response\n(recovery dips at f = k*fs/P confirm patch aliasing)")
    ax.set_ylim(-0.05, 1.35); ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "recovery_vs_frequency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_recovery_cpp(all_rows: dict[str, list[dict]]):
    """Aliasing-native figure: recovery vs cycles-per-patch (cpp = freq*P/fs).

    Because each model's x-axis is its own cpp, the theoretical nulls of EVERY model —
    whatever its P — collapse onto the same integer ticks (cpp = 1, 2, 3, ...). A shared
    dip at the integers is the cleanest single confirmation of patch aliasing.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    cpp_max = max(r["cpp"] for rows in all_rows.values() for r in rows)
    for k in range(1, int(cpp_max) + 1):
        ax.axvline(k, color="grey", ls="--", alpha=0.5, lw=1,
                   label="integer cpp (predicted nulls)" if k == 1 else None)
    for label, rows in all_rows.items():
        c = [r["cpp"] for r in rows]
        rec = [r["recovery_mean"] for r in rows]
        ax.plot(c, rec, lw=2, marker="o", ms=3, label=f"{label} (P={parse_ps(label)[0]})")
    ax.axhline(1.0, color="green", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("cycles per patch  cpp = freq * P / fs"); ax.set_ylabel("amplitude recovery")
    ax.set_title("Patch aliasing in the native coordinate\n(every P's nulls collapse onto integer cpp)")
    ax.set_ylim(-0.05, 1.35); ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "recovery_vs_cpp.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_stride_group(all_rows: dict[str, list[dict]]):
    """Focus figure: the P=16 stride axis (S=16->4) — does overlap fill the null?
    The official model serves as the S=16 anchor (best-trained P=16 S=16 available)."""
    p16 = [n for n in all_rows if parse_ps(n)[0] == 16]
    if not p16:
        return
    p16.sort(key=lambda n: parse_ps(n)[1], reverse=True)  # S=16 first
    fig, ax = plt.subplots(figsize=(12, 6))
    for f0 in null_freqs(16, float(FREQS.max())):
        ax.axvline(f0, color="grey", ls="--", alpha=0.5, lw=1)
    for label in p16:
        rows = all_rows[label]
        ax.plot([r["freq"] for r in rows], [r["recovery_mean"] for r in rows],
                lw=2, marker="o", ms=3, label=f"{label} (S={parse_ps(label)[1]})")
    ax.axhline(1.0, color="green", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("signal frequency [Hz]"); ax.set_ylabel("amplitude recovery")
    ax.set_title("Stride / overlap axis (P=16): does reducing stride S fill the aliasing null?")
    ax.set_ylim(-0.05, 1.35); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "recovery_stride_axis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_phase(all_rows: dict[str, list[dict]]):
    """Phase error vs frequency."""
    fig, ax = plt.subplots(figsize=(13, 6))
    for label, rows in all_rows.items():
        f = [r["freq"] for r in rows]
        pe = np.degrees([r["phase_err_mean"] for r in rows])
        ax.plot(f, pe, lw=2, marker="o", ms=3, label=label)
    ax.set_xlabel("signal frequency [Hz]"); ax.set_ylabel("|phase error| [degrees]")
    ax.set_title("Phase error vs frequency"); ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "phase_error_vs_frequency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================================ #
#  Main                                                                         #
# ============================================================================ #
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Frequency sweep on {DEVICE} | {len(FREQS)} freqs x {len(PHASES)} phases "
          f"| context={CONTEXT_LENGTH} horizon={PREDICTION_LENGTH}")

    all_rows: dict[str, list[dict]] = {}

    # Official amazon/chronos-bolt-tiny IS the P=16 S=16 data point (200k, full corpus)
    print(f"\nLoading {OFFICIAL_LABEL}: {OFFICIAL_MODEL}")
    official = ChronosBoltPipeline.from_pretrained(OFFICIAL_MODEL, device_map=DEVICE)
    all_rows[OFFICIAL_LABEL] = evaluate_model(official, OFFICIAL_LABEL)

    for name in MODEL_NAMES:
        print(f"\nLoading {name}")
        pipe = load_variant(name)
        all_rows[name] = evaluate_model(pipe, name)

    # tidy CSV: one row per model x frequency
    csv_path = OUTPUT_DIR / "metrics.csv"
    fields = ["model", "freq", "cpp", "recovery_mean", "recovery_std",
              "phase_err_mean", "phase_err_std", "mse_mean"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for rows in all_rows.values():
            w.writerows(rows)
    print(f"\nSaved {csv_path}")

    plot_recovery(all_rows)
    plot_recovery_cpp(all_rows)
    plot_stride_group(all_rows)
    plot_phase(all_rows)
    print(f"Saved figures to {OUTPUT_DIR}")

    # console summary: mean recovery + measured null depth per model
    print("\n=== Summary (mean amplitude recovery across the grid) ===")
    for label, rows in all_rows.items():
        rec = np.array([r["recovery_mean"] for r in rows])
        P = parse_ps(label)[0]
        nulls = null_freqs(P, float(FREQS.max()))
        note = ""
        if nulls:
            i = int(np.argmin(np.abs(FREQS - nulls[0])))
            note = f" | recovery@first-null({nulls[0]:.0f}Hz)={rec[i]:.3f}"
        print(f"  {label:24s} mean_recovery={rec.mean():.3f}{note}")


if __name__ == "__main__":
    main()
