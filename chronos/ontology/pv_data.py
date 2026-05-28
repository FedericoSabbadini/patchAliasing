"""PV signal + ontology adapter — semantic layer for the analysis notebooks."""
import json
import numpy as np
import pandas as pd


def load_power(csv_path, col="Power_W"):
    df = pd.read_csv(csv_path, usecols=["Date", col], parse_dates=["Date"])
    return df["Date"].values, df[col].to_numpy(float)


def resample(power, dt_min):
    """Block-mean downsample to dt_min minutes per sample (dt_min=1 → unchanged)."""
    if dt_min <= 1:
        return power
    n = (len(power) // dt_min) * dt_min
    return power[:n].reshape(-1, dt_min).mean(axis=1)


def load_ontology(path):
    with open(path) as f:
        return json.load(f)


def concept_table(ontology, P, dt_min=1):
    """Map every periodic concept → cpp → structural-aliasing risk."""
    nyq = P / 2
    rows = []
    for c in ontology["concepts"]:
        per = c["period_minutes"]
        if per is None:
            cpp, risk = None, "non-periodic (state/event)"
        else:
            cpp = P * dt_min / per
            k   = round(cpp)
            if   cpp > nyq:                         risk = "above token-Nyquist (aliased)"
            elif cpp < 0.5:                         risk = "near-DC (safe)"
            elif abs(cpp - k) < 0.1 and k >= 1:    risk = f"BLIND SPOT (cpp~{k})"
            else:                                   risk = "observable"
        rows.append({"concept": c["name"], "kind": c["kind"], "period_min": per,
                     "cpp": round(cpp, 4) if cpp is not None else None,
                     "period_samples": round(per / dt_min, 1) if per else None,
                     "risk": risk, "meaning": c["meaning"]})
    return pd.DataFrame(rows)


def find_regime_windows(power, T=512, hi_period=16, stride=None):
    """Find representative windows for night_zero, clear_day, cloudy_day."""
    stride = stride or T // 2
    def hi_energy(seg):
        s = seg - seg.mean()
        if np.allclose(s, 0): return 0.0
        mag = np.abs(np.fft.rfft(s)) ** 2
        f   = np.fft.rfftfreq(len(s))
        return mag[f > 1.0 / hi_period].sum() / (mag.sum() + 1e-9)
    starts = list(range(0, len(power) - T, stride))
    means  = np.array([power[s:s+T].mean() for s in starts])
    night  = next((s for s in starts
                   if power[s:s+T].std() < 1e-6 and power[s:s+T].mean() < 1e-6), None)
    thr    = np.quantile(means, 0.90)
    day    = [s for s, m in zip(starts, means) if m >= thr]
    clear  = min(day, key=lambda s: hi_energy(power[s:s+T])) if day else None
    cloudy = max(day, key=lambda s: hi_energy(power[s:s+T])) if day else None
    return {"night_zero": night, "clear_day": clear, "cloudy_day": cloudy}


def dominant_cpp(x, P):
    """cpp of the highest-energy non-DC frequency component."""
    x   = np.asarray(x, float) - np.mean(x)
    mag = np.abs(np.fft.rfft(x))
    if len(mag) < 2 or np.allclose(mag, 0): return 0.0
    return float(np.fft.rfftfreq(len(x))[np.argmax(mag[1:]) + 1] * P)


def blind_band_fraction(x, P, tol=0.15):
    """Fraction of spectral energy within tol of an integer cpp in [1, P/2]."""
    x   = np.asarray(x, float) - np.mean(x)
    mag = np.abs(np.fft.rfft(x)) ** 2
    cpp = np.fft.rfftfreq(len(x)) * P
    near = np.zeros(len(cpp), bool)
    for k in range(1, P // 2 + 1):
        near |= np.abs(cpp - k) <= tol
    return float(mag[near].sum() / (mag[cpp > 0].sum() + 1e-12))
