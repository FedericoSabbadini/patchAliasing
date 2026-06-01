"""PV signal + ontology adapter — semantic layer for the analysis notebooks.

Aligned with the OWL2 DL ontology (ontology.ttl / ontology_pv.json).
Provides: data loading, resampling, concept→cpp risk mapping,
regime-window detection, spectral helpers, and OWL-grounded state
classification using the inferred_classes restrictions.
"""
import json
import numpy as np
import pandas as pd


# ── Data I/O ──────────────────────────────────────────────────────

def load_power(csv_path, col="Power_W"):
    df = pd.read_csv(csv_path, usecols=["Date", col], parse_dates=["Date"])
    return df["Date"].values, df[col].to_numpy(float)


def resample(power, dt_min):
    """Block-mean downsample to dt_min minutes per sample (dt_min=1 → unchanged)."""
    if dt_min <= 1:
        return power
    n = (len(power) // dt_min) * dt_min
    return power[:n].reshape(-1, dt_min).mean(axis=1)


# ── Ontology loading ─────────────────────────────────────────────

def load_ontology(path):
    with open(path) as f:
        return json.load(f)


# ── Concept → cpp → aliasing risk ────────────────────────────────

def concept_table(ontology, P, dt_min=1):
    """Map every periodic concept → cpp → structural-aliasing risk.

    Uses ontology["concepts"] (list of dicts with name, kind,
    period_minutes, meaning, and optionally owl_class).
    """
    nyq = P / 2
    rows = []
    for c in ontology["concepts"]:
        per = c["period_minutes"]
        owl = c.get("owl_class", "")
        if per is None:
            cpp, risk = None, "non-periodic (state/event)"
        else:
            cpp = P * dt_min / per
            k = round(cpp)
            if   cpp > nyq:                       risk = "above token-Nyquist (aliased)"
            elif cpp < 0.5:                       risk = "near-DC (safe)"
            elif abs(cpp - k) < 0.1 and k >= 1:  risk = f"BLIND SPOT (cpp~{k})"
            else:                                 risk = "observable"
        rows.append({
            "concept":        c["name"],
            "kind":           c["kind"],
            "owl_class":      owl,
            "period_min":     per,
            "cpp":            round(cpp, 4) if cpp is not None else None,
            "period_samples": round(per / dt_min, 1) if per else None,
            "risk":           risk,
            "meaning":        c["meaning"],
        })
    return pd.DataFrame(rows)


# ── OWL inferred-class state classifier ──────────────────────────

def classify_state(ontology, G_h=None, PV_power=None):
    """Classify the current PV operating point into the OWL inferred class(es).

    Evaluates the owl_restrictions from ontology["inferred_classes"] against
    the provided sensor values.  Returns a list of matching class names.
    Only checks classes whose parent is PhotovoltaicArray.
    """
    matched = []
    for ic in ontology.get("inferred_classes", []):
        if ic["parent"] != "PhotovoltaicArray":
            continue
        restr = ic["owl_restrictions"]
        ok = True
        for prop, bounds in restr.items():
            val = {"G_h": G_h, "PV_power": PV_power}.get(prop)
            if val is None:
                ok = False; break
            if "maxInclusive" in bounds and val > bounds["maxInclusive"]:
                ok = False; break
            if "minExclusive" in bounds and val <= bounds["minExclusive"]:
                ok = False; break
            if "minInclusive" in bounds and val < bounds["minInclusive"]:
                ok = False; break
            if "maxExclusive" in bounds and val >= bounds["maxExclusive"]:
                ok = False; break
        if ok:
            matched.append(ic["name"])
    return matched


def inferred_class_table(ontology):
    """Return a DataFrame summarising the OWL inferred classes and their restrictions."""
    rows = []
    for ic in ontology.get("inferred_classes", []):
        rows.append({
            "class":      ic["name"],
            "parent":     ic["parent"],
            "state_type": ic["state_type"],
            "restrictions": json.dumps(ic["owl_restrictions"]),
            "comment":    ic.get("comment", ""),
        })
    return pd.DataFrame(rows)


# ── Regime-window detection ──────────────────────────────────────

def find_regime_windows(power, T=512, hi_period=16, stride=None):
    """Find representative windows for night_zero, clear_day, cloudy_day."""
    stride = stride or T // 2
    def hi_energy(seg):
        s = seg - seg.mean()
        if np.allclose(s, 0):
            return 0.0
        mag = np.abs(np.fft.rfft(s)) ** 2
        f = np.fft.rfftfreq(len(s))
        return mag[f > 1.0 / hi_period].sum() / (mag.sum() + 1e-9)
    starts = list(range(0, len(power) - T, stride))
    means = np.array([power[s:s+T].mean() for s in starts])
    night = next((s for s in starts
                  if power[s:s+T].std() < 1e-6 and power[s:s+T].mean() < 1e-6), None)
    thr = np.quantile(means, 0.90)
    day = [s for s, m in zip(starts, means) if m >= thr]
    clear  = min(day, key=lambda s: hi_energy(power[s:s+T])) if day else None
    cloudy = max(day, key=lambda s: hi_energy(power[s:s+T])) if day else None
    return {"night_zero": night, "clear_day": clear, "cloudy_day": cloudy}


# ── Spectral helpers ─────────────────────────────────────────────

def dominant_cpp(x, P):
    """cpp of the highest-energy non-DC frequency component."""
    x = np.asarray(x, float) - np.mean(x)
    mag = np.abs(np.fft.rfft(x))
    if len(mag) < 2 or np.allclose(mag, 0):
        return 0.0
    return float(np.fft.rfftfreq(len(x))[np.argmax(mag[1:]) + 1] * P)


def blind_band_fraction(x, P, tol=0.15):
    """Fraction of spectral energy within tol of an integer cpp in [1, P/2]."""
    x = np.asarray(x, float) - np.mean(x)
    mag = np.abs(np.fft.rfft(x)) ** 2
    cpp = np.fft.rfftfreq(len(x)) * P
    near = np.zeros(len(cpp), bool)
    for k in range(1, P // 2 + 1):
        near |= np.abs(cpp - k) <= tol
    return float(mag[near].sum() / (mag[cpp > 0].sum() + 1e-12))
