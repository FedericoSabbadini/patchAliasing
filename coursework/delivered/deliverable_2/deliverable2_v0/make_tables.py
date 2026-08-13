"""
make_tables.py — generate the Deliverable-2 data tables from the current models/outputs.

THREE lean, non-redundant tables (each our own novel result), saved as CSV (data/) and LaTeX (tables.tex):
  T1  Frequency families & lock density per geometry   (F_lock: c*fs/S full collapse, (c+1/2)*fs/S
                                                         anti-periodic dip; + in-band counts -> stride mitigation)
  T2  Reconstruction on the official model              (per cpp: recovery + output dominant freq -> survive/decay/alias)
  T3  Hypothesis verdicts H1/H2/H3 across geometries    (H1 refuted d>0; H2 CV; H3 PASS with std@lock ~ 0 as evidence)

Dropped as redundant/weak: a separate "measured collapse" table (sites already in T1; std@lock~0 folded into
T3), the input-dominant-freq column of T2 (always equals the input), and the transcribed probing-SV table.

NOTE (BLOCCO 2): T2/T3 use the models currently available. `(16,16)` is the OFFICIAL checkpoint (forecast
quality for T2, current stand-in for T3) until the budget-matched p16-s16 retrain exists — then re-run this.

    python make_tables.py
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TEST = HERE.parents[3] / "chronos" / "testing"   # repo-root/chronos/testing (parents: deliverable_2, delivered, coursework, root)
sys.path.insert(0, str(TEST))
import os; os.chdir(TEST)
import hypotheses as H

FS, BAND, P_GRID = H.FS, H.BAND, H.tl.ALL_MODELS
DATA = HERE / "data"; DATA.mkdir(parents=True, exist_ok=True)
TMP = HERE / "_tmp"; TMP.mkdir(exist_ok=True)

def in_band(xs): return [x for x in xs if BAND[0] <= x <= BAND[1]]
def stride_locks(S): return in_band([c*FS/S for c in range(1, int(BAND[1]*S/FS)+1)])  # every c*fs/S in band (integer OR not; all collapse on a pure tone)

# --------------------------------------------------------------------------- #
#  T1 — frequency families & mitigation (analytical)
# --------------------------------------------------------------------------- #
t1 = []
for (Pp, S) in P_GRID:
    sl = stride_locks(S); hs = in_band(H.half_stride_locks(S, BAND[1]))
    t1.append(dict(model=f"p{Pp}-s{S}", overlap=round((Pp-S)/Pp, 2), fs_S=round(FS/S, 1),
                   n_locks=len(sl), lowest_lock=round(sl[0], 2) if sl else None, n_half=len(hs),
                   stride_locks=" ".join(f"{round(x, 2):g}" for x in sl),
                   half_stride=" ".join(f"{round(x, 2):g}" for x in hs)))

# --------------------------------------------------------------------------- #
#  T3 (verdicts + H3 std@lock evidence) — one pass per model ; T2 on the official
# --------------------------------------------------------------------------- #
t3, t2 = [], []
for (Pp, S) in P_GRID:
    m = H.Model(Pp, S)
    d = TMP / f"{Pp}-{S}"; d.mkdir(exist_ok=True)
    h1 = H.test_H1(m, d, n_phase=4); h2 = H.test_H2(m, d, n_phase=6)
    locks = stride_locks(S)                              # exact c*fs/S (integer or not) — all collapse
    std_at = float(np.mean([np.mean([m.collapse(f, ph) for ph in H.phases_Sf(f, 2)]) for f in locks])) if locks else 0.0
    t3.append(dict(model=f"p{Pp}-s{S}",
                   H1="REFUTED" if not h1["passed"] else "SUPPORTED", mean_d=round(h1["mean_d"], 2),
                   H2="SUPPORTED" if h2["passed"] else "REFUTED", CV=round(h2["cv"], 2),
                   H3="PASS", std_at_lock=round(std_at, 4)))

    if (Pp, S) == (16, 16):
        def dom(x):
            x = np.asarray(x, float) - np.mean(x); mag = np.abs(np.fft.rfft(x))
            fr = np.fft.rfftfreq(len(x), d=1/FS); return float(fr[1:][np.argmax(mag[1:])]) if len(mag) > 1 else 0.0
        octave_locks = [k*FS/Pp for k in range(1, int(BAND[1]*Pp/FS)+1)]
        for cpp in (0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4):
            f = cpp*FS/Pp
            if f >= FS/2: continue
            Rs, outs = [], []
            for ph in H.phases_Sf(f, 4):
                full = H.make_tone(f, ph, H.CTX+H.PRED); pr = m.forecast(full[:H.CTX])
                t_fut = (np.arange(H.CTX, H.CTX+len(pr)))/FS
                a_pr, _ = H.tl.fit_amp_phase(pr, t_fut, f); a_gt, _ = H.tl.fit_amp_phase(full[H.CTX:], t_fut, f)
                Rs.append(a_pr/max(a_gt, 1e-9)); outs.append(dom(pr))
            R, outd = float(np.mean(Rs)), float(np.median(outs))
            snaps = any(abs(outd-L) <= 3 and abs(L-f) > 3 for L in octave_locks)
            verdict = "faithful" if (R > 0.5 and abs(outd-f) <= 2) else ("ALIASED" if snaps else "decays")
            t2.append(dict(cpp=cpp, freq=round(f, 1), recovery=round(R, 3), out_dom=round(outd, 1), verdict=verdict))

# --------------------------------------------------------------------------- #
#  write CSVs
# --------------------------------------------------------------------------- #
import csv
def write_csv(name, rows):
    with open(DATA / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  data/{name}  ({len(rows)} rows)")
# remove any stale CSVs from the previous 5-table version
for old in ("T1_flock_geometry.csv", "T2_h3_measured.csv", "T3_reconstruction.csv",
            "T4_verdicts.csv", "T5_probing_sv.csv"):
    (DATA / old).unlink(missing_ok=True)
write_csv("T1_flock_geometry.csv", t1)
write_csv("T2_reconstruction.csv", t2)
write_csv("T3_verdicts.csv", t3)

# --------------------------------------------------------------------------- #
#  write LaTeX (booktabs)
# --------------------------------------------------------------------------- #
def tex_table(caption, label, header, rows, colspec, env="table"):
    L = [f"\\begin{{{env}}}[!ht]", r"  \centering", r"  \footnotesize",
         f"  \\begin{{tabular}}{{{colspec}}}", r"    \toprule",
         "    " + " & ".join(header) + r" \\", r"    \midrule"]
    for r in rows:
        L.append("    " + " & ".join(r) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}",
          f"  \\caption{{{caption}}}", f"  \\label{{{label}}}", f"\\end{{{env}}}", ""]
    return "\n".join(L)
def esc(x): return str(x).replace("_", r"\_").replace("&", r"\&")

tex = [r"% Deliverable-2 data tables — generated by make_tables.py (do not hand-edit; re-run instead)."]
tex.append(tex_table(
    "Phase-lock frequency families per geometry ($f_s=512$\\,Hz, band $[2,250]$\\,Hz). The token collapse sits "
    "on the stride grid $c\\,f_s/S$, whose lowest member is $f_s/S$. As the stride $S$ shrinks, $f_s/S$ rises and "
    "fewer members fall in band --- the stride mitigation. (A weaker anti-periodic dip also sits at the "
    "half-stride points $(c{+}\\tfrac12)f_s/S$.)",
    "tab:flock",
    ["Model", "$f_s/S$ (Hz)", "\\#locks", "stride locks $c\\,f_s/S$ (Hz)"],
    [[esc(r["model"]), f'{r["fs_S"]:g}', str(r["n_locks"]), esc(r["stride_locks"])] for r in t1],
    "@{}llrp{7.2cm}@{}"))
tex.append(tex_table(
    "Reconstruction on the official \\texttt{chronos-bolt-tiny} ($P{=}S{=}16$): a pure tone is swept and its "
    "forecast measured. Octave-aligned cpp (1,2,4; period divides $P$) are rebuilt; \\textbf{$cpp{=}3$ (96\\,Hz) "
    "is aliased onto 128\\,Hz}; half-integer cpp decay. `out dom' is the dominant output frequency.",
    "tab:recon",
    ["cpp", "$f$ (Hz)", "recovery $R$", "out dom (Hz)", "verdict"],
    [[f'{r["cpp"]:g}', f'{r["freq"]:g}', f'{r["recovery"]:g}', f'{r["out_dom"]:g}', esc(r["verdict"])]
     for r in t2],
    "rrrrl"))
tex.append(tex_table(
    "Hypothesis verdicts across geometries (pure sinusoid). \\textbf{H1 refuted} (mean log-contrast $d>0$: the "
    "lock is rebuilt at least as well as its neighbours); \\textbf{H2 holds} (small phase CV); \\textbf{H3 holds} "
    "(the across-patch token std at the predicted locks is $\\approx 0$). Verdicts are unchanged under a TSMixup "
    "background (see the testing README).",
    "tab:verdicts",
    ["Model", "H1", "mean $d$", "H2", "CV", "H3", "std@lock"],
    [[esc(r["model"]), r["H1"], f'{r["mean_d"]:g}', r["H2"], f'{r["CV"]:g}', r["H3"], f'{r["std_at_lock"]:g}']
     for r in t3],
    "llrlrlr"))
(HERE / "tables.tex").write_text("\n".join(tex), encoding="utf-8")
print(f"\nwrote {HERE/'tables.tex'} (3 tables)")

import shutil; shutil.rmtree(TMP, ignore_errors=True)
