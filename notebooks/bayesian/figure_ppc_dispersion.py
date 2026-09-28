"""Figure 4 of Deliverable 3 (Appendix E): posterior predictive check of every fit that decides a claim.

Reads the per-level check table the Bayesian notebook writes (tables/05_ppc.csv of a run folder) and
draws, for every level of every stratum, the observed mean against its replicated interval (left)
and the replicated over the observed standard deviation (right).

usage: python figure_ppc_dispersion.py <run_folder>/tables/05_ppc.csv [output.pdf]
"""
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FITS = [("A_both", "Model A"), ("C_phase", "Model C"), ("B_adjusted", "Model B"),
        ("D1_tsmixup_both", "Model D1, Light TSMixup"), ("D1_kernelsynth_both", "Model D1, KernelSynth"),
        ("D2_stride_f1", "Model D2, stride branch")]
CLIP = 20.0  # levels with an observed standard deviation of zero are drawn at this ratio


def main(ppc_csv: str, out: str = "bayes_ppc_dispersion.pdf") -> None:
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8.5,
                         "axes.linewidth": 0.6})
    ppc = pd.read_csv(ppc_csv)
    ppc = ppc[ppc.stratum != "global"].copy()
    ppc["mid"] = (ppc.rep_low + ppc.rep_high) / 2
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True, gridspec_kw={"wspace": 0.08})
    fail, ok, ink = "#1f5fa8", "#9aa3ad", "#2b2b2b"
    rng = np.random.default_rng(1)
    for row, (fit, _) in enumerate(FITS):
        g = ppc[ppc.fit == fit]
        means, sds = g[g.metric == "mean"], g[g.metric == "sd"]
        ms = means.merge(sds[["stratum", "level", "observed"]], on=["stratum", "level"],
                         suffixes=("", "_sd"))
        z = ((ms.observed - ms.mid) / ms.observed_sd.where(ms.observed_sd > 1e-9)).dropna()
        passed = ms.loc[z.index, "ppc_ok"].values
        y = row + rng.uniform(-0.18, 0.18, len(z))
        axs[0].scatter(z[passed], y[passed], s=11, facecolors="none", edgecolors=ok, linewidths=0.7)
        axs[0].scatter(z[~passed], y[~passed], s=11, color=fail, linewidths=0)
        ratio = (sds.mid / sds.observed.where(sds.observed > 1e-3)).fillna(CLIP).clip(upper=CLIP)
        passed = sds.ppc_ok.values
        y = row + rng.uniform(-0.18, 0.18, len(ratio))
        axs[1].scatter(ratio[passed], y[passed], s=11, facecolors="none", edgecolors=ok, linewidths=0.7)
        axs[1].scatter(ratio[~passed], y[~passed], s=11, color=fail, linewidths=0)
    axs[0].axvline(0, color=ink, lw=0.6)
    axs[1].axvline(1, color=ink, lw=0.6)
    axs[0].set_xlim(-0.35, 0.35)
    axs[0].set_xlabel("stratum mean: observed minus replicated,\nin units of the stratum standard deviation")
    axs[1].set_xscale("log")
    axs[1].set_xlim(0.25, 25)
    axs[1].set_xticks([0.3, 1, 3, 10, CLIP])
    axs[1].set_xticklabels(["0.3", "1", "3", "10", r"$\geq$20"])
    axs[1].set_xlabel("stratum standard deviation:\nreplicated over observed")
    axs[0].set_yticks(range(len(FITS)))
    axs[0].set_yticklabels([label for _, label in FITS])
    axs[0].invert_yaxis()
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e3e3e3", lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=0.6)
    handles = [Line2D([], [], marker="o", ls="", color=fail, ms=4, label="outside the 95% replicated interval"),
               Line2D([], [], marker="o", ls="", mfc="none", mec=ok, ms=4, label="inside")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.03))
    fig.savefig(out, bbox_inches="tight")


if __name__ == "__main__":
    main(*sys.argv[1:3])
