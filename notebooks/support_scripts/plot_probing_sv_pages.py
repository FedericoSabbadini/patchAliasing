"""Appendix pages of the stage-wise space saving, every checkpoint, pure tones.

Reads the pure-tone export of probing_pure_tones.ipynb (results/probing/pure_sv.csv)
and draws three pages grouped as the frequency sweep of Appendix F: one panel per
checkpoint, ten stages by seven band tasks, with the two controls drawn as ticks on
each bar instead of the printed values of the single-geometry figure.

    python plot_probing_sv_pages.py <pure_sv.csv> <output directory>
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Same order, labels, colours and hatches as probing_exports.py.
STAGES = ("enc_0", "enc_1", "enc_2", "enc_3",
          "dec_0", "dec_1", "dec_2", "dec_3", "output_reg", "output_head")
STAGE_LABELS = {
    "enc_0": "Encoder 1", "enc_1": "Encoder 2", "enc_2": "Encoder 3", "enc_3": "Encoder 4",
    "dec_0": "Decoder 1", "dec_1": "Decoder 2", "dec_2": "Decoder 3", "dec_3": "Decoder 4",
    "output_reg": "Projection input", "output_head": "Output head",
}
TASKS = ("LL", "L", "LH", "Mid", "HL", "H", "HH")
COLOURS = ("#22577a", "#3c8d99", "#73b5a1", "#d4a323", "#cd8032", "#b45878", "#765785")
HATCHES = ("", "//", "..", "xx", "\\\\", "++", "oo")

# Grouped as Figures sweepA-C; the published checkpoint opens the first page.
PAGES = (
    ("official-p16-s16", "retrained-p8-s8", "retrained-p16-s8",
     "retrained-p16-s12", "retrained-p16-s16", "retrained-p24-s8"),
    ("retrained-p24-s12", "retrained-p24-s16", "retrained-p24-s20",
     "retrained-p24-s24", "retrained-p32-s8"),
    ("retrained-p32-s12", "retrained-p32-s16", "retrained-p32-s20",
     "retrained-p32-s24", "retrained-p32-s32"),
)


def _title(model: str) -> str:
    kind, geometry = model.split("-", 1)
    return f"{geometry}, published" if kind == "official" else geometry


def _panel(axis, frame: pd.DataFrame, xlim: tuple[float, float], labels: bool) -> None:
    height = 0.115
    for stage_index, stage in enumerate(STAGES):
        for task_index, task in enumerate(TASKS):
            row = frame.loc[(frame.stage == stage) & (frame.task == task)].iloc[0]
            y = stage_index + (task_index - 3) * height
            axis.barh(y, row.SV, height=height * .9, color=COLOURS[task_index],
                      edgecolor="#333333", linewidth=.25, hatch=HATCHES[task_index])
            for value, colour in ((row.SV_random_init, "#111111"), (row.SV_shuffled, "#8a8a8a")):
                axis.plot([value, value], [y - height * .5, y + height * .5],
                          color=colour, linewidth=.9, solid_capstyle="butt")
    for divider in (3.5, 7.5):
        axis.axhline(divider, color="#999999", linewidth=.5, linestyle=":")
    axis.axvline(0, color="#333333", linewidth=.6)
    axis.set_xlim(*xlim)
    axis.set_ylim(len(STAGES) - .5, -.5)
    axis.set_yticks(np.arange(len(STAGES)),
                    [STAGE_LABELS[s] for s in STAGES] if labels else [""] * len(STAGES))
    axis.tick_params(axis="y", length=0)
    axis.set_xticks([0, .2, .4, .6, .8])
    axis.grid(axis="x", color="#dddddd", linewidth=.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def draw(frame: pd.DataFrame, out: Path) -> list[Path]:
    plt.rcParams.update({"font.size": 7, "hatch.linewidth": .35})
    values = frame[["SV", "SV_random_init", "SV_shuffled"]].to_numpy(dtype=float)
    xlim = (min(-.1, float(np.nanmin(values)) - .03), .85)
    written = []
    for number, models in enumerate(PAGES, start=1):
        figure, axes = plt.subplots(2, 3, figsize=(6.5, 7.9), sharex=True)
        for index, axis in enumerate(axes.flat):
            if index >= len(models):
                axis.set_visible(False)
                continue
            _panel(axis, frame[frame.model == models[index]], xlim, labels=index % 3 == 0)
            axis.set_title(_title(models[index]), fontsize=8, fontweight="bold", pad=3)
            if index // 3 == 1 or index + 3 >= len(models):
                axis.set_xlabel("Space saving, SV")
                axis.xaxis.set_tick_params(labelbottom=True)
        handles = [Patch(facecolor=c, edgecolor="#333333", hatch=h, label=t)
                   for t, c, h in zip(TASKS, COLOURS, HATCHES)]
        handles += [Line2D([], [], color="#111111", linewidth=1.2, label="untrained"),
                    Line2D([], [], color="#8a8a8a", linewidth=1.2, label="shuffled")]
        figure.legend(handles=handles, loc="lower center", ncol=9, frameon=False,
                      fontsize=7, handlelength=1.6, columnspacing=1.0)
        figure.subplots_adjust(left=.13, right=.99, top=.965, bottom=.085,
                               wspace=.08, hspace=.16)
        for suffix in ("pdf", "png"):
            path = out / f"fig_probing_sv_page{number}.{suffix}"
            figure.savefig(path, dpi=200, facecolor="white")
            written.append(path)
        plt.close(figure)
    return written


if __name__ == "__main__":
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    target.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(source)
    data = data[data.family == "pure"]
    for path in draw(data, target):
        print(path)
