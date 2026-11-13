"""Checked report exports for the two SV probing notebooks.

The compression count concerns labels of the injected tone, conditional on a
background.  It is not waveform compression or forecast-line recovery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__:
    from . import checkpointing as cp
    from .comparison_lib import MODELS
else:
    import checkpointing as cp
    from comparison_lib import MODELS


STAGES = ("dec_0", "dec_1", "dec_2", "dec_3", "output_head")
TASKS = ("LL", "L", "LH", "Mid", "HL", "H", "HH")
TASK_BANDS = {
    "LL": (2, 64), "L": (2, 126), "LH": (65, 126),
    "Mid": (2, 250), "HL": (127, 188), "H": (127, 250), "HH": (189, 250),
}
FAMILIES = ("tsmixup", "kernelsynth")
FAMILY_LABELS = {"tsmixup": "Light TSMixup", "kernelsynth": "KernelSynth"}
REFERENCE = "retrained-p16-s16"
IDENTIFIERS = ("model", "P", "S", "official", "checkpoint", "family", "background_id", "stage", "task")
NUMBERS = ("n", "L_bits", "L_uniform", "SV", "L_shuffled_bits", "SV_shuffled", "first_chunk_n")
PAGANI_URL = "https://arxiv.org/html/2605.06361v1"

# Ordered colours and hatch patterns retain task identity in monochrome prints.
COLOURS = ("#22577a", "#3c8d99", "#73b5a1", "#d4a323", "#cd8032", "#b45878", "#765785")
HATCHES = ("", "//", "..", "xx", "\\\\", "++", "oo")


def _csv(path: Path, frame: pd.DataFrame) -> None:
    cp.atomic_write(path, lambda target: frame.to_csv(target, index=False))


def _text(path: Path, content: str) -> None:
    cp.atomic_write(path, lambda target: target.write_text(content, encoding="utf-8"))


def _save_figure(figure: Any, stem: Path) -> dict[str, Path]:
    paths = {"png": stem.with_suffix(".png"), "pdf": stem.with_suffix(".pdf")}
    for suffix, path in paths.items():
        cp.atomic_write(path, lambda target, fmt=suffix: figure.savefig(
            target, format=fmt, dpi=180, bbox_inches="tight", facecolor="white"))
    return paths


def _validated_numbers(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(IDENTIFIERS + NUMBERS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing probing result columns: {sorted(missing)}")
    if frame.empty or frame[list(IDENTIFIERS)].isna().any().any():
        raise ValueError("Empty probing results or null identifiers")
    result = frame.copy()
    for field in NUMBERS:
        result[field] = pd.to_numeric(result[field], errors="raise")
    values = result[list(NUMBERS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite probing results cannot be exported")
    if ((result.n <= 0) | (result.first_chunk_n <= 0) | (result.first_chunk_n >= result.n)).any():
        raise ValueError("Invalid sample or first-block count")
    if not np.equal(result.n, np.floor(result.n)).all():
        raise ValueError("Sample counts must be integers")
    if (result[["L_bits", "L_shuffled_bits"]] < 0).any().any():
        raise ValueError("Codelengths must be nonnegative")
    if not np.allclose(result.L_uniform, result.n, rtol=0, atol=1e-10):
        raise ValueError("Binary uniform codelength must equal the sample count")
    for metric, length in (("SV", "L_bits"), ("SV_shuffled", "L_shuffled_bits")):
        if not np.allclose(result[metric], 1 - result[length] / result.n, rtol=0, atol=1e-10):
            raise ValueError(f"{metric} does not match its codelength")
    return result


def _require_product(frame: pd.DataFrame, columns: list[str], expected: pd.MultiIndex) -> None:
    if frame.duplicated(columns).any():
        raise ValueError(f"Duplicate probing results for {columns}")
    actual = pd.MultiIndex.from_frame(frame[columns])
    missing, unexpected = expected.difference(actual), actual.difference(expected)
    if len(missing) or len(unexpected):
        raise ValueError(
            f"Incomplete probing results: {len(missing)} missing, {len(unexpected)} unexpected cells")


def _plot_bars(frame: pd.DataFrame, title: str, note: str):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure, axis = plt.subplots(figsize=(11.2, 8.8))
    height = 0.095
    centres = np.arange(len(STAGES))
    # Only true-task values are drawn as bars; shuffled values are text labels.
    # Let the drawn quantities set the axis so a poor control cannot flatten them.
    numbers = frame["SV"].to_numpy(dtype=float)
    lower = min(-0.08, float(numbers.min()) - 0.07)
    upper = max(0.88, float(numbers.max()) + 0.05)
    label_x = upper + 0.025
    for stage_index, stage in enumerate(STAGES):
        for task_index, task in enumerate(TASKS):
            row = frame.loc[(frame.stage == stage) & (frame.task == task)].iloc[0]
            y = stage_index + (task_index - 3) * height
            axis.barh(y, row.SV, height=height * .90,
                      color=COLOURS[task_index], edgecolor="#333333", linewidth=.35,
                      hatch=HATCHES[task_index])
            axis.text(label_x, y, f"{row.SV:+.3f} ({row.SV_shuffled:+.3f})",
                      va="center", ha="left", fontsize=8.3, family="monospace")
    axis.axvline(0, color="#333333", linewidth=.8)
    axis.set_xlim(lower, upper)
    axis.set_yticks(centres, ["Decoder 0", "Decoder 1", "Decoder 2", "Decoder 3", "Final output"])
    axis.invert_yaxis()
    axis.set_xlabel("Space saving, SV = 1 − MDL codelength / uniform codelength")
    axis.set_title(title, loc="left", fontsize=13, pad=16)
    axis.text(label_x, -.58, "SV (shuffled labels)", ha="left", va="bottom", fontsize=9)
    axis.grid(axis="x", color="#dddddd", linewidth=.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    handles = [Patch(facecolor=colour, edgecolor="#333333", hatch=hatch, label=task)
               for task, colour, hatch in zip(TASKS, COLOURS, HATCHES)]
    figure.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
                  bbox_to_anchor=(.47, .062))
    figure.text(.10, .025, note, ha="left", va="bottom", fontsize=8.5)
    figure.subplots_adjust(left=.13, right=.72, bottom=.15, top=.92)
    return figure


def export_pure(frame: pd.DataFrame, out: Path, mode: str = "full") -> dict[str, Path]:
    """Export one five-stage SV figure per checkpoint, with a named 16–16 figure."""
    import matplotlib.pyplot as plt

    if mode not in {"full", "smoke"}:
        raise ValueError("Expected full or smoke mode")
    frame = _validated_numbers(frame)
    if set(frame.family) != {"pure"} or set(frame.background_id) != {-1}:
        raise ValueError("Pure probing must use family=pure and background_id=-1")
    model_tags = list(dict.fromkeys(frame.model))
    if mode == "full" and set(model_tags) != {spec.tag for spec in MODELS}:
        raise ValueError("Full pure export requires all 16 checkpoints")
    expected = pd.MultiIndex.from_product([model_tags, STAGES, TASKS], names=["model", "stage", "task"])
    _require_product(frame, ["model", "stage", "task"], expected)
    if mode == "full":
        expected_n = frame.task.map(lambda task: 10 * (TASK_BANDS[task][1] - TASK_BANDS[task][0] + 1))
        if not np.array_equal(frame.n, expected_n):
            raise ValueError("Full pure export requires all frequencies and ten phases")
    out = Path(out)
    prefix = "smoke_" if mode == "smoke" else ""
    paths: dict[str, Path] = {"results_csv": out / f"{prefix}pure_sv.csv"}
    _csv(paths["results_csv"], frame)
    for tag, subset in frame.groupby("model", sort=False):
        row = subset.iloc[0]
        kind = "published" if bool(row.official) else "retrained"
        heading = f"{kind.capitalize()} Chronos-Bolt-Tiny · P = {int(row.P)}, S = {int(row.S)}"
        if mode == "smoke":
            heading = "SMOKE TEST · " + heading
        note = ("Seven binary frequency tasks · grouped five-block prequential MDL\n"
                "First block uses the uniform code; subsequent blocks use only past frequencies.")
        figure = _plot_bars(subset, heading, note)
        model_paths = _save_figure(figure, out / "figures" / f"{prefix}sv_{tag}")
        plt.close(figure)
        paths.update({f"{tag}_{fmt}": path for fmt, path in model_paths.items()})
        if tag == REFERENCE:
            figure = _plot_bars(subset, heading, note)
            main = _save_figure(figure, out / f"{prefix}fig_probing_sv_p16_s16")
            plt.close(figure)
            paths.update({f"reference_{fmt}": path for fmt, path in main.items()})
    caption = (
        ("SMOKE TEST ONLY; reduced design, unsuitable for report insertion.\n" if mode == "smoke" else "")
        + "Space saving (SV) for seven hierarchical frequency-band tasks at decoder blocks 0–3 "
        "and the final output of the retrained P=S=16 Chronos-Bolt-Tiny (training seed 42). "
        "The full design sweeps 2–250 Hz in 1 Hz steps at fs=512 Hz with ten distinct phases. "
        "All phases of a frequency stay in one of five class-stratified prequential blocks. "
        "The first block is encoded uniformly; every later block is predicted using only "
        "preceding blocks, with training-only standardisation, PCA (up to 20 components), "
        "and logistic regression. SV=1−L/n; negative values indicate expansion. "
        "The uniform first block limits attainable SV to 1−n_first/n (approximately 0.8). "
        "Parentheses give SV after label permutation with seed 42. "
        "The grouped-bar layout follows Pagani et al., Fig. 3; this is the repository's "
        "adapted five-block MDL protocol. " + PAGANI_URL + "\n"
    )
    paths["caption"] = out / f"{prefix}fig_probing_sv_p16_s16_caption.txt"
    _text(paths["caption"], caption)
    return paths


def compression_summary(frame: pd.DataFrame, models: Iterable[Any], cfg: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the complete population and return counts plus continuous SV summaries.

    A full export requires every one of the 100 background IDs in every cell.
    No failed or absent case is removed from the denominator.
    """
    frame = _validated_numbers(frame)
    models = tuple(models)
    mode, n_backgrounds = cfg.mode, int(cfg.n_backgrounds)
    if mode not in {"full", "smoke"} or n_backgrounds < 1:
        raise ValueError("Invalid export mode or denominator")
    tags = [model.tag for model in models]
    if len(tags) != len(set(tags)):
        raise ValueError("Duplicate model specifications")
    if mode == "full" and (n_backgrounds != 100 or set(tags) != {spec.tag for spec in MODELS}):
        raise ValueError("Full compression counts require 100 backgrounds and all 16 checkpoints")
    for model in models:
        part = frame.loc[frame.model == model.tag]
        if not ((part.P == model.P) & (part.S == model.S) & (part.official == model.official)).all():
            raise ValueError(f"Checkpoint geometry/identity mismatch for {model.tag}")
        if part.checkpoint.nunique() != 1:
            raise ValueError(f"Expected one immutable checkpoint identity for {model.tag}")
    keys = ["model", "family", "background_id", "stage", "task"]
    expected = pd.MultiIndex.from_product(
        [tags, FAMILIES, range(n_backgrounds), STAGES, TASKS], names=keys)
    _require_product(frame, keys, expected)
    if mode == "full":
        expected_n = frame.task.map(lambda task: 10 * (TASK_BANDS[task][1] - TASK_BANDS[task][0] + 1))
        if not np.array_equal(frame.n, expected_n):
            raise ValueError("Full complex export requires all frequencies and ten phases")
    group_fields = ["model", "P", "S", "official", "checkpoint", "family", "stage", "task"]
    rows = []
    for identity, group in frame.groupby(group_fields, sort=False, dropna=False):
        values = group.SV.to_numpy(dtype=float)
        shuffled = group.SV_shuffled.to_numpy(dtype=float)
        row = dict(zip(group_fields, identity))
        row.update(n_backgrounds=len(group), compressed=int(np.count_nonzero(values > 0)),
                   shuffled_compressed=int(np.count_nonzero(shuffled > 0)),
                   SV_min=float(values.min()), SV_q25=float(np.quantile(values, .25)),
                   SV_median=float(np.median(values)), SV_q75=float(np.quantile(values, .75)),
                   SV_max=float(values.max()), SV_shuffled_median=float(np.median(shuffled)))
        rows.append(row)
    distribution = pd.DataFrame(rows)
    counts = distribution.loc[distribution.stage == "output_head", group_fields + [
        "n_backgrounds", "compressed", "shuffled_compressed"]].reset_index(drop=True)
    counts["compresses"] = counts.compressed.astype(str) + "/" + counts.n_backgrounds.astype(str)
    return counts, distribution


def _latex_table(counts: pd.DataFrame, models: tuple[Any, ...], denominator: int) -> str:
    lines = [r"\begin{table}[!tbp]", r"    \centering", r"    \small",
             r"    \setlength{\tabcolsep}{5pt}", r"    \begin{tabular}{@{}ccrrrrrrr@{}}", r"        \toprule"]
    for family_index, family in enumerate(FAMILIES):
        if family_index:
            lines.append(r"        \midrule")
        lines.append(r"        \multicolumn{9}{c}{\textbf{" + FAMILY_LABELS[family] + r"}} \\")
        lines.append(r"        \midrule")
        lines.append(r"        \textbf{$P$} & \textbf{$S$} & " + " & ".join(
            r"\textbf{" + task + "}" for task in TASKS) + r" \\")
        lines.append(r"        \midrule")
        previous_patch = None
        for model in models:
            if previous_patch is not None and model.P != previous_patch:
                lines.append(r"        \addlinespace[2pt]")
            part = counts.loc[(counts.family == family) & (counts.model == model.tag)].set_index("task")
            stride = f"${model.S}" + (r"^{\dagger}$" if model.official else "$")
            lines.append(f"        ${model.P}$ & {stride} & " + " & ".join(
                f"${int(part.loc[task, 'compressed'])}$" for task in TASKS) + r" \\")
            if model.official:
                lines.append(r"        \midrule")
            previous_patch = model.P
    lines.extend([r"        \bottomrule", r"    \end{tabular}",
                  r"\caption{Number of backgrounds out of $" + str(denominator) + r"$ for which the "
                  r"injected tone's band labels compress ($SV>0$) at the final output. "
                  r"Each background supplies a separate full frequency/phase sweep for each of the seven "
                  r"hierarchical tasks. Counts describe label compression, using five-block grouped "
                  r"prequential MDL; they do not measure recovery of a forecast component. "
                  r"$\dagger$ marks the published Chronos-Bolt-Tiny checkpoint.}",
                  r"    \label{tab:probingCompression}", r"\end{table}", ""])
    return "\n".join(lines)


def _plot_distribution(frame: pd.DataFrame, title: str):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(12, 8.8), sharex=True, sharey=True)
    positions = np.arange(len(TASKS))
    width = .13
    for axis, family in zip(axes, FAMILIES):
        part = frame.loc[frame.family == family]
        for stage_index, stage in enumerate(STAGES):
            values = [part.loc[(part.stage == stage) & (part.task == task), "SV"].to_numpy() for task in TASKS]
            boxes = axis.boxplot(values, positions=positions + (stage_index - 2) * width,
                                 widths=width * .90, patch_artist=True, showfliers=True,
                                 flierprops={"markersize": 2.5, "alpha": .45},
                                 medianprops={"color": "#222222", "linewidth": 1})
            for patch in boxes["boxes"]:
                patch.set_facecolor(COLOURS[stage_index])
                patch.set_hatch(HATCHES[stage_index])
                patch.set_linewidth(.6)
        axis.axhline(0, color="#555555", linewidth=.8)
        axis.grid(axis="y", color="#dddddd", linewidth=.5)
        axis.set_axisbelow(True)
        axis.set_title(FAMILY_LABELS[family], loc="left", fontsize=11)
        axis.set_ylabel("Space saving, SV")
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(positions, TASKS)
    axes[-1].set_xlabel("Hierarchical frequency-band task")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLOURS[i], hatch=HATCHES[i], edgecolor="#333333", label=stage.replace("dec_", "Decoder ").replace("output_head", "Final output"))
               for i, stage in enumerate(STAGES)]
    figure.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, .027))
    figure.suptitle(title, x=.09, ha="left", fontsize=13)
    figure.text(.09, .008, "Boxes: median and interquartile range across backgrounds; whiskers: 1.5 × IQR; points: remaining draws.", fontsize=8.2)
    figure.subplots_adjust(left=.09, right=.98, top=.92, bottom=.13, hspace=.20)
    return figure


def export_complex(frame: pd.DataFrame, out: Path, models: Iterable[Any], cfg: Any) -> dict[str, Path]:
    """Export complete counts and continuous SV results; smoke runs never emit report LaTeX."""
    import matplotlib.pyplot as plt

    models = tuple(models)
    counts, distribution = compression_summary(frame, models, cfg)
    out = Path(out)
    prefix = "smoke_" if cfg.mode == "smoke" else ""
    paths = {
        "results_csv": out / f"{prefix}complex_sv.csv",
        "counts_csv": out / f"{prefix}compression_counts.csv",
        "distribution_csv": out / f"{prefix}sv_distribution.csv",
    }
    _csv(paths["results_csv"], frame)
    _csv(paths["counts_csv"], counts)
    _csv(paths["distribution_csv"], distribution)
    for family in FAMILIES:
        part = counts.loc[counts.family == family]
        panel = part.pivot(index="model", columns="task", values="compressed").reindex(
            index=[model.tag for model in models], columns=TASKS)
        panel = panel.reset_index()
        panel.insert(1, "denominator", cfg.n_backgrounds)
        paths[f"{family}_table_csv"] = out / f"{prefix}compression_{family}.csv"
        _csv(paths[f"{family}_table_csv"], panel)
    if cfg.mode == "full":
        paths["table_tex"] = out / "probing_compression.tex"
        _text(paths["table_tex"], _latex_table(counts, models, cfg.n_backgrounds))
    ref = frame.loc[frame.model == REFERENCE]
    if not ref.empty:
        heading = "SV distributions · retrained P = S = 16"
        if cfg.mode == "smoke":
            heading = f"SMOKE TEST · {cfg.n_backgrounds} backgrounds per family · " + heading
        figure = _plot_distribution(ref, heading)
        figure_paths = _save_figure(figure, out / f"{prefix}sv_distribution_p16_s16")
        plt.close(figure)
        paths.update({f"distribution_{fmt}": path for fmt, path in figure_paths.items()})
    paths["caption"] = out / f"{prefix}probing_compression_caption.txt"
    _text(paths["caption"], (
        ("SMOKE TEST ONLY; reduced design, unsuitable for report insertion.\n" if cfg.mode == "smoke" else "")
        + f"Each table entry counts SV>0 among {cfg.n_backgrounds} distinct backgrounds of the named "
        "family, at the final model output. Each background generates a separate labelled sweep; "
        "this is label compression for the added tone, not compression of one waveform. "
        "Seven fixed band tasks are evaluated with five-block grouped prequential MDL. "
        "No background is dropped from the denominator. Full five-stage results and shuffled-label "
        "controls are retained in complex_sv.csv and sv_distribution.csv.\n"))
    return paths
