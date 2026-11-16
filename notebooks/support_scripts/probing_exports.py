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


STAGES = ("enc_0", "enc_1", "enc_2", "enc_3",
          "dec_0", "dec_1", "dec_2", "dec_3", "output_reg", "output_head")
STAGE_LABELS = {
    "enc_0": "Encoder 1", "enc_1": "Encoder 2", "enc_2": "Encoder 3", "enc_3": "Encoder 4",
    "dec_0": "Decoder 1", "dec_1": "Decoder 2", "dec_2": "Decoder 3", "dec_3": "Decoder 4",
    "output_reg": "Projection input", "output_head": "Output quantile head",
}
STAGE_SHORT = {"enc_0": "E1", "enc_1": "E2", "enc_2": "E3", "enc_3": "E4",
               "dec_0": "D1", "dec_1": "D2", "dec_2": "D3", "dec_3": "D4",
               "output_reg": "In", "output_head": "Head"}
NETWORKS = ("trained", "random_init")
RANDOM_INIT = ("L_random_init_bits", "SV_random_init")
# Trial tones closer than this to a boundary of a task on their path are set
# aside in the off-boundary reading, since a band probe is weakest at its own cut.
BOUNDARY_GUARD_HZ = 4.0
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


def _validated_numbers(frame: pd.DataFrame, random_init: bool | None = None) -> pd.DataFrame:
    """Check codelengths; `random_init` True requires the untrained control, None accepts either."""
    has_control = set(RANDOM_INIT).issubset(frame.columns)
    if random_init and not has_control:
        raise ValueError("Missing the random-initialisation control columns")
    if has_control:
        control = frame[list(RANDOM_INIT)].apply(pd.to_numeric, errors="raise")
        if (not np.isfinite(control.to_numpy(float)).all() or (control.L_random_init_bits < 0).any()
                or not np.allclose(control.SV_random_init,
                                   1 - control.L_random_init_bits / pd.to_numeric(frame.n),
                                   rtol=0, atol=1e-10)):
            raise ValueError("SV_random_init does not match its codelength")
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


def _plot_bars(frame: pd.DataFrame, watermark: str | None = None):
    """Ten stages by seven tasks; no title or note inside the image (they belong to the caption)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    has_control = "SV_random_init" in frame.columns
    figure, axis = plt.subplots(figsize=(11.2, 15.5))
    height = 0.105
    centres = np.arange(len(STAGES))
    # Only true-task values are drawn as bars; the controls are text labels.
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
            label = f"{row.SV:+.3f} ({row.SV_shuffled:+.3f}"
            label += f"; {row.SV_random_init:+.3f})" if has_control else ")"
            axis.text(label_x, y, label, va="center", ha="left", fontsize=7.6, family="monospace")
    for divider in (3.5, 7.5):
        axis.axhline(divider, color="#999999", linewidth=.6, linestyle=":")
    axis.axvline(0, color="#333333", linewidth=.8)
    axis.set_xlim(lower, upper)
    axis.set_yticks(centres, [STAGE_LABELS[stage] for stage in STAGES])
    axis.invert_yaxis()
    axis.set_xlabel("Space saving, SV")
    axis.text(label_x, -.62, "SV (shuffled; untrained)" if has_control else "SV (shuffled)",
              ha="left", va="bottom", fontsize=8.6)
    if watermark:
        axis.text(.5, .5, watermark, transform=axis.transAxes, ha="center", va="center",
                  fontsize=46, color="#cc0000", alpha=.18, rotation=30)
    axis.grid(axis="x", color="#dddddd", linewidth=.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    handles = [Patch(facecolor=colour, edgecolor="#333333", hatch=hatch, label=task)
               for task, colour, hatch in zip(TASKS, COLOURS, HATCHES)]
    figure.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
                  bbox_to_anchor=(.45, .035))
    figure.subplots_adjust(left=.16, right=.68, bottom=.085, top=.975)
    return figure


# --------------------------------------------------------------------------- #
#  Lock/control reading
# --------------------------------------------------------------------------- #
TRIAL_FIELDS = ("model", "P", "S", "official", "family", "background_id", "network", "stage",
                "pair", "arm", "frequency", "lock_hz", "branch", "boundary_distance",
                "n_contexts", "located", "level1", "level2", "level3", "bits_per_context")


def lock_control_summary(trials: pd.DataFrame) -> pd.DataFrame:
    """Located rate at candidate sites and beside them, per checkpoint, family, network, stage.

    Rates are percentages of contexts whose three band labels are all returned.
    Net is lock minus control in percentage points.  The *_off columns repeat the
    reading on pairs whose lock and control tones all lie at least BOUNDARY_GUARD_HZ
    from the task boundaries on their path, where a boundary cannot pass for a site.
    No pair is removed from the all-pairs columns.
    """
    missing = set(TRIAL_FIELDS) - set(trials.columns)
    if missing or trials.empty:
        raise ValueError(f"Invalid lock/control results: missing {sorted(missing)}")
    frame = trials.copy()
    if set(frame.arm) != {"lock", "control"} or not set(frame.stage) <= set(STAGES):
        raise ValueError("Lock/control results need both arms at registered stages")
    counts = frame[["located", "n_contexts"]].to_numpy(float)
    if not np.isfinite(counts).all() or (counts[:, 0] < 0).any() or (counts[:, 0] > counts[:, 1]).any():
        raise ValueError("Invalid located counts")
    pair_keys = ["model", "family", "background_id", "pair"]
    frame["off_boundary"] = frame.groupby(pair_keys).boundary_distance.transform("min") >= BOUNDARY_GUARD_HZ
    keys = ["model", "P", "S", "official", "family", "network", "stage"]
    rows = []
    for identity, group in frame.groupby(keys, sort=False):
        row = dict(zip(keys, identity))
        for suffix, part in (("", group), ("_off", group.loc[group.off_boundary])):
            for arm in ("lock", "control"):
                side = part.loc[part.arm == arm]
                n = int(side.n_contexts.sum())
                row[f"{arm}_n{suffix}"] = n
                row[f"{arm}_pct{suffix}"] = 100 * side.located.sum() / n if n else np.nan
                row[f"{arm}_bits{suffix}"] = (float(np.average(side.bits_per_context, weights=side.n_contexts))
                                               if n else np.nan)
            row[f"net_pct{suffix}"] = row[f"lock_pct{suffix}"] - row[f"control_pct{suffix}"]
            row[f"pairs{suffix}"] = int(part[pair_keys].drop_duplicates().shape[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _geometry_rows(models: tuple[Any, ...], cells) -> list[str]:
    lines, previous_patch = [], None
    for model in models:
        if previous_patch is not None and model.P != previous_patch and not model.official:
            lines.append(r"        \addlinespace[2pt]")
        stride = f"${model.S}" + (r"^{\dagger}$" if model.official else "$")
        lines.append(f"        ${model.P}$ & {stride} & " + " & ".join(cells(model)) + r" \\")
        if model.official:
            lines.append(r"        \midrule")
        previous_patch = model.P
    return lines


def _latex_lock_control(summary: pd.DataFrame, models: tuple[Any, ...], families: tuple[str, ...],
                        caption: str, label: str, suffix: str = "") -> str:
    """Rows are geometries, columns the ten stages, cells lock/control located rates in per cent."""
    width = 2 + len(STAGES)
    lines = [r"\begin{table*}[!tbp]", r"    \centering", r"    \small",
             r"    \setlength{\tabcolsep}{3.5pt}",
             r"    \begin{tabular}{@{}cc" + "c" * len(STAGES) + r"@{}}", r"        \toprule"]
    trained = summary.loc[summary.network == "trained"]
    for index, family in enumerate(families):
        if len(families) > 1:
            if index:
                lines.append(r"        \midrule")
            lines.append(rf"        \multicolumn{{{width}}}{{c}}{{\textbf{{{FAMILY_LABELS[family]}}}}} \\")
            lines.append(r"        \midrule")
        if index == 0 or len(families) > 1:
            lines.append(r"        \textbf{$P$} & \textbf{$S$} & " + " & ".join(
                r"\textbf{" + STAGE_SHORT[stage] + "}" for stage in STAGES) + r" \\")
            lines.append(r"        \midrule")
        part = trained.loc[trained.family == family].set_index(["model", "stage"])

        def cells(model, part=part):
            out = []
            for stage in STAGES:
                row = part.loc[(model.tag, stage)]
                lock, control = row[f"lock_pct{suffix}"], row[f"control_pct{suffix}"]
                out.append("--" if not np.isfinite(lock + control) else f"${lock:.0f}/{control:.0f}$")
            return out
        lines.extend(_geometry_rows(models, cells))
    lines.extend([r"        \bottomrule", r"    \end{tabular}", r"    \caption{" + caption + "}",
                  r"    \label{" + label + "}", r"\end{table*}", ""])
    return "\n".join(lines)


def _latex_stage_medians(frame: pd.DataFrame, models: tuple[Any, ...]) -> str:
    """Median SV over the seven tasks at each stage, trained and untrained network."""
    panels = [("SV", "Trained network")]
    if "SV_random_init" in frame.columns:
        panels.append(("SV_random_init", "Untrained network"))
    width = 2 + len(STAGES)
    lines = [r"\begin{table*}[!tbp]", r"    \centering", r"    \small",
             r"    \setlength{\tabcolsep}{3.5pt}",
             r"    \begin{tabular}{@{}cc" + "r" * len(STAGES) + r"@{}}", r"        \toprule"]
    for index, (column, title) in enumerate(panels):
        if index:
            lines.append(r"        \midrule")
        lines.append(rf"        \multicolumn{{{width}}}{{c}}{{\textbf{{{title}}}}} \\")
        lines.append(r"        \midrule")
        lines.append(r"        \textbf{$P$} & \textbf{$S$} & " + " & ".join(
            r"\textbf{" + STAGE_SHORT[stage] + "}" for stage in STAGES) + r" \\")
        lines.append(r"        \midrule")
        median = frame.groupby(["model", "stage"])[column].median()
        lines.extend(_geometry_rows(models, lambda model: [
            f"${median.loc[(model.tag, stage)]:.2f}$" for stage in STAGES]))
    lines.extend([
        r"        \bottomrule", r"    \end{tabular}",
        r"    \caption{Median space saving over the seven band tasks at each probe point, pure tones. "
        r"E1--E4 and D1--D4 are the encoder and decoder blocks, In the input of the output projection "
        r"and Head the output quantile head. The lower panel repeats the reading on the same "
        r"architecture with untrained weights. $\dagger$ marks the published checkpoint.}",
        r"    \label{tab:probingStages}", r"\end{table*}", ""])
    return "\n".join(lines)


def export_pure(frame: pd.DataFrame, out: Path, mode: str = "full",
                trials: pd.DataFrame | None = None) -> dict[str, Path]:
    """Export the SV figure of every checkpoint, the stage medians and the lock/control table."""
    import matplotlib.pyplot as plt

    if mode not in {"full", "smoke"}:
        raise ValueError("Expected full or smoke mode")
    frame = _validated_numbers(frame, random_init=(mode == "full"))
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
        if trials is None:
            raise ValueError("Full pure export requires the lock/control results")
    out = Path(out)
    prefix = "smoke_" if mode == "smoke" else ""
    watermark = "SMOKE TEST" if mode == "smoke" else None
    models = tuple(spec for spec in MODELS if spec.tag in model_tags)
    paths: dict[str, Path] = {"results_csv": out / f"{prefix}pure_sv.csv"}
    _csv(paths["results_csv"], frame)
    columns = ["SV", "SV_shuffled"] + (["SV_random_init"] if "SV_random_init" in frame else [])
    medians = frame.groupby(["model", "P", "S", "official", "stage"], sort=False)[columns].median()
    medians = medians.add_suffix("_median").reset_index()
    medians["positive_tasks"] = frame.groupby(["model", "stage"], sort=False).SV.agg(
        lambda values: int((values > 0).sum())).to_numpy()
    paths["stage_csv"] = out / f"{prefix}pure_sv_stage_median.csv"
    _csv(paths["stage_csv"], medians)
    for tag, subset in frame.groupby("model", sort=False):
        figure = _plot_bars(subset, watermark)
        model_paths = _save_figure(figure, out / "figures" / f"{prefix}sv_{tag}")
        plt.close(figure)
        paths.update({f"{tag}_{fmt}": path for fmt, path in model_paths.items()})
        if tag == REFERENCE:
            figure = _plot_bars(subset, watermark)
            main = _save_figure(figure, out / f"{prefix}fig_probing_sv_p16_s16")
            plt.close(figure)
            paths.update({f"reference_{fmt}": path for fmt, path in main.items()})
    caption = (
        ("SMOKE TEST ONLY; reduced design, unsuitable for report insertion.\n" if mode == "smoke" else "")
        + "Space saving of the seven band tasks at the ten probe points, reference geometry "
        "$P=S=16$. Bars give $\\mathrm{SV}=1-L/n$ under the five-block prequential code; values in "
        "parentheses are the shuffled-label and untrained-network controls. The first block is "
        "coded uniformly, which caps $\\mathrm{SV}$ near $0.8$.\n")
    paths["caption"] = out / f"{prefix}fig_probing_sv_p16_s16_caption.txt"
    _text(paths["caption"], caption)
    if trials is not None:
        summary = lock_control_summary(trials)
        paths["lock_control_trials_csv"] = out / f"{prefix}pure_lock_control_trials.csv"
        paths["lock_control_csv"] = out / f"{prefix}pure_lock_control.csv"
        _csv(paths["lock_control_trials_csv"], trials)
        _csv(paths["lock_control_csv"], summary)
        if mode == "full":
            paths["stage_tex"] = out / "probing_sv_stages.tex"
            _text(paths["stage_tex"], _latex_stage_medians(frame, models))
            paths["lock_control_tex"] = out / "probing_lock_control.tex"
            _text(paths["lock_control_tex"], _latex_lock_control(
                summary, models, ("pure",),
                r"Pure tones located at each probe point, in per cent of the contexts, at the candidate "
                r"sites of each geometry and at the controls beside them (lock/control). A tone is "
                r"located when the three band probes on its path all return its label; the probes are "
                r"fitted without the frequencies within $1$\,Hz of the tone. E1--E4 and D1--D4 are the "
                r"encoder and decoder blocks, In the input of the output projection and Head the output "
                r"quantile head. $\dagger$ marks the published checkpoint.",
                "tab:probingLockControl"))
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
        if "SV_random_init" in group:
            row["SV_random_init_median"] = float(group.SV_random_init.median())
        rows.append(row)
    distribution = pd.DataFrame(rows)
    counts = distribution.loc[distribution.stage == "output_head", group_fields + [
        "n_backgrounds", "compressed", "shuffled_compressed"]].reset_index(drop=True)
    counts["compresses"] = counts.compressed.astype(str) + "/" + counts.n_backgrounds.astype(str)
    return counts, distribution


def _latex_table(distribution: pd.DataFrame, models: tuple[Any, ...], denominator: int) -> str:
    """Output-head SV across backgrounds: median with the quartiles as sub- and superscript.

    A count of SV > 0 saturates near the denominator; the continuous summary does not.
    """
    head = distribution.loc[distribution.stage == "output_head"]
    lines = [r"\begin{table*}[!tbp]", r"    \centering", r"    \small",
             r"    \setlength{\tabcolsep}{5pt}", r"    \begin{tabular}{@{}ccrrrrrrr@{}}", r"        \toprule"]
    for family_index, family in enumerate(FAMILIES):
        if family_index:
            lines.append(r"        \midrule")
        lines.append(r"        \multicolumn{9}{c}{\textbf{" + FAMILY_LABELS[family] + r"}} \\")
        lines.append(r"        \midrule")
        lines.append(r"        \textbf{$P$} & \textbf{$S$} & " + " & ".join(
            r"\textbf{" + task + "}" for task in TASKS) + r" \\")
        lines.append(r"        \midrule")
        part = head.loc[head.family == family].set_index(["model", "task"])
        lines.extend(_geometry_rows(models, lambda model, part=part: [
            "${:.2f}_{{{:.2f}}}^{{{:.2f}}}$".format(*part.loc[(model.tag, task),
                                                             ["SV_median", "SV_q25", "SV_q75"]])
            for task in TASKS]))
    lines.extend([r"        \bottomrule", r"    \end{tabular}",
                  r"    \caption{Space saving of the band labels of the added tone at the output quantile "
                  r"head, across $" + str(denominator) + r"$ backgrounds per generator: median, with the "
                  r"first and third quartiles as subscript and superscript. Each background supplies its "
                  r"own frequency and phase sweep. The values describe how readable the band of the tone "
                  r"is, not whether the forecast returns it. $\dagger$ marks the published checkpoint.}",
                  r"    \label{tab:probingCompression}", r"\end{table*}", ""])
    return "\n".join(lines)


def _plot_distribution(frame: pd.DataFrame, watermark: str | None = None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure, axes = plt.subplots(2, 1, figsize=(13.5, 8.8), sharex=True, sharey=True)
    positions = np.arange(len(STAGES))
    width = .115
    for axis, family in zip(axes, FAMILIES):
        part = frame.loc[frame.family == family]
        for task_index, task in enumerate(TASKS):
            values = [part.loc[(part.stage == stage) & (part.task == task), "SV"].to_numpy() for stage in STAGES]
            boxes = axis.boxplot(values, positions=positions + (task_index - 3) * width,
                                 widths=width * .90, patch_artist=True, showfliers=True,
                                 flierprops={"markersize": 2.2, "alpha": .4},
                                 medianprops={"color": "#222222", "linewidth": 1})
            for patch in boxes["boxes"]:
                patch.set_facecolor(COLOURS[task_index])
                patch.set_hatch(HATCHES[task_index])
                patch.set_linewidth(.6)
        axis.axhline(0, color="#555555", linewidth=.8)
        axis.grid(axis="y", color="#dddddd", linewidth=.5)
        axis.set_axisbelow(True)
        axis.set_title(FAMILY_LABELS[family], loc="left", fontsize=11)
        axis.set_ylabel("Space saving, SV")
        axis.spines[["top", "right"]].set_visible(False)
        if watermark:
            axis.text(.5, .5, watermark, transform=axis.transAxes, ha="center", va="center",
                      fontsize=40, color="#cc0000", alpha=.18)
    axes[-1].set_xticks(positions, [STAGE_LABELS[stage] for stage in STAGES], rotation=20, ha="right")
    handles = [Patch(facecolor=COLOURS[i], hatch=HATCHES[i], edgecolor="#333333", label=task)
               for i, task in enumerate(TASKS)]
    figure.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(.5, .0))
    figure.subplots_adjust(left=.07, right=.99, top=.95, bottom=.17, hspace=.16)
    return figure


def export_complex(frame: pd.DataFrame, out: Path, models: Iterable[Any], cfg: Any,
                   trials: pd.DataFrame | None = None) -> dict[str, Path]:
    """Export continuous SV summaries, counts and the lock/control table; smoke runs emit no LaTeX."""
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
        _text(paths["table_tex"], _latex_table(distribution, models, cfg.n_backgrounds))
    if trials is not None:
        summary = lock_control_summary(trials)
        paths["lock_control_trials_csv"] = out / f"{prefix}complex_lock_control_trials.csv"
        paths["lock_control_csv"] = out / f"{prefix}complex_lock_control.csv"
        _csv(paths["lock_control_trials_csv"], trials)
        _csv(paths["lock_control_csv"], summary)
        if cfg.mode == "full":
            if not (summary.pairs == cfg.n_backgrounds).all():
                raise ValueError("Every lock/control cell needs one pair per background")
            paths["lock_control_tex"] = out / "probing_lock_control_backgrounds.tex"
            _text(paths["lock_control_tex"], _latex_lock_control(
                summary, models, FAMILIES,
                r"Added tones located at each probe point, in per cent of the contexts, when the tone "
                r"sits on a candidate site and when it sits beside it (lock/control), over $"
                + str(cfg.n_backgrounds) + r"$ backgrounds per generator. Sites and controls are those "
                r"of the recovery count (\autoref{tab:decomposition}); each tone is read at ten phases "
                r"by probes fitted on the sweep of its own background, without the frequencies within "
                r"$1$\,Hz of it. Columns as in \autoref{tab:probingLockControl}. $\dagger$ marks the "
                r"published checkpoint.", "tab:probingLockControlBg"))
    ref = frame.loc[frame.model == REFERENCE]
    if not ref.empty:
        figure = _plot_distribution(ref, "SMOKE TEST" if cfg.mode == "smoke" else None)
        figure_paths = _save_figure(figure, out / f"{prefix}sv_distribution_p16_s16")
        plt.close(figure)
        paths.update({f"distribution_{fmt}": path for fmt, path in figure_paths.items()})
    paths["caption"] = out / f"{prefix}probing_compression_caption.txt"
    _text(paths["caption"], (
        ("SMOKE TEST ONLY; reduced design, unsuitable for report insertion.\n" if cfg.mode == "smoke" else "")
        + f"Space saving of the band labels of the added tone across {cfg.n_backgrounds} backgrounds "
        "per generator, reference geometry $P=S=16$, at the ten probe points. Boxes span the "
        "quartiles with the median marked; whiskers reach 1.5 times the interquartile range. "
        "Each background supplies its own frequency and phase sweep.\n"))
    return paths
