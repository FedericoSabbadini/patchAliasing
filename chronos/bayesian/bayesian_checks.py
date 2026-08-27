"""Shareable scientific-validity gates for ``bayesian_analysis.ipynb``.

These functions deliberately separate numerical results from the rules that decide whether a
result is reportable.  Smoke/synthetic parameter-recovery runs can exercise the rules, but their
outputs are never promoted to Deliverable 3 evidence.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd


RHAT_MAX = 1.01
ESS_MIN = 1_000
MAX_DIVERGENCES = 0
POSTERIOR_CUTOFF = 0.95


def fit_diagnostics(name: str, idata: Any, az_module: Any) -> dict[str, Any]:
    """Evaluate the convergence gate specified in Deliverable 3."""
    summary = az_module.summary(idata, kind="diagnostics")
    max_rhat = float(np.nanmax(summary["r_hat"]))
    min_bulk = float(np.nanmin(summary["ess_bulk"]))
    min_tail = float(np.nanmin(summary["ess_tail"]))
    divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    finite = bool(np.isfinite([max_rhat, min_bulk, min_tail]).all())
    ok = bool(
        finite
        and max_rhat < RHAT_MAX
        and min_bulk > ESS_MIN
        and min_tail > ESS_MIN
        and divergences == MAX_DIVERGENCES
    )
    return {
        "fit": name,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_bulk,
        "min_ess_tail": min_tail,
        "divergences": divergences,
        "diagnostics_ok": ok,
    }


def diagnostics_table(fits: Mapping[str, Any], az_module: Any) -> pd.DataFrame:
    return pd.DataFrame([fit_diagnostics(name, idata, az_module) for name, idata in fits.items()])


def loo_compare(
    fits: Mapping[str, Any], az_module: Any
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare fits and gate the comparison on PSIS Pareto-k reliability."""
    if len(fits) < 2:
        raise ValueError("LOO comparison needs at least two fitted models")
    loo_objects: dict[str, Any] = {}
    quality_rows: list[dict[str, Any]] = []
    for name, idata in fits.items():
        loo = az_module.loo(idata, pointwise=True)
        pareto_k = np.asarray(loo.pareto_k, dtype=float)
        max_k = float(np.nanmax(pareto_k))
        good_k = float(loo.good_k)
        warning = bool(loo.warning)
        reliable = bool(np.isfinite(max_k) and not warning and max_k <= good_k)
        loo_objects[name] = loo
        quality_rows.append(
            {
                "fit": name,
                "max_pareto_k": max_k,
                "good_k": good_k,
                "loo_warning": warning,
                "loo_reliable": reliable,
            }
        )

    quality = pd.DataFrame(quality_rows).set_index("fit")
    comparison = az_module.compare(loo_objects, ic="loo")
    if "elpd_diff" not in comparison or "dse" not in comparison:
        raise ValueError("ArviZ comparison does not expose elpd_diff and dse")
    winner = str(comparison.index[0])
    runner_up = str(comparison.index[1])
    margin = float(comparison.iloc[1]["elpd_diff"])
    dse = float(comparison.iloc[1]["dse"])
    reliable = bool(quality["loo_reliable"].all())
    separated = bool(np.isfinite([margin, dse]).all() and margin >= 2.0 * dse)
    decision = {
        "winner": winner,
        "runner_up": runner_up,
        "elpd_margin": margin,
        "dse": dse,
        "loo_reliable": reliable,
        "separated_2dse": separated,
        "comparison_ok": reliable and separated,
    }
    return comparison, quality, decision


def required_loo_win(decision: Mapping[str, Any], required_winner: str) -> bool:
    return bool(decision.get("comparison_ok") and decision.get("winner") == required_winner)


def three_way_verdict(
    support_probability: float,
    refute_probability: float | None,
    gate_ok: bool,
    cutoff: float = POSTERIOR_CUTOFF,
) -> str:
    """Support/refute/inconclusive rule with a fail-closed scientific-validity gate."""
    if not gate_ok:
        return "NOT REPORTABLE"
    if np.isfinite(support_probability) and support_probability >= cutoff:
        return "supported"
    if (
        refute_probability is not None
        and np.isfinite(refute_probability)
        and refute_probability >= cutoff
    ):
        return "refuted"
    return "inconclusive"


def joint_support_verdict(
    posterior_probability: float,
    comparison_won: bool,
    gate_ok: bool,
    cutoff: float = POSTERIOR_CUTOFF,
) -> str:
    """Verdict for hypotheses whose support rule requires posterior mass AND a LOO win."""
    if not gate_ok:
        return "NOT REPORTABLE"
    return "supported" if posterior_probability >= cutoff and comparison_won else "inconclusive"


def posterior_probability(idata: Any, variable: str, predicate: Callable[[np.ndarray], Any]) -> float:
    values = np.asarray(idata.posterior[variable], dtype=float)
    return float(np.mean(predicate(values)))


def parameter_recovery_table(
    recovered: Mapping[str, Mapping[str, tuple[float, float, float]]],
    truths: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Build a table of synthetic-recovery coverage from median/HDI triples.

    ``recovered[model][parameter]`` is ``(median, hdi_low, hdi_high)``.  This table is a method
    validation artifact only; callers must never present its numbers as empirical findings.
    """
    rows: list[dict[str, Any]] = []
    for model, parameters in truths.items():
        for parameter, truth in parameters.items():
            median, low, high = recovered[model][parameter]
            rows.append(
                {
                    "model": model,
                    "parameter": parameter,
                    "truth": float(truth),
                    "median": float(median),
                    "hdi_low": float(low),
                    "hdi_high": float(high),
                    "covered": bool(low <= truth <= high),
                }
            )
    return pd.DataFrame(rows)


def recovery_gate(table: pd.DataFrame, required_models: set[str]) -> bool:
    """Require every named model/parameter truth to be covered by its recovery HDI."""
    if table.empty or not required_models.issubset(set(table["model"])):
        return False
    required = table[table["model"].isin(required_models)]
    return bool(required["covered"].notna().all() and required["covered"].all())


def ppc_gate(
    table: pd.DataFrame, required_strata: set[str], minimum_coverage: float = 0.90
) -> bool:
    """Require every requested stratum and at least ``minimum_coverage`` interval coverage."""
    if table.empty or not {"stratum", "ppc_ok"}.issubset(table.columns):
        return False
    observed = set(table["stratum"])
    required = table[table["stratum"].isin(required_strata)]
    return bool(
        required_strata.issubset(observed)
        and len(required) > 0
        and float(required["ppc_ok"].astype(bool).mean()) >= minimum_coverage
    )


def sensitivity_gate(
    table: pd.DataFrame, probability_columns: tuple[str, ...], max_spread: float = 0.10
) -> bool:
    """Require prior/likelihood sensitivity probabilities to remain within a stated band."""
    if table.empty:
        return False
    for column in probability_columns:
        if column not in table or not np.isfinite(table[column]).all():
            return False
        if float(table[column].max() - table[column].min()) > max_spread:
            return False
    return True


def identified_branches(
    sites: pd.DataFrame, minimum_sites: int = 10, modes: tuple[str, ...] = ("tsmixup", "kernelsynth")
) -> dict[str, bool]:
    """Apply Deliverable 3's minimum-ten-unambiguous-sites bar to mean collapse curves."""
    required = {"branch", "n_sites", "rep", "mode"}
    if sites.empty or not required.issubset(sites.columns):
        return {"stride": False, "patch": False}
    mean_curves = sites[(sites["rep"] == -1) & sites["mode"].isin(modes)]
    totals = mean_curves.groupby("branch")["n_sites"].sum()
    return {branch: int(totals.get(branch, 0)) >= minimum_sites for branch in ("stride", "patch")}
