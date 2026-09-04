"""
collect.py, Part 2 of the Bayesian workflow: let Chronos make the observations.

This is the *only* place where a model is run. It turns the fifteen Deliverable 3 geometries into
tidy tables that the Bayesian notebook then treats as plain data, so Parts 1 and 3-5 of the
notebook need neither a GPU nor the `chronos` package.

    table            feeds                                    deliverable clause
   ,---------,---------------------------------,---------------------------------
    contrasts        Model A (H1 behavioural), Model C (H2)    Local Contrast Analysis, Eq. (8)-(9)
                                                               Phase-Invariance Analysis, Eq. (11)
    mdl_cells        Model B (H1 representational)             Absolute Performance Analysis, Eq. (10)
    mdl_bandtasks    descriptive cross-check only              Probing Methodology
    collapse         Model D1 (H3 location)                    Site-Geometry Analysis, Eq. (12)
    sites            Model D2 (H3 movement)                    Site-Geometry Analysis, Eq. (13)

Everything is sharded per geometry under `<out>/raw/`.  Atomic files and a design/checkpoint
manifest make resume fail closed: a shard is reused only when its hash, schema, design fingerprint
and immutable checkpoint identity all match.

    python -m collect --out ./bayes_data                 # all 15 geometries, full design
    python -m collect --out ./bayes_data --smoke         # tiny grid, minutes not tens of minutes
    python -m collect --out ./bayes_data --models p16-s16 p8-s8
    python -m collect --out ./bayes_data --force         # ignore existing shards
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as pl
import checkpointing as cp
import model_loader as ml

TABLES = ("contrasts", "mdl_cells", "mdl_bandtasks", "collapse", "sites")
RAW_TABLES = ("contrasts", "mdl_cells", "mdl_bandtasks", "collapse")
MANIFEST_NAME = "collection_manifest.json"
MANIFEST_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------------- #
#  Design configuration
# --------------------------------------------------------------------------------- #
@dataclass
class Config:
    """Every knob of the experimental design, in one place.

    The defaults are the FULL design described in the deliverable. `Config.smoke()` shrinks every
    grid so the whole pipeline can be exercised end-to-end in a few minutes; smoke output is
    written under a separate directory by the notebook so it can never be mistaken for a real run.
    """
    smoke: bool = False
    batch_size: int = 64
    device: str | None = None

    #, contrasts (H1 behavioural / H2),
    n_phase_contrast: int = 10      # phase offsets per lock; the S_f cap of Pagani et al. Eq. 6
    n_bg: int = 100                 # background draws per generator. Deliverable 3 specifies 100
                                    # TSMixup and 100 KernelSynth signals; these are those signals,
                                    # and they are also the u_background levels of Eq. (9).
    generators: tuple[str, ...] = pl.GENERATORS

    #, frequency-local MDL cells (H1 representational),
    #, localisation estimator (H1 behavioural / H2),
    fhat_topk: int = pl.FHAT_TOPK      # peaks retained per arm; set by the Part 2 Delta x K sweep
    fhat_tol_hz: float = pl.FHAT_TOL_HZ

    mdl_delta_f: float = 1.0        # the +/- offset of the two tones the probe must separate [Hz]
    mdl_n_per_class: int = 40       # examples per class in a cell (so 2 * 40 = 80 rows per probe)
    mdl_n_phase: int = 10
    mdl_n_bg: int = 10              # background draws entering an MDL cell

    #, band tasks (descriptive),
    band_tasks: bool = True
    bt_step: float = 2.0            # frequency step of the descriptive sweep [Hz]
    bt_n_phase: int = 4
    bt_random_init: bool = True     # untrained-architecture control, on a 4x coarser grid

    #, collapse sweep (H3),
    collapse_step: float = 1.0      # uniform part of the union grid [Hz]
    collapse_reps: int = 3          # phase (and background) replicates per frequency
    collapse_modes: tuple[str, ...] = ("pure", "tsmixup", "kernelsynth")
    site_merge_tol_hz: float = 1.5
    site_assignment_tol_hz: float = 1.5
    site_ambiguity_hz: float = 0.25
    min_d2_sites: int = 10          # Deliverable 3 identification bar, per branch
    min_sites_per_geometry: int = 2 # candidate sites that must survive the ceiling, per geometry

    seed: int = pl.SEED

    @staticmethod
    def smoke_cfg(**kw) -> "Config":
        base = dict(
            smoke=True, batch_size=32,
            n_phase_contrast=2, n_bg=1, generators=("tsmixup",),
            mdl_n_per_class=10, mdl_n_phase=3, mdl_n_bg=1,
            band_tasks=True, bt_step=32.0, bt_n_phase=2, bt_random_init=False,
            collapse_step=8.0, collapse_reps=1, collapse_modes=("pure", "tsmixup"),
        )
        base.update(kw)
        return Config(**base)


# --------------------------------------------------------------------------------- #
#  1. contrasts, the matched (f_k, f_k - delta, f_k + delta) triplets
# --------------------------------------------------------------------------------- #
def collect_contrasts(probe: "pl.Probe", cfg: Config) -> pd.DataFrame:
    """The localisation indicator h of deliverable Eq. (8), ONE ROW PER ARM.

    The response is whether the forecast rebuilt the frequency the arm carries:
    h = 1[|f_hat - f| <= 1 Hz], with f_hat the dominant frequency of the median forecast over the
    horizon (`pl.dominant_freq`). Models A and C are Bernoulli fits on h with a lock indicator, so
    the three arms of a triplet are three OBSERVATIONS rather than one contrast; this table is
    therefore long where it used to be wide, and carries `role`, `f` and `is_lock` per row.

    Nothing is discarded. An arm whose forecast rebuilds nothing measurable returns h = 0, which is
    a reading and not a degenerate one, so the old `live` filter and the epsilon stabiliser that
    kept a log-ratio finite are both gone. R is retained per arm because `pl.Probe.measure` gets it
    from the same forward pass at no extra cost, but no model is fitted to it.

    Design (deliverable Local Contrast Analysis): each lock frequency f_k in F_lock is paired with
    two controls at f_k +/- 0.25 fs/S. The three signals of a triplet share the SAME background
    realisation and the SAME phase, that is what makes the contrast paired, and it is the one
    place where this collection deliberately departs from `hypotheses.py`, which averages over
    phases drawn separately per frequency.

    Keeping the phase index is what makes H2 testable at all: Model C bins the phase circle and
    gives each bin its own offset, which is impossible once phases have been averaged away.
    """
    P, S = probe.P, probe.S

    # Per-site control offset (Deliverable 3, "What enters the inference"): the largest offset
    # not exceeding 0.25*fs/S that keeps BOTH controls clear of BOTH grids. A site is dropped only
    # if no such offset exists. The original rule fixed the offset at 0.25*fs/S; being defined
    # from the stride alone it cannot see the patch grid, and at P=16,S=12 it lands one control of
    # every stride-only site on a patch null, discarding all four.
    offsets = {f: pl.control_offset(P, S, f) for f in pl.f_lock(P, S)}
    sites = [f for f, d in offsets.items() if np.isfinite(d)]

    contexts, futures, freqs, meta = [], [], [], []
    for gen in cfg.generators:
        pool = pl.background_pool(gen, cfg.n_bg, pl.CTX + pl.PRED)
        for fk in sites:
            phases = pl.phases_Sf(fk, cfg.n_phase_contrast)
            for bg_id, bg in enumerate(pool):
                for ph_idx, ph in enumerate(phases):
                    d_fk = offsets[fk]
                    for role, f in (("lock", fk), ("lo", fk - d_fk), ("hi", fk + d_fk)):
                        # context and its TRUE continuation are slices of one long realisation,
                        # so the forecast target is the real future rather than a fresh draw
                        full = pl.build_context(bg, f, ph, pl.CTX + pl.PRED)
                        contexts.append(full[:pl.CTX])
                        futures.append(full[pl.CTX:])
                        freqs.append(f)
                        meta.append(dict(generator=gen, bg_id=bg_id, f_lock=fk, delta=d_fk,
                                         phase_idx=ph_idx, phase=float(ph), role=role,
                                         f=float(f)))

    if not contexts:
        return pd.DataFrame()

    # one forward pass, three readings; the indicator is the response, R is descriptive
    R, dphase, f_hat, f_hat_truth = probe.measure(
        np.stack(contexts), np.stack(futures), np.array(freqs), k=cfg.fhat_topk)

    out = pd.DataFrame(meta)
    out["R"] = R
    out["dphase"] = dphase
    out["f_hat"] = f_hat[:, 0]                   # strongest peak, for description
    out["h"] = pl.localisation_hit(f_hat, out["f"].to_numpy(float), tol=cfg.fhat_tol_hz)
    # the same estimator on the TRUE continuation: where this misses, the instrument failed on
    # that row and it carries no evidence about the model. Part 5 fits with and without the
    # conditioning and reports the LOO comparison rather than assuming which scope is right.
    out["h_truth"] = pl.localisation_hit(f_hat_truth, out["f"].to_numpy(float), tol=cfg.fhat_tol_hz)
    out["is_lock"] = (out["role"] == "lock").astype(np.int8)

    out["model"] = probe.tag
    out["P"], out["S"] = P, S
    out["overlap"] = (P - S) / P                 # O_c of Eq. (9)
    out["cpp"] = out["f_lock"] * P / pl.FS       # cycles per patch
    out["family"] = [pl.lock_family(f, P, S) for f in out["f_lock"]]
    return out[["model", "P", "S", "overlap", "generator", "bg_id", "f_lock", "family", "cpp",
                "delta", "phase_idx", "phase", "role", "f", "is_lock", "R", "dphase",
                "f_hat", "h", "h_truth"]]


# --------------------------------------------------------------------------------- #
#  2. mdl_cells, the frequency-local prequential codelength
# --------------------------------------------------------------------------------- #
def collect_mdl_cells(probe: "pl.Probe", cfg: Config) -> pd.DataFrame:
    """L(D) per (probe stage, candidate frequency), with the IsLocked flag of deliverable Eq. (10).

    Why a *local* task. The seven hierarchical band tasks span the whole operational band, so one
    codelength summarises hundreds of frequencies and the IsLocked indicator has nothing to attach
    to. Here each cell asks a single, sharply local question: from the 256-d [REG] vector alone,
    can the probe tell a tone at f_c - 1 Hz from a tone at f_c + 1 Hz? The prequential protocol,
    the probe pipeline and K are unchanged, only the scope narrows.

    The reading is direct: at a locked frequency whose neighbourhood has become linearly
    indistinguishable in representation space, the labels cost more bits.
    """
    P, S = probe.P, probe.S
    offsets = {f: pl.control_offset(P, S, f) for f in pl.f_lock(P, S)}
    sites = [f for f, d in offsets.items() if np.isfinite(d)]
    # exactly the frequencies the contrast design visits: every lock and both of its controls,
    # which makes the locked / non-locked split balanced 1:2 by construction
    centers = sorted({round(f, 6) for fk in sites
                      for f in (fk, fk - offsets[fk], fk + offsets[fk])})
    locks = pl.f_lock(P, S)

    rows = []
    for f_c in centers:
        X_all, y_all = [], []
        for label, f in ((0, f_c - cfg.mdl_delta_f), (1, f_c + cfg.mdl_delta_f)):
            combos = []
            for gen in cfg.generators:
                pool = pl.background_pool(gen, cfg.mdl_n_bg, pl.CTX)
                for bg in pool:
                    for ph in pl.phases_Sf(f, cfg.mdl_n_phase):
                        combos.append((bg, ph))
            # Spread the selection over the whole combo list rather than taking a prefix. The list
            # is background-major, so `combos[:n]` would silently restrict every cell to the first
            # few backgrounds and throw away the corpus variety the design is paying for.
            if len(combos) > cfg.mdl_n_per_class:
                idx = np.linspace(0, len(combos) - 1, cfg.mdl_n_per_class).round().astype(int)
                combos = [combos[i] for i in idx]
            ctx = np.stack([pl.build_context(bg, f, ph, pl.CTX) for bg, ph in combos])
            X_all.append(probe.capture_reg(ctx))
            y_all.append(np.full(len(ctx), label))

        y = np.concatenate(y_all)
        n = len(y)
        is_locked = int(any(abs(f_c - x) < 1e-6 for x in locks))
        dist = float(min(abs(f_c - x) for x in locks)) if locks else np.nan
        for stage in probe.stages:
            Xs = np.vstack([blk[stage] for blk in X_all])
            L = pl.mdl_codelength(Xs, y, seed=cfg.seed)
            rows.append(dict(model=probe.tag, P=P, S=S, stage=stage, f_center=float(f_c),
                             is_locked=is_locked, dist_to_lock=dist, n=n,
                             L_bits=L, L_uniform=float(n), SV=pl.space_saving(L, n),
                             family=pl.lock_family(f_c, P, S)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------- #
#  3. mdl_bandtasks, the seven global band tasks (descriptive cross-check)
# --------------------------------------------------------------------------------- #
def collect_band_tasks(probe: "pl.Probe", cfg: Config) -> pd.DataFrame:
    """The probing notebook's seven binary band tasks, per stage, with both controls.

    Not the response of any Bayesian model, it is retained because the deliverable's Probing
    Methodology promises it and because it answers a different question (is the band decodable at
    all, and is that decodability learned?) from the one Eq. (10) asks.
    """
    freqs = np.arange(pl.BAND[0], pl.BAND[1] + 1e-9, cfg.bt_step)
    gen = cfg.generators[0]
    pool = pl.background_pool(gen, max(cfg.n_bg, 2), pl.CTX)

    def sweep(pipe=None, freq_list=None, n_ph=None):
        fl = freqs if freq_list is None else freq_list
        nph = cfg.bt_n_phase if n_ph is None else n_ph
        ctx, lab = [], []
        for i, f in enumerate(fl):
            for j, ph in enumerate(pl.phases_Sf(f, nph)):
                ctx.append(pl.build_context(pool[(i + j) % len(pool)], f, ph, pl.CTX))
                lab.append(f)
        return probe.capture_reg(np.stack(ctx), pipe=pipe), np.asarray(lab)

    X, lab = sweep()
    rng = np.random.default_rng(cfg.seed)

    X_rand, lab_rand = (None, None)
    if cfg.bt_random_init:
        clone = probe.random_init_clone()
        coarse = np.arange(pl.BAND[0], pl.BAND[1] + 1e-9, cfg.bt_step * 4)
        X_rand, lab_rand = sweep(pipe=clone, freq_list=coarse, n_ph=max(2, cfg.bt_n_phase // 2))
        del clone

    rows = []
    for (name, lo, hi, bnd) in pl.BAND_TASKS:
        m = (lab >= lo) & (lab <= hi)
        y = (lab[m] > bnd).astype(int)
        if len(np.unique(y)) < 2:
            continue
        y_shuf = rng.permutation(y)
        for stage in probe.stages:
            Xs = X[stage][m]
            L = pl.mdl_codelength(Xs, y, seed=cfg.seed)
            L_shuf = pl.mdl_codelength(Xs, y_shuf, seed=cfg.seed)
            sv_rand = np.nan
            if X_rand is not None:
                mr = (lab_rand >= lo) & (lab_rand <= hi)
                yr = (lab_rand[mr] > bnd).astype(int)
                if len(np.unique(yr)) >= 2:
                    sv_rand = pl.space_saving(
                        pl.mdl_codelength(X_rand[stage][mr], yr, seed=cfg.seed), len(yr))
            rows.append(dict(model=probe.tag, P=probe.P, S=probe.S, stage=stage, task=name,
                             n=int(m.sum()), L_bits=L, SV=pl.space_saving(L, int(m.sum())),
                             SV_shuffled=pl.space_saving(L_shuf, int(m.sum())),
                             SV_random_init=sv_rand))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------- #
#  4. collapse, the token-collapse profile on the shared union grid
# --------------------------------------------------------------------------------- #
def collect_collapse(probe: "pl.Probe", cfg: Config) -> pd.DataFrame:
    """z_g(f): across-patch token dispersion, swept on the union grid of deliverable Eq. (12).

    The grid is the union of a uniform sweep with the predicted sites of EVERY geometry, not just
    this one. That is the whole point: a model must dip where its own geometry predicts and stay
    flat where a competing geometry predicts, otherwise the comb comparison M_S vs M_P could not
    distinguish them.

    Three signal modes are collected. On a pure sinusoid the collapse is exactly zero at a lock,
    the clean statement of the phenomenon. On a TSMixup or KernelSynth background it becomes a deep
    dip instead, because the background breaks patch identity; fitting the comb model to all three
    is what shows the conclusion is not an artefact of noise-free inputs.
    """
    grid = pl.union_grid(pl.MODELS, cfg.collapse_step)  # union over ALL geometries, not just the fitted ones
    rows = []
    for mode in cfg.collapse_modes:
        for rep in range(cfg.collapse_reps):
            bg = None
            if mode != "pure":
                bg = pl.background_pool(mode, cfg.collapse_reps, pl.CTX)[rep]
            ctx = np.stack([
                pl.build_context(bg, f, pl.phases_Sf(f, cfg.collapse_reps)[
                    rep % len(pl.phases_Sf(f, cfg.collapse_reps))], pl.CTX)
                for f in grid])
            z = probe.collapse(ctx)
            rows.append(pd.DataFrame(dict(model=probe.tag, P=probe.P, S=probe.S, mode=mode,
                                          rep=rep, f=grid, z=z)))
    out = pd.concat(rows, ignore_index=True)
    # normalise each curve by its own off-lock median so dip DEPTH is comparable across geometries
    # and modes; Eq. (12) is fitted on log z_norm.
    out["z_norm"] = out.groupby(["model", "mode", "rep"])["z"].transform(
        lambda s: s / max(float(np.median(s)), 1e-12))
    return out


def derive_sites(collapse: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Detected collapse sites, split by branch, for the H3 movement models.

    Deliverable 3, H3: "Every detected dip is assigned to the branch that predicts it, and sites
    belonging to both are set aside. The fundamental of branch F is then compared with the spacing
    that branch predicts." A row is therefore (geometry, signal mode, replicate, BRANCH) and it
    carries the fundamental of that branch only. A single pooled spacing would be meaningless:
    F_lock is a union, and the union of two combs has two interlaced spacings rather than one.

    Sites are detected per replicate (giving the model its residual variance) and once more on the
    replicate-averaged curve, recorded as rep = -1.
    """
    cfg = cfg or Config()
    rows = []
    for (model, mode), g in collapse.groupby(["model", "mode"]):
        P, S = int(g["P"].iloc[0]), int(g["S"].iloc[0])
        pieces = [(r, sub) for r, sub in g.groupby("rep")]
        mean_curve = g.groupby("f", as_index=False)["z"].mean()
        pieces.append((-1, mean_curve))
        for rep, sub in pieces:
            sub = sub.sort_values("f")
            sites = pl.detect_collapse_sites(sub["f"].to_numpy(), sub["z"].to_numpy(),
                                             pure=(mode == "pure"))
            sites = pl.merge_adjacent(sites, tol=cfg.site_merge_tol_hz)

            # assign every detected site to the branch that predicts it; "both" is set aside
            by_branch: dict[str, list[float]] = {"stride": [], "patch": []}
            n_both = 0
            n_unassigned = 0
            assignment_residuals: dict[str, list[float]] = {"stride": [], "patch": []}
            for s in sites:
                fam, residual = pl.assign_site_family(
                    s,
                    P,
                    S,
                    tol_hz=cfg.site_assignment_tol_hz,
                    ambiguity_hz=cfg.site_ambiguity_hz,
                )
                if fam == "both":
                    n_both += 1
                elif fam in by_branch:
                    by_branch[fam].append(s)
                    assignment_residuals[fam].append(residual)
                else:
                    n_unassigned += 1

            for branch, members in by_branch.items():
                rows.append(dict(model=model, P=P, S=S, mode=mode, rep=int(rep),
                                 branch=branch,
                                 predicted_spacing=pl.FS / (S if branch == "stride" else P),
                                 sites=" ".join(f"{s:.3f}" for s in members),
                                 n_ambiguous=n_both, n_unassigned=n_unassigned,
                                 max_assignment_error=(max(assignment_residuals[branch])
                                                       if assignment_residuals[branch] else np.nan),
                                 **pl.site_summary(members)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------- #
#  Driver: shard per model, resume, merge, guard
# --------------------------------------------------------------------------------- #
def _shard(out: Path, table: str, tag: str) -> Path:
    return out / "raw" / f"{table}__{tag}.parquet"


def _expected_raw_tables(cfg: Config) -> tuple[str, ...]:
    return RAW_TABLES if cfg.band_tasks else tuple(t for t in RAW_TABLES if t != "mdl_bandtasks")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "pyarrow", "torch", "chronos-forecasting"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _design_payload(cfg: Config, planned_models: list[tuple[int, int]]) -> dict:
    design = asdict(cfg)
    # Batching and device affect throughput, not the observations.  Every inferential knob remains
    # in the fingerprint; changing one requires a fresh run namespace.
    design.pop("batch_size", None)
    design.pop("device", None)
    sources = [
        Path(__file__).resolve(),
        Path(pl.__file__).resolve(),
        Path(ml.__file__).resolve(),
        Path(pl.__file__).resolve().parent.parent / "data" / "synthetic" / "generators"
        / "kernelsynth_generator.py",
    ]
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "deliverable": "coursework/deliverable3",
        "reportable": not cfg.smoke,
        "config": design,
        "planned_models": [pl.model_tag(P, S) for P, S in planned_models],
        "checkpoints": {
            pl.model_tag(P, S): ml.checkpoint_identity(P, S) for P, S in planned_models
        },
        "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[2])): cp.sha256_file(path)
                          for path in sources},
        "package_versions": _package_versions(),
    }


def _manifest_path(out: Path) -> Path:
    return Path(out) / MANIFEST_NAME


def _load_or_create_manifest(
    out: Path, cfg: Config, planned_models: list[tuple[int, int]]
) -> dict:
    payload = _design_payload(cfg, planned_models)
    design_fingerprint = cp.fingerprint(payload)
    path = _manifest_path(out)
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema in {path}")
        if manifest.get("design_fingerprint") != design_fingerprint:
            raise ValueError(
                "collection manifest does not match this design/source/checkpoint state; "
                "use a new RUN_ID/output directory")
        return manifest

    orphaned = list((out / "raw").glob("*.parquet")) if (out / "raw").exists() else []
    if orphaned:
        raise ValueError(
            f"found {len(orphaned)} raw shard(s) without {MANIFEST_NAME}; use a new output "
            "directory or explicitly remove the orphaned run")
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "design_fingerprint": design_fingerprint,
        "design": payload,
        "status": "partial",
        "created_utc": now,
        "updated_utc": now,
        "shards": {},
        "merged": {},
    }
    cp.atomic_json(path, manifest)
    return manifest


_REQUIRED_COLUMNS = {
    "contrasts": {"model", "P", "S", "generator", "f_lock", "role", "f", "is_lock",
                  "f_hat", "h", "h_truth"},
    "mdl_cells": {"model", "P", "S", "stage", "is_locked", "L_bits"},
    "mdl_bandtasks": {"model", "P", "S", "stage", "task", "L_bits"},
    "collapse": {"model", "P", "S", "mode", "rep", "f", "z", "z_norm"},
    "sites": {"model", "P", "S", "mode", "rep", "branch", "n_sites", "f1", "delta_hat"},
}


def _validate_frame(frame: pd.DataFrame, table: str, tag: str | None = None) -> None:
    if table not in _REQUIRED_COLUMNS:
        raise ValueError(f"unknown table {table!r}")
    missing = _REQUIRED_COLUMNS[table] - set(frame.columns)
    if missing:
        raise ValueError(f"{table} missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{table} is empty")
    if tag is not None and set(frame["model"].astype(str)) != {tag}:
        raise ValueError(f"{table} shard for {tag} contains models {sorted(frame['model'].unique())}")
    finite_columns = {
        "contrasts": ("f", "f_hat", "h", "h_truth"),
        "mdl_cells": ("L_bits",),
        "mdl_bandtasks": ("L_bits",),
        "collapse": ("f", "z", "z_norm"),
        "sites": ("n_sites",),
    }[table]
    for column in finite_columns:
        if not np.isfinite(pd.to_numeric(frame[column], errors="coerce")).all():
            raise ValueError(f"{table}.{column} contains non-finite values")


def _validate_shard(path: Path, table: str, tag: str, entry: dict) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"manifest lists missing shard {path}")
    observed_hash = cp.sha256_file(path)
    if observed_hash != entry.get("sha256"):
        raise ValueError(f"hash mismatch for {path}; refuse to resume mixed/corrupt data")
    frame = pd.read_parquet(path)
    _validate_frame(frame, table, tag)
    if len(frame) != int(entry.get("rows", -1)):
        raise ValueError(f"row-count mismatch for {path}")
    return frame


def collect_model(
    P: int,
    S: int,
    out: Path,
    cfg: Config,
    manifest: dict,
    force: bool = False,
    probe_factory=None,
) -> None:
    """Collect one geometry, recording each atomic shard in the run manifest."""
    tag = pl.model_tag(P, S)
    wanted = list(_expected_raw_tables(cfg))
    missing: list[str] = []
    for table in wanted:
        path = _shard(out, table, tag)
        key = f"{table}__{tag}"
        entry = manifest["shards"].get(key)
        if force:
            missing.append(table)
        elif entry is None:
            if path.exists():
                raise ValueError(f"untracked shard {path}; rerun this geometry with force=True")
            missing.append(table)
        else:
            _validate_shard(path, table, tag, entry)
    if not missing:
        print(f"  {tag}: all manifest-validated shards present, skipping model load")
        return

    factory = pl.Probe if probe_factory is None else probe_factory
    print(f"  {tag}: collecting {missing}")
    probe = factory(P, S, device=cfg.device, batch_size=cfg.batch_size)
    print(f"    loaded {probe.label}  (P={probe.P}, S={probe.S}, device={probe.device})")
    expected_identity = manifest["design"]["checkpoints"][tag]["identity_sha256"]
    actual_identity = getattr(probe, "checkpoint_identity", None)
    if actual_identity is None:
        actual_identity = ml.checkpoint_identity(P, S)
    if actual_identity.get("identity_sha256") != expected_identity:
        probe.close()
        raise ValueError(f"checkpoint identity changed while loading {tag}")
    try:
        builders = {
            "contrasts": lambda: collect_contrasts(probe, cfg),
            "mdl_cells": lambda: collect_mdl_cells(probe, cfg),
            "mdl_bandtasks": lambda: collect_band_tasks(probe, cfg),
            "collapse": lambda: collect_collapse(probe, cfg),
        }
        for table in missing:
            frame = builders[table]()
            _validate_frame(frame, table, tag)
            path = _shard(out, table, tag)
            cp.atomic_parquet(path, frame)
            key = f"{table}__{tag}"
            manifest["shards"][key] = {
                "file": str(path.relative_to(out)),
                "sha256": cp.sha256_file(path),
                "rows": len(frame),
                "columns": list(frame.columns),
                "checkpoint_identity_sha256": expected_identity,
            }
            manifest["status"] = "partial"
            manifest.pop("last_error", None)
            manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
            cp.atomic_json(_manifest_path(out), manifest)
            print(f"    {table}: {len(frame):>6d} rows -> {path.name}")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["last_error"] = f"{type(exc).__name__}: {exc}"
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        cp.atomic_json(_manifest_path(out), manifest)
        raise
    finally:
        probe.close()


def check_design(
    tables: dict[str, pd.DataFrame],
    expected_models: list[tuple[int, int]] | None = None,
    cfg: Config | None = None,
) -> dict:
    """Fail closed on incomplete/non-identifiable inputs before any posterior is sampled."""
    cfg = cfg or Config()
    expected_models = expected_models or list(pl.DELIVERABLE3_MODELS)
    expected_tags = {pl.model_tag(P, S) for P, S in expected_models}
    failures: list[str] = []
    summary: dict[str, object] = {"expected_models": sorted(expected_tags)}
    print("\n  design check")

    required_tables = set(_expected_raw_tables(cfg)) | {"sites"}
    missing_tables = required_tables - set(tables)
    if missing_tables:
        failures.append(f"missing tables: {sorted(missing_tables)}")

    for table, frame in tables.items():
        _validate_frame(frame, table)
        observed = set(frame["model"].astype(str))
        if table in required_tables and observed != expected_tags:
            failures.append(
                f"{table} model coverage mismatch: missing={sorted(expected_tags - observed)}, "
                f"extra={sorted(observed - expected_tags)}")

    if "contrasts" in tables:
        contrasts = tables["contrasts"]
        for model, group in contrasts.groupby("model"):
            # Both levels of the lock indicator must be present and neither response level may be
            # empty: a geometry whose arms all hit, or all miss, identifies no gamma at all.
            n_lock = int((group["is_lock"] == 1).sum())
            n_ctrl = int((group["is_lock"] == 0).sum())
            hit_rate = float(group["h"].mean())
            print(f"    contrasts {model:<9s} rows={len(group):>6d} lock={n_lock:>5d} "
                  f"ctrl={n_ctrl:>5d} hit={hit_rate:>5.3f} "
                  f"locks={group['f_lock'].nunique():>3d} phases={group['phase_idx'].nunique():>3d} "
                  f"generators={group['generator'].nunique()}")
            if n_lock == 0 or n_ctrl == 0:
                failures.append(f"{model}: gamma NOT IDENTIFIED (one IsLock level empty)")
            if group["h"].nunique() < 2:
                failures.append(f"{model}: localisation indicator is constant at {hit_rate:.0f}")
            # Survival is a per-geometry property, not a global one. P=S=8 has only three lock
            # sites in band, so a low ceiling can reduce it to about one while the pooled table
            # still looks healthy. A geometry contributing one site contributes a beta_c that is
            # one number, and M1 is a regression over those numbers.
            surviving = int(group.loc[group["is_lock"] == 1, "h_truth"].astype(bool).sum() and
                            group.loc[group["h_truth"].astype(bool) & (group["is_lock"] == 1),
                                      "f_lock"].nunique())
            if surviving < cfg.min_sites_per_geometry:
                failures.append(f"{model}: only {surviving} candidate site(s) survive the "
                                f"instrument ceiling, below the bar of "
                                f"{cfg.min_sites_per_geometry}")
            if set(group["generator"]) != set(cfg.generators):
                failures.append(f"{model}: generator coverage mismatch in contrasts")

    if "mdl_cells" in tables:
        for model, group in tables["mdl_cells"].groupby("model"):
            n1 = int((group["is_locked"] == 1).sum())
            n0 = int((group["is_locked"] == 0).sum())
            print(f"    mdl_cells {model:<9s} locked={n1:>5d} unlocked={n0:>5d}")
            if n1 == 0 or n0 == 0:
                failures.append(f"{model}: theta_lock NOT IDENTIFIED (one IsLocked level empty)")

    if "collapse" in tables:
        for model, group in tables["collapse"].groupby("model"):
            if set(group["mode"]) != set(cfg.collapse_modes):
                failures.append(f"{model}: collapse-mode coverage mismatch")

    if "sites" in tables:
        mean_sites = tables["sites"][tables["sites"].rep == -1]
        for _, row in mean_sites.iterrows():
            print(f"    sites     {row['model']:<9s} mode={row['mode']:<11s} "
                  f"branch={row['branch']:<6s} n={int(row['n_sites']):>3d} "
                  f"f1={row['f1']:.2f} delta_hat={row['delta_hat']:.2f}")
        generated = mean_sites[mean_sites["mode"].isin(cfg.generators)]
        totals = generated.groupby("branch")["n_sites"].sum()
        identification = {
            branch: {
                "unambiguous_sites": int(totals.get(branch, 0)),
                "identified": int(totals.get(branch, 0)) >= cfg.min_d2_sites,
            }
            for branch in ("stride", "patch")
        }
        summary["d2_identification"] = identification

    summary["failures"] = failures
    summary["ok"] = not failures
    print("  design check:", "PASS" if not failures else "FAIL, do not sample on this data")
    if failures:
        raise ValueError("design validation failed: " + "; ".join(failures))
    return summary


def merge(
    out: Path,
    cfg: Config,
    planned_models: list[tuple[int, int]] | None = None,
    allow_partial: bool = False,
    manifest: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """Merge only exact manifest-listed shards; stale glob matches are never included."""
    out = Path(out)
    planned_models = list(pl.DELIVERABLE3_MODELS if planned_models is None else planned_models)
    manifest = manifest or _load_or_create_manifest(out, cfg, planned_models)
    wanted_tables = _expected_raw_tables(cfg)

    complete_models: list[tuple[int, int]] = []
    for P, S in planned_models:
        tag = pl.model_tag(P, S)
        if all(f"{table}__{tag}" in manifest["shards"] for table in wanted_tables):
            complete_models.append((P, S))
    missing_models = [m for m in planned_models if m not in complete_models]
    if missing_models and not allow_partial:
        raise ValueError(
            "collection is incomplete; missing complete shard sets for "
            + ", ".join(pl.model_tag(P, S) for P, S in missing_models))
    if not complete_models:
        raise ValueError("no geometry has a complete, manifest-tracked shard set")

    merged: dict[str, pd.DataFrame] = {}
    for table in wanted_tables:
        frames = []
        for P, S in complete_models:
            tag = pl.model_tag(P, S)
            key = f"{table}__{tag}"
            frames.append(_validate_shard(_shard(out, table, tag), table, tag,
                                          manifest["shards"][key]))
        frame = pd.concat(frames, ignore_index=True)
        merged[table] = frame
        target = out / f"02_{table}.parquet"
        cp.atomic_parquet(target, frame)
        manifest["merged"][table] = {
            "file": target.name,
            "sha256": cp.sha256_file(target),
            "rows": len(frame),
        }

    sites = derive_sites(merged["collapse"], cfg)
    _validate_frame(sites, "sites")
    merged["sites"] = sites
    sites_target = out / "02_sites.parquet"
    cp.atomic_parquet(sites_target, sites)
    manifest["merged"]["sites"] = {
        "file": sites_target.name,
        "sha256": cp.sha256_file(sites_target),
        "rows": len(sites),
    }

    design_summary = check_design(merged, complete_models, cfg)
    manifest["design_check"] = design_summary
    manifest["status"] = "complete" if not missing_models else "partial"
    manifest.pop("last_error", None)
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    cp.atomic_json(_manifest_path(out), manifest)
    return merged


def load_collection(
    out: Path | str,
    cfg: Config | None = None,
    planned_models: list[tuple[int, int]] | None = None,
    require_complete: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load merged tables only after manifest, hashes, coverage and design gates pass."""
    out = Path(out)
    cfg = cfg or Config()
    planned_models = list(pl.DELIVERABLE3_MODELS if planned_models is None else planned_models)
    manifest = _load_or_create_manifest(out, cfg, planned_models)
    if require_complete and manifest.get("status") != "complete":
        raise ValueError(f"run manifest status is {manifest.get('status')!r}, expected 'complete'")
    tables: dict[str, pd.DataFrame] = {}
    for table in (*_expected_raw_tables(cfg), "sites"):
        entry = manifest.get("merged", {}).get(table)
        if entry is None:
            raise ValueError(f"manifest has no merged {table} table")
        path = out / entry["file"]
        if not path.is_file() or cp.sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"merged {table} file is missing or has a hash mismatch")
        frame = pd.read_parquet(path)
        _validate_frame(frame, table)
        tables[table] = frame
    expected = planned_models if require_complete else [
        (int(group.P.iloc[0]), int(group.S.iloc[0]))
        for _, group in tables["contrasts"].groupby("model")
    ]
    check_design(tables, expected, cfg)
    return tables


def collect_all(
    out: Path | str,
    models: list[tuple[int, int]] | None = None,
    cfg: Config | None = None,
    force: bool = False,
    planned_models: list[tuple[int, int]] | None = None,
    allow_partial: bool = False,
    probe_factory=None,
) -> dict[str, pd.DataFrame]:
    """Collect a session subset against one frozen fifteen-model design."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg or Config()
    planned_models = list(pl.DELIVERABLE3_MODELS if planned_models is None else planned_models)
    models = list(planned_models if models is None else models)
    if not set(models).issubset(set(planned_models)):
        raise ValueError("session models must be a subset of the frozen planned model registry")
    if set(models) != set(planned_models) and not allow_partial:
        raise ValueError("a model subset requires allow_partial=True and cannot be reportable yet")
    manifest = _load_or_create_manifest(out, cfg, planned_models)
    print(f"collecting into {out}  |  {'SMOKE (NON-REPORTABLE)' if cfg.smoke else 'FULL'} "
          f"design  |  session {len(models)}/{len(planned_models)} geometries")

    index_path = out / "signals" / "signals_index.parquet"
    signal_entry = manifest.get("signals")
    if index_path.exists() != (signal_entry is not None):
        raise ValueError("untracked or missing signal archive; use a new output directory")
    if signal_entry is not None and cp.sha256_file(index_path) != signal_entry.get("sha256"):
        raise ValueError("signal index hash mismatch; refuse to resume altered inputs")
    meta = pl.save_signal_pool(out, generators=cfg.generators,
                               n_bg=max(cfg.n_bg, cfg.mdl_n_bg, cfg.collapse_reps))
    manifest["signals"] = {
        "file": str(index_path.relative_to(out)),
        "sha256": cp.sha256_file(index_path),
        "rows": len(meta),
    }
    cp.atomic_json(_manifest_path(out), manifest)
    print(f"  archived/validated {len(meta)} signals -> {out / 'signals'}")
    for P, S in models:
        collect_model(P, S, out, cfg, manifest, force=force, probe_factory=probe_factory)
    return merge(out, cfg, planned_models, allow_partial=allow_partial, manifest=manifest)


# --------------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="bayes_data", help="output directory")
    ap.add_argument("--models", nargs="*", default=None,
                    help="session p{P}-s{S} tags (default: all 15 Deliverable 3 models)")
    ap.add_argument("--smoke", action="store_true", help="tiny grids, for a pipeline check")
    ap.add_argument("--force", action="store_true", help="recompute shards that already exist")
    ap.add_argument("--no-band-tasks", action="store_true", help="skip the descriptive band tasks")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--merge-only", action="store_true", help="only re-merge existing shards")
    args = ap.parse_args(argv)

    cfg = Config.smoke_cfg() if args.smoke else Config()
    if args.no_band_tasks:
        cfg.band_tasks = False
    if args.device:
        cfg.device = args.device
    if args.batch_size:
        cfg.batch_size = args.batch_size

    wanted = set(args.models) if args.models else None
    models = [(P, S) for (P, S) in pl.DELIVERABLE3_MODELS
              if wanted is None or pl.model_tag(P, S) in wanted]
    if not models:
        print(f"no geometry matched {args.models}; available: "
              f"{[pl.model_tag(P, S) for P, S in pl.DELIVERABLE3_MODELS]}")
        return 1

    out = Path(args.out)
    if args.merge_only:
        merge(out, cfg, list(pl.DELIVERABLE3_MODELS), allow_partial=args.models is not None)
    else:
        collect_all(out, models, cfg, force=args.force,
                    planned_models=list(pl.DELIVERABLE3_MODELS),
                    allow_partial=args.models is not None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
