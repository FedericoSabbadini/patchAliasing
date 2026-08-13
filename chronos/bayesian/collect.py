"""
collect.py, Part 2 of the Bayesian workflow: let Chronos make the observations.

This is the *only* place where a model is run. It turns the five Chronos-Bolt geometries into five
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

Everything is sharded per geometry under `<out>/raw/`, so an interrupted run resumes at model
granularity: a shard that already exists is not recomputed and its model is never loaded.

    python -m collect --out ./bayes_data                 # all five geometries, full design
    python -m collect --out ./bayes_data --smoke         # tiny grid, minutes not tens of minutes
    python -m collect --out ./bayes_data --models p16-s16 p8-s8
    python -m collect --out ./bayes_data --force         # ignore existing shards
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as pl

TABLES = ("contrasts", "mdl_cells", "mdl_bandtasks", "collapse", "sites")


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
    n_bg: int = 100                 # background draws per generator. Deliverable 1 specifies 100
                                    # TSMixup and 100 KernelSynth signals; these are those signals,
                                    # and they are also the u_background levels of Eq. (9).
    generators: tuple[str, ...] = pl.GENERATORS

    #, frequency-local MDL cells (H1 representational),
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
    """The paired local contrast d of deliverable Eq. (8), with the phase index RETAINED.

    Design (deliverable Local Contrast Analysis): each lock frequency f_k in F_lock is paired with
    two controls at f_k +/- 0.25 fs/S. The three signals of a triplet share the SAME background
    realisation and the SAME phase, that is what makes the contrast paired, and it is the one
    place where this collection deliberately departs from `hypotheses.py`, which averages over
    phases drawn separately per frequency.

    Keeping the phase index is what makes H2 testable at all: Model C bins the phase circle and
    gives each bin its own offset, which is impossible once phases have been averaged away.
    """
    P, S = probe.P, probe.S
    delta = pl.control_offset(S)

    # Only sites whose controls are themselves non-locked: at P=16,S=12 the patch null 32 Hz has
    # its upper control sitting exactly on the first stride lock, so it is dropped rather than
    # compared against another lock.
    sites = [f for f in pl.f_lock(P, S) if pl.controls_are_clean(f, delta, P, S)]

    contexts, futures, freqs, meta = [], [], [], []
    for gen in cfg.generators:
        pool = pl.background_pool(gen, cfg.n_bg, pl.CTX + pl.PRED)
        for fk in sites:
            phases = pl.phases_Sf(fk, cfg.n_phase_contrast)
            for bg_id, bg in enumerate(pool):
                for ph_idx, ph in enumerate(phases):
                    for role, f in (("lock", fk), ("lo", fk - delta), ("hi", fk + delta)):
                        # context and its TRUE continuation are slices of one long realisation,
                        # so the forecast target is the real future rather than a fresh draw
                        full = pl.build_context(bg, f, ph, pl.CTX + pl.PRED)
                        contexts.append(full[:pl.CTX])
                        futures.append(full[pl.CTX:])
                        freqs.append(f)
                        meta.append(dict(generator=gen, bg_id=bg_id, f_lock=fk,
                                         phase_idx=ph_idx, phase=float(ph), role=role))

    if not contexts:
        return pd.DataFrame()

    R, dphase = probe.recovery(np.stack(contexts), np.stack(futures), np.array(freqs))

    # fold the three roles of each triplet back into one row
    long = pd.DataFrame(meta)
    long["R"] = R
    long["dphase"] = dphase
    key = ["generator", "bg_id", "f_lock", "phase_idx"]
    wide = long.pivot_table(index=key + ["phase"], columns="role",
                            values="R", aggfunc="first").reset_index()
    ph_lock = (long[long.role == "lock"].set_index(key)["dphase"])
    wide = wide.join(ph_lock.rename("dphase_lock"), on=key)

    eps = 0.01                                   # the deliverable's stabiliser, so log(0) cannot occur
    log_lock = np.log(wide["lock"] + eps)
    log_ctrl = 0.5 * (np.log(wide["lo"] + eps) + np.log(wide["hi"] + eps))
    wide["d"] = log_lock - log_ctrl              # Eq. (8): d < 0 means localized attenuation
    wide["y_deficit"] = -wide["d"]               # Eq. (11): the same measurement, sign-flipped
    wide["R_ctrl"] = 0.5 * (wide["lo"] + wide["hi"])
    wide = wide.rename(columns={"lock": "R_lock", "lo": "R_lo", "hi": "R_hi"})

    wide["model"] = probe.tag
    wide["P"], wide["S"] = P, S
    wide["delta"] = delta
    wide["overlap"] = (P - S) / P                # O_c of Eq. (9)
    wide["cpp"] = wide["f_lock"] * P / pl.FS     # cycles per patch
    wide["family"] = [pl.lock_family(f, P, S) for f in wide["f_lock"]]
    # "live" = there is recoverable signal on at least one side; where nothing is recovered on
    # either side the contrast is a ratio of two noise floors and carries no information about H1.
    wide["live"] = wide[["R_lock", "R_lo", "R_hi"]].max(axis=1) > 0.05
    return wide[["model", "P", "S", "overlap", "generator", "bg_id", "f_lock", "family", "cpp",
                 "delta", "phase_idx", "phase", "R_lock", "R_lo", "R_hi", "R_ctrl",
                 "dphase_lock", "d", "y_deficit", "live"]]


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
    delta = pl.control_offset(S)
    sites = [f for f in pl.f_lock(P, S) if pl.controls_are_clean(f, delta, P, S)]
    # exactly the frequencies the contrast design visits: every lock and both of its controls,
    # which makes the locked / non-locked split balanced 1:2 by construction
    centers = sorted({round(f, 6) for fk in sites for f in (fk, fk - delta, fk + delta)})
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
    grid = pl.union_grid(pl.MODELS, cfg.collapse_step)
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


def derive_sites(collapse: pd.DataFrame) -> pd.DataFrame:
    """Detected collapse sites and the two summaries the H3 movement model regresses on (Eq. 14).

    Sites are detected per replicate (giving Eq. 14 its residual variance) and once more on the
    replicate-averaged curve, recorded as rep = -1.
    """
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
            sites = pl.merge_adjacent(sites)
            summary = pl.site_summary(sites)
            rows.append(dict(model=model, P=P, S=S, mode=mode, rep=int(rep),
                             sites=" ".join(f"{s:.3f}" for s in sites),
                             predicted_stride=" ".join(f"{s:.3f}" for s in pl.stride_locks(S)),
                             predicted_patch=" ".join(f"{s:.3f}" for s in pl.patch_nulls(P)),
                             **summary))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------- #
#  Driver: shard per model, resume, merge, guard
# --------------------------------------------------------------------------------- #
def _shard(out: Path, table: str, tag: str) -> Path:
    return out / "raw" / f"{table}__{tag}.parquet"


def collect_model(P: int, S: int, out: Path, cfg: Config, force: bool = False) -> None:
    """Collect every table for one geometry, skipping shards that already exist."""
    tag = pl.model_tag(P, S)
    wanted = [t for t in TABLES if t != "sites"]
    missing = [t for t in wanted if force or not _shard(out, t, tag).exists()]
    if not cfg.band_tasks and "mdl_bandtasks" in missing:
        missing.remove("mdl_bandtasks")
    if not missing:
        print(f"  {tag}: all shards present, skipping (model not loaded)")
        return

    print(f"  {tag}: collecting {missing}")
    probe = pl.Probe(P, S, device=cfg.device, batch_size=cfg.batch_size)
    print(f"    loaded {probe.label}  (P={probe.P}, S={probe.S}, device={probe.device})")
    try:
        builders = {
            "contrasts": lambda: collect_contrasts(probe, cfg),
            "mdl_cells": lambda: collect_mdl_cells(probe, cfg),
            "mdl_bandtasks": lambda: collect_band_tasks(probe, cfg),
            "collapse": lambda: collect_collapse(probe, cfg),
        }
        for table in missing:
            df = builders[table]()
            path = _shard(out, table, tag)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            print(f"    {table}: {len(df):>6d} rows -> {path.name}")
    finally:
        probe.close()


def merge(out: Path, cfg: Config) -> dict[str, pd.DataFrame]:
    """Concatenate the per-model shards, derive `sites`, and check the design is identifiable."""
    merged: dict[str, pd.DataFrame] = {}
    for table in TABLES:
        if table == "sites":
            continue
        shards = sorted((out / "raw").glob(f"{table}__*.parquet"))
        if not shards:
            continue
        df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
        merged[table] = df
        df.to_parquet(out / f"02_{table}.parquet", index=False)

    if "collapse" in merged:
        sites = derive_sites(merged["collapse"])
        merged["sites"] = sites
        sites.to_parquet(out / "02_sites.parquet", index=False)

    check_design(merged)
    return merged


def check_design(tables: dict[str, pd.DataFrame]) -> None:
    """Refuse to hand the notebook a design in which an effect cannot be estimated.

    An earlier attempt in this project fitted a lock-vs-control effect on 80 control windows and
    ZERO locked windows: the likelihood carried no information and the posterior simply reprinted
    the prior. The cheapest guard against repeating that is to assert, before any sampling, that
    every contrast has both of its controls and that both IsLocked levels are populated per model.
    """
    print("\n  design check")
    ok = True
    if "contrasts" in tables:
        c = tables["contrasts"]
        for model, g in c.groupby("model"):
            n_live = int(g["live"].sum())
            print(f"    contrasts {model:<9s} rows={len(g):>5d} live={n_live:>5d} "
                  f"locks={g['f_lock'].nunique():>3d} phases={g['phase_idx'].nunique():>3d} "
                  f"generators={g['generator'].nunique()}")
            if g[["R_lock", "R_lo", "R_hi"]].isna().any().any():
                print(f"      !! {model}: incomplete triplets (NaN recovery)")
                ok = False
    if "mdl_cells" in tables:
        m = tables["mdl_cells"]
        for model, g in m.groupby("model"):
            n1 = int((g["is_locked"] == 1).sum())
            n0 = int((g["is_locked"] == 0).sum())
            print(f"    mdl_cells {model:<9s} locked={n1:>5d} unlocked={n0:>5d}")
            if n1 == 0 or n0 == 0:
                print(f"      !! {model}: theta_lock is NOT identified (one IsLocked level empty)")
                ok = False
    if "sites" in tables:
        s = tables["sites"][tables["sites"].rep == -1]
        for _, r in s.iterrows():
            print(f"    sites     {r['model']:<9s} mode={r['mode']:<11s} n={int(r['n_sites']):>3d} "
                  f"f1={r['f1']:.2f} delta_hat={r['delta_hat']:.2f}  (fs/S={pl.FS / r['S']:.2f})")
    print("  design check:", "PASS" if ok else "FAIL, do not sample on this data")


def collect_all(out: Path | str, models: list[tuple[int, int]] | None = None,
                cfg: Config | None = None, force: bool = False) -> dict[str, pd.DataFrame]:
    """Collect every table for every geometry and return the merged tables."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg or Config()
    models = models or pl.MODELS
    print(f"collecting into {out}  |  {'SMOKE' if cfg.smoke else 'FULL'} design  |  "
          f"{len(models)} geometries")

    # Archive the generated signals themselves. Every number downstream is computed from these
    # backgrounds, so they are written to disk rather than left in an in-memory cache; a reader
    # who has only the tables cannot regenerate the inputs, and KernelSynth in particular is a
    # random GP draw that is only reproducible given its seed.
    meta = pl.save_signal_pool(out, generators=cfg.generators,
                               n_bg=max(cfg.n_bg, cfg.mdl_n_bg, cfg.collapse_reps))
    print(f"  archived {len(meta)} signals -> {out / 'signals'}")
    for (P, S) in models:
        collect_model(P, S, out, cfg, force=force)
    return merge(out, cfg)


# --------------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="bayes_data", help="output directory")
    ap.add_argument("--models", nargs="*", default=None, help="p{P}-s{S} tags (default: all five)")
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
    models = [(P, S) for (P, S) in pl.MODELS if wanted is None or pl.model_tag(P, S) in wanted]
    if not models:
        print(f"no geometry matched {args.models}; available: "
              f"{[pl.model_tag(P, S) for P, S in pl.MODELS]}")
        return 1

    out = Path(args.out)
    if args.merge_only:
        merge(out, cfg)
    else:
        collect_all(out, models, cfg, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
