"""
run_all_models.py — execute the two testing notebooks for EVERY (P, S) model, unattended.

It runs the notebooks THEMSELVES (via `jupyter nbconvert --execute`), once per model, passing
the model geometry through the `PROBE_PATCH` / `PROBE_STRIDE` environment variables that the
notebooks read in their model-selection cell. So the figures you get are exactly the notebooks'
own figures — nothing here re-plots anything, and no notebook has to be edited by hand.

    cd chronos/testing
    python run_all_models.py                     # all models, full run
    python run_all_models.py --smoke             # fast pass (coarse grid; sets PROBE_SMOKE=1)
    python run_all_models.py 16-16 8-8           # only these models, by p{P}-s{S} tag
    python run_all_models.py --only contamination.ipynb   # run just one notebook

Each notebook writes its figures where it always does:
    chronosBolt_layer_probing.ipynb -> outputs/per_model/p{P}-s{S}/       (FIG1/3/4/5/5b, …)
    contamination.ipynb             -> outputs/contamination/p{P}-s{S}/    (recovery_at_lock, …)
The fully executed copy of each notebook is also saved under outputs/executed/p{P}-s{S}/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import testing_lib as tl

NOTEBOOKS = ["chronosBolt_layer_probing.ipynb", "contamination.ipynb"]


def run_notebook(nb: str, P: int, S: int, smoke: bool) -> bool:
    """Execute one notebook for geometry (P, S). Returns True on success."""
    env = os.environ.copy()
    env["PROBE_PATCH"] = str(P)
    env["PROBE_STRIDE"] = str(S)
    if smoke:
        env["PROBE_SMOKE"] = "1"
    out_dir = HERE / "outputs" / "executed" / tl.model_tag(P, S)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
           "--ExecutePreprocessor.timeout=-1",
           "--output-dir", str(out_dir), "--output", nb, nb]
    print(f"    -> {nb}")
    r = subprocess.run(cmd, cwd=HERE, env=env)          # CWD = chronos/testing (relative paths resolve)
    if r.returncode != 0:
        print(f"    !! FAILED: {nb} for {tl.model_tag(P, S)} (exit {r.returncode})")
    return r.returncode == 0


def main(argv: list[str]):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="*", help="p{P}-s{S} tags to run (default: all)")
    ap.add_argument("--smoke", action="store_true", help="fast coarse-grid pass (PROBE_SMOKE=1)")
    ap.add_argument("--only", action="append", choices=NOTEBOOKS,
                    help="run only this notebook (repeatable)")
    args = ap.parse_args(argv)

    wanted = set(args.models) or None
    models = [(P, S) for (P, S) in tl.ALL_MODELS if wanted is None or f"{P}-{S}" in wanted]
    notebooks = args.only or NOTEBOOKS

    print(f"models   : {[tl.model_tag(P, S) for P, S in models]}")
    print(f"notebooks: {notebooks}"
          + ("   [SMOKE]" if args.smoke else ""))

    failures = []
    for (P, S) in models:
        print(f"\n=== {tl.model_tag(P, S)} ===")
        for nb in notebooks:
            # contamination is a FORECAST-based test: it is only meaningful on the fully-trained
            # official model. The retrained variants forecast too poorly for it to be informative,
            # so it is run for (16, 16) only. (Probing/MDL, being decodability, runs for all.)
            if nb == "contamination.ipynb" and (P, S) != (16, 16):
                print(f"    -- skip {nb} (only meaningful on the official p16-s16)")
                continue
            if not run_notebook(nb, P, S, args.smoke):
                failures.append((tl.model_tag(P, S), nb))

    print("\n" + ("all runs succeeded." if not failures
                  else f"{len(failures)} run(s) failed: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
