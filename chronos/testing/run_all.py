"""
run_all.py — single reproducible entry point for the whole testing/ workflow.

Reproduces, in one pass and against one code version, EXACTLY the set of outputs under
`outputs/` (probing + H1/H2/H3, pure + TSMixup + KernelSynth, per model and cross):

  outputs/probing/p{P}-s{S}/        FIG3 / FIG4 / FIG5 / FIG5b   (pure sinusoid, all 5 models)
  outputs/hypotheses/p{P}-s{S}/      H1 / H2 / H3                 (pure sinusoid)
  outputs/hypotheses/p{P}-s{S}_tsm/  H1 / H2 / H3                 (TSMixup background)
  outputs/hypotheses/p{P}-s{S}_ks/   H1 / H2 / H3                 (KernelSynth background)
  outputs/hypotheses/*_all_models*   H1 / H3 cross figures + collapse CSVs (pure / tsm / ks)

Usage:
    python run_all.py                 # full run: 5 models x {probing, pure, tsm, ks} + cross
    python run_all.py --smoke         # fast coarse pass (PROBE_SMOKE=1 for the notebook)
    python run_all.py 16-16 8-8       # restrict to these models (still all signal modes)
    python run_all.py --no-probing    # skip the probing notebook (hypotheses only)
    python run_all.py --modes pure    # only the pure-sinusoid hypotheses (comma list: pure,tsm,ks)

Everything is logged (tee) to run_all_<timestamp>.log alongside this file, so the run has a
single, complete provenance record. Model selection is centralised in testing_lib.load_pipeline.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import testing_lib as tl

NOTEBOOK = "chronosBolt_layer_probing.ipynb"
MODE_FLAG = {"pure": None, "tsm": "--background-tsm", "ks": "--background-ks"}


class _Tee:
    """Mirror stdout to the console and a log file, so the run leaves one complete record."""

    def __init__(self, path: Path):
        self._f = open(path, "w", encoding="utf-8")
        self._out = sys.__stdout__

    def write(self, s):
        self._out.write(s); self._out.flush()
        self._f.write(s); self._f.flush()

    def flush(self):
        self._out.flush(); self._f.flush()

    def close(self):
        self._f.close()


def _run(cmd: list[str], env: dict | None = None) -> bool:
    """Run a subprocess, streaming its output through the tee. True on success."""
    print(f"    $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=HERE, env=env,
                       stdout=sys.stdout, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"    !! FAILED (exit {r.returncode}): {' '.join(cmd)}")
    return r.returncode == 0


def run_probing(P: int, S: int, smoke: bool) -> bool:
    """Execute the probing notebook for geometry (P, S) in pure-sinusoid mode (the default)."""
    env = os.environ.copy()
    env["PROBE_PATCH"] = str(P)
    env["PROBE_STRIDE"] = str(S)
    env["PROBE_PURE"] = "1"                 # pure sinusoid: the Pagani-parity reproduction
    if smoke:
        env["PROBE_SMOKE"] = "1"
    tmp = HERE / "outputs" / "_gen_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
           "--ExecutePreprocessor.timeout=-1",
           "--output-dir", str(tmp), "--output", NOTEBOOK, NOTEBOOK]
    ok = _run(cmd, env)
    (tmp / NOTEBOOK).unlink(missing_ok=True)               # the notebook writes figures itself
    return ok


def run_hypotheses(P: int, S: int, mode: str, cross: bool = False) -> bool:
    """Run hypotheses.py for one model (or --cross) in signal `mode` (pure/tsm/ks)."""
    cmd = [sys.executable, "hypotheses.py"]
    cmd += ["--cross"] if cross else ["--P", str(P), "--S", str(S)]
    if MODE_FLAG[mode]:
        cmd.append(MODE_FLAG[mode])
    return _run(cmd)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="P-S tags to run (default: all in ALL_MODELS)")
    ap.add_argument("--smoke", action="store_true", help="fast coarse pass for the notebook")
    ap.add_argument("--no-probing", action="store_true", help="skip the probing notebook")
    ap.add_argument("--no-hypotheses", action="store_true", help="skip the H1/H2/H3 runs")
    ap.add_argument("--modes", default="pure,tsm,ks",
                    help="comma list of signal modes for hypotheses (subset of pure,tsm,ks)")
    args = ap.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODE_FLAG]
    if bad:
        ap.error(f"unknown mode(s) {bad}; choose from {list(MODE_FLAG)}")

    wanted = set(args.models) or None
    models = [(P, S) for (P, S) in tl.ALL_MODELS if wanted is None or f"{P}-{S}" in wanted]

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tee = _Tee(HERE / f"run_all_{stamp}.log")
    sys.stdout = tee
    try:
        print(f"run_all.py  {stamp}")
        print(f"models   : {[tl.model_tag(P, S) for P, S in models]}")
        print(f"probing  : {'skip' if args.no_probing else 'pure sinusoid'}"
              + ("   [SMOKE]" if args.smoke else ""))
        print(f"hypotheses modes: {'skip' if args.no_hypotheses else modes}")

        failures = []

        if not args.no_probing:
            print("\n" + "=" * 60 + "\nPROBING NOTEBOOK (pure sinusoid, all models)\n" + "=" * 60)
            for (P, S) in models:
                print(f"\n--- probing {tl.model_tag(P, S)} ---")
                if not run_probing(P, S, args.smoke):
                    failures.append((tl.model_tag(P, S), "probing"))

        if not args.no_hypotheses:
            for mode in modes:
                print("\n" + "=" * 60 + f"\nHYPOTHESES  [{mode}]  (all models + cross)\n" + "=" * 60)
                for (P, S) in models:
                    print(f"\n--- hypotheses {tl.model_tag(P, S)} [{mode}] ---")
                    if not run_hypotheses(P, S, mode):
                        failures.append((tl.model_tag(P, S), f"hypotheses:{mode}"))
                if len(models) > 1:
                    print(f"\n--- cross [{mode}] ---")
                    if not run_hypotheses(0, 0, mode, cross=True):
                        failures.append(("cross", f"hypotheses:{mode}"))

        print("\n" + ("ALL RUNS SUCCEEDED." if not failures
                      else f"{len(failures)} run(s) FAILED: {failures}"))
        return 1 if failures else 0
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
