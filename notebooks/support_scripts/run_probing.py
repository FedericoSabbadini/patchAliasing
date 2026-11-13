"""Run the two SV probing experiments, resuming validated result shards."""
from __future__ import annotations

import argparse
from pathlib import Path

from probing_lib import MODELS, ProbeConfig, run_complex, run_pure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("pure", "complex", "both"), default="both")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1] / "_run" / "probing")
    parser.add_argument("--models", nargs="+", choices=[s.tag for s in MODELS])
    args = parser.parse_args()
    cfg = ProbeConfig.smoke() if args.mode == "smoke" else ProbeConfig()
    models = tuple(s for s in MODELS if s.tag in args.models) if args.models else (
        tuple(s for s in MODELS if s.tag == "retrained-p16-s16") if args.mode == "smoke" else MODELS)
    for kind, runner in (("pure", run_pure), ("complex", run_complex)):
        if args.kind in (kind, "both"):
            _results, out = runner(args.root, cfg, models, args.device)
            print(f"Completed {kind}: {out}", flush=True)


if __name__ == "__main__":
    main()
