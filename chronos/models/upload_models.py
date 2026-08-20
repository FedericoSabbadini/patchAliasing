"""
upload_models.py — push every FINISHED checkpoint to the HuggingFace sweep repository.

Each geometry's training run writes a ``DONE`` marker file when it completes successfully.
This script scans ``chronos/models/weights/`` for those markers and uploads the associated
artefacts (weights, config, loss curves) to the shared HF repository so that downstream
code (model_loader.py, probe_lib.py) can fetch them without local copies.

Only directories that contain a ``DONE`` file are uploaded; interrupted or in-progress
runs are silently skipped.  Re-running the script is safe: it overwrites the same
sub-folders on the hub, so the latest local version always wins.

Usage::

    python chronos/models/upload_models.py
"""
from pathlib import Path

from huggingface_hub import HfApi

# The HuggingFace repository that hosts the full patch-stride sweep.
# model_loader.py and probe_lib.py fall back to this repo when local weights are absent.
REPO = "federicosabbadini/chronos-bolt-patch-sweep"

# Local directory where train_sweep.py writes finished checkpoints.
ROOT = Path(__file__).resolve().parent / "weights"

# Per-geometry artefacts to upload.  Every file is optional (upload skips missing ones)
# so that partially saved runs still get their available outputs on the hub.
FILES = ["config.json", "model.safetensors", "run_config.json",
         "loss_history.npy", "val_history.npy", "loss_curve.png", "DONE"]

# Authenticate via the locally cached HF token (huggingface-cli login).
api = HfApi()

# Create the repo if it does not exist yet; no-op if it already does.
api.create_repo(REPO, exist_ok=True)

# Find every geometry that finished training (has a DONE marker).
done = sorted(ROOT.glob("*/DONE"))
print(f"{len(done)} finished model(s) -> {REPO}")

for marker in done:
    # The parent directory name is the geometry tag, e.g. "p16-s12-seed42".
    name = marker.parent.name
    print(f"uploading {name} ...")
    for f in FILES:
        p = marker.parent / f
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{name}/{f}", repo_id=REPO)

# The manifest lists every geometry with its patch/stride/seed metadata.
# It sits at the repo root so users can enumerate available checkpoints without
# downloading any of them.
manifest = ROOT / "manifest.csv"
if manifest.exists():
    api.upload_file(path_or_fileobj=str(manifest), path_in_repo="manifest.csv", repo_id=REPO)

print(f"done -> https://huggingface.co/{REPO}")
