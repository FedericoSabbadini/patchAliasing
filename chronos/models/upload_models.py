"""
upload_models.py — push every FINISHED model (has a DONE marker) to the HuggingFace repo.

Only directories with a DONE marker are uploaded, so an unfinished run (e.g. a model that
stopped mid-training) is automatically skipped. Re-runs overwrite the same subfolders.

    python chronos/models/upload_models.py
"""
from pathlib import Path

from huggingface_hub import HfApi

REPO = "federicosabbadini/chronos-bolt-patch-sweep"
ROOT = Path(__file__).resolve().parent.parent / "outputs" / "models"
FILES = ["config.json", "model.safetensors", "run_config.json",
         "loss_history.npy", "val_history.npy", "loss_curve.png", "DONE"]

api = HfApi()
api.create_repo(REPO, exist_ok=True)

done = sorted(ROOT.glob("*/DONE"))
print(f"{len(done)} finished model(s) -> {REPO}")
for marker in done:
    name = marker.parent.name
    print(f"uploading {name} ...")
    for f in FILES:
        p = marker.parent / f
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{name}/{f}", repo_id=REPO)

manifest = ROOT / "manifest.csv"
if manifest.exists():
    api.upload_file(path_or_fileobj=str(manifest), path_in_repo="manifest.csv", repo_id=REPO)

print(f"done -> https://huggingface.co/{REPO}")
