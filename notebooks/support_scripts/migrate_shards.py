"""Riusa le shard di una raccolta precedente in una nuova, DOPO averlo dimostrato.

Il problema
-----------
L'impronta del manifest include `planned_models` e gli hash sorgente di collect.py e
probe_lib.py. Passando da 15 a 22 geometrie cambiano entrambi, e collect.py rifiuta di
riprendere la vecchia cartella. Il rifiuto e' corretto: serve a impedire che dati raccolti
con codice diverso finiscano nello stesso risultato sulla base di un ragionamento.

Cosa fa questo script
---------------------
Non aggira il controllo e non riscrive l'impronta. Fa una cosa diversa: **verifica
empiricamente** che il cambio di codice non abbia toccato le misure, e solo se la verifica
passa importa le shard, registrando apertamente da dove vengono.

  1. Nella cartella NUOVA viene raccolta UNA geometria di controllo con il codice nuovo.
  2. Le sue tabelle vengono confrontate, valore per valore, con quelle della stessa
     geometria nella cartella VECCHIA.
  3. Se coincidono esattamente, le altre geometrie comuni vengono importate: i file
     copiati, gli hash ricalcolati sui file copiati, le voci scritte nel manifest nuovo.
  4. Il manifest nuovo riceve un blocco `imported_from` con l'impronta vecchia, gli hash
     sorgente vecchi, la geometria di controllo e l'esito del confronto. La provenienza
     resta scritta nel dato, non in una nostra affermazione.
  5. Se il confronto fallisce, lo script si ferma e stampa dove differiscono. In quel caso
     il cambio di codice NON e' inerte e va raccolto tutto da capo.

Le geometrie non presenti nella vecchia raccolta restano da raccogliere normalmente.

Uso
---
    python migrate_shards.py --old  /percorso/rigenerata/data \\
                             --new  /percorso/rigenerata_all22/data \\
                             --control p16-s8
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MANIFEST = "collection_manifest.json"
TABLES = ("contrasts", "mdl_cells", "mdl_bandtasks", "collapse")


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shard(root: Path, table: str, tag: str) -> Path:
    return root / "raw" / f"{table}__{tag}.parquet"


def tags_of(man: dict) -> set[str]:
    return {k.split("__", 1)[1] for k in man.get("shards", {})}


def complete_tags(root: Path, man: dict) -> set[str]:
    out = set()
    for tag in tags_of(man):
        keys = [f"{t}__{tag}" for t in TABLES if f"{t}__{tag}" in man["shards"]]
        if keys and all(shard(root, k.split("__")[0], tag).is_file() for k in keys):
            out.add(tag)
    return out


def confronta(a: Path, b: Path) -> tuple[bool, str]:
    """Confronto valore per valore, indipendente da come il parquet e' stato scritto."""
    da, db = pd.read_parquet(a), pd.read_parquet(b)
    if list(da.columns) != list(db.columns):
        return False, f"colonne diverse:\n  {list(da.columns)}\n  {list(db.columns)}"
    if len(da) != len(db):
        return False, f"righe diverse: {len(da)} vs {len(db)}"
    ordine = [c for c in da.columns if c in ("model", "generator", "mode", "rep", "bg_id",
                                             "f_lock", "phase_idx", "role", "f", "stage")]
    if ordine:
        da = da.sort_values(ordine).reset_index(drop=True)
        db = db.sort_values(ordine).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(da, db, check_exact=True, check_dtype=False)
    except AssertionError as exc:
        num = [c for c in da.columns if pd.api.types.is_numeric_dtype(da[c])]
        peggio = ""
        if num:
            diff = (da[num].astype(float) - db[num].astype(float)).abs().max()
            peggio = "\n  massimo scarto per colonna:\n" + diff[diff > 0].to_string()
        return False, str(exc).split("\n")[0] + peggio
    return True, "identiche"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="raccolta precedente, gia' completa")
    ap.add_argument("--new", required=True, help="raccolta nuova (popolazione allargata)")
    ap.add_argument("--control", default=None,
                    help="geometria da riraccogliere e confrontare (default: la prima comune)")
    ap.add_argument("--population", default="all22",
                    choices=["deliverable3", "extended21", "all22"])
    ap.add_argument("--pipeline", default=".", help="cartella con collect.py")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--entry", default="collect.py",
                    help="script di raccolta (es. _collect_lowmem.py)")
    args = ap.parse_args(argv)

    old, new, pipe = Path(args.old), Path(args.new), Path(args.pipeline)
    man_old_p = old / MANIFEST
    if not man_old_p.is_file():
        print(f"ERRORE: {man_old_p} non esiste"); return 1
    man_old = json.loads(man_old_p.read_text())
    disponibili = complete_tags(old, man_old)
    if not disponibili:
        print("ERRORE: la raccolta precedente non ha geometrie complete"); return 1
    control = args.control or sorted(disponibili)[0]
    if control not in disponibili:
        print(f"ERRORE: {control} non e' completa nella raccolta precedente "
              f"({sorted(disponibili)})"); return 1

    print("=" * 74)
    print(f"vecchia: {old}   geometrie complete: {len(disponibili)}")
    print(f"nuova:   {new}   popolazione: {args.population}")
    print(f"controllo: {control}")
    print("=" * 74)

    # 1. raccolta della geometria di controllo con il CODICE NUOVO
    print(f"\n[1/4] raccolgo {control} con il codice nuovo (serve solo a confrontare)")
    rc = subprocess.call([sys.executable, args.entry, "--out", str(new),
                          "--device", args.device, "--batch-size", str(args.batch_size),
                          "--models", control, "--population", args.population], cwd=str(pipe))
    if rc != 0:
        print(f"ERRORE: la raccolta di controllo e' uscita con codice {rc}"); return 1

    # 2. confronto valore per valore
    print(f"\n[2/4] confronto {control}: codice nuovo contro raccolta precedente")
    man_new = json.loads((new / MANIFEST).read_text())
    esiti = {}
    for t in TABLES:
        pa, pb = shard(new, t, control), shard(old, t, control)
        if not pa.is_file() and not pb.is_file():
            continue
        if pa.is_file() != pb.is_file():
            print(f"  {t:14s} presente solo in una delle due: non confrontabile"); return 1
        ok, msg = confronta(pa, pb)
        esiti[t] = ok
        print(f"  {t:14s} {'IDENTICHE' if ok else 'DIVERSE'}   {'' if ok else msg}")
    if not esiti or not all(esiti.values()):
        print("\nIl cambio di codice NON e' inerte sulle misure: nessuna importazione.")
        print("Va raccolta l'intera popolazione da zero.")
        return 2
    print("\nVerifica superata: su questa geometria il codice nuovo produce esattamente")
    print("le stesse tabelle di quello vecchio.")

    # 3. importazione delle altre geometrie comuni
    da_importare = sorted(disponibili - {control})
    print(f"\n[3/4] importo {len(da_importare)} geometrie: {', '.join(da_importare)}")
    (new / "raw").mkdir(parents=True, exist_ok=True)
    importate = []
    for tag in da_importare:
        for t in TABLES:
            key = f"{t}__{tag}"
            if key not in man_old["shards"]:
                continue
            src, dst = shard(old, t, tag), shard(new, t, tag)
            if not src.is_file():
                print(f"  ERRORE: manca {src}"); return 1
            shutil.copy2(src, dst)
            voce = dict(man_old["shards"][key])
            voce["sha256"] = sha256(dst)          # ricalcolato sul file copiato
            voce["file"] = str(dst.relative_to(new))
            voce["imported"] = True
            man_new["shards"][key] = voce
        importate.append(tag)
        print(f"  {tag}")

    # 4. provenienza scritta nel manifest, non affermata a voce
    man_new["imported_from"] = {
        "path": str(old),
        "design_fingerprint": man_old.get("design_fingerprint"),
        "source_sha256": man_old.get("design", {}).get("source_sha256"),
        "planned_models": man_old.get("design", {}).get("planned_models"),
        "control_geometry": control,
        "control_tables_identical": sorted(esiti),
        "imported_tags": importate,
        "imported_utc": datetime.now(timezone.utc).isoformat(),
        "note": ("shard riusate da una raccolta con impronta diversa; l'equivalenza delle "
                 "misure e' stata verificata riraccogliendo la geometria di controllo con "
                 "il codice nuovo e confrontando ogni tabella valore per valore"),
    }
    man_new["status"] = "partial"
    man_new["updated_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = (new / MANIFEST).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man_new, indent=2))
    tmp.replace(new / MANIFEST)

    print(f"\n[4/4] fatto. Nel manifest nuovo: {len(complete_tags(new, man_new))} geometrie "
          f"complete su {len(man_new.get('design', {}).get('planned_models', []))} pianificate.")
    print("Restano da raccogliere normalmente quelle non presenti nella vecchia raccolta.")
    print("La provenienza e' nel manifest, campo 'imported_from'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
