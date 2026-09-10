#!/usr/bin/env python3
"""Tidy the deliverable: one figures/ folder, and everything unreferenced set aside.

Run from the deliverable3 directory.

    python3 cleanup_unused.py           # dry run, reports what it would do
    python3 cleanup_unused.py --apply   # does it

Two things happen, in this order.

1.  Every file in figures/, solar/, probing/, tables/ and sections/ that nothing
    reachable from main.tex references is moved into _to_delete/<subfolder>/.
    Nothing is deleted: rebuild, check, then remove that folder.

2.  Every referenced image still living outside figures/ is moved into figures/,
    and the \\includegraphics paths in the source are rewritten to match, so the
    document keeps building. solar/ and probing/ are removed once empty.

The keep-set is derived from the sources at run time rather than hard-coded: the
script follows \\input from main.tex to find the section files that are actually
part of the document, then collects every \\includegraphics path and every
\\input{tables/...} inside those files. A file nothing reachable references is
unreferenced by construction, so this stays correct if the document changes.

Step 2 refuses to overwrite: if a name would collide inside figures/ the script
stops and reports it instead of guessing which copy is wanted.
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "figures"
SCAN_DIRS = ["figures", "solar", "probing", "tables", "sections"]
IMAGE_DIRS = ["solar", "probing"]          # consolidated into figures/
NEVER_MOVE = {"main.tex"}
APPLY = "--apply" in sys.argv


def read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def strip_comments(text: str) -> str:
    """Drop LaTeX comments so a commented-out \\input does not count as a use."""
    out = []
    for line in text.splitlines():
        i, esc = 0, False
        while i < len(line):
            if line[i] == "\\":
                esc = not esc
            elif line[i] == "%" and not esc:
                line = line[:i]
                break
            else:
                esc = False
            i += 1
        out.append(line)
    return "\n".join(out)


# ---- 1. the files reachable from main.tex --------------------------------------------------
reachable, queue = set(), ["main.tex"]
while queue:
    rel = queue.pop()
    if rel in reachable:
        continue
    reachable.add(rel)
    path = ROOT / (rel if rel.endswith(".tex") else rel + ".tex")
    if not path.is_file():
        continue
    for m in re.finditer(r"\\input\{([^}]+)\}", strip_comments(read(path))):
        queue.append(m.group(1))

source_files = [ROOT / (r if r.endswith(".tex") else r + ".tex") for r in sorted(reachable)]
source_files = [f for f in source_files if f.is_file()]

# ---- 2. the images and tables those files reference -----------------------------------------
def resolve(img: str):
    cand = ROOT / img
    if cand.is_file():
        return cand.resolve()
    for ext in (".png", ".pdf", ".jpg", ".jpeg", ".eps"):
        if (ROOT / (img + ext)).is_file():
            return (ROOT / (img + ext)).resolve()
    return None


keep = {f.resolve() for f in source_files}
used_images = {}                            # resolved path -> path as written in the source
for f in source_files:
    for m in re.finditer(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", strip_comments(read(f))):
        r = resolve(m.group(1))
        if r is not None:
            keep.add(r)
            used_images[r] = m.group(1)

# ---- 3. what is unreferenced ----------------------------------------------------------------
moves = []
for d in SCAN_DIRS:
    folder = ROOT / d
    if not folder.is_dir():
        continue
    for f in sorted(folder.iterdir()):
        if f.is_file() and not f.name.startswith(".") \
                and f.name not in NEVER_MOVE and f.resolve() not in keep:
            moves.append(f)

# ---- 4. what would be consolidated into figures/ --------------------------------------------
consolidate, collisions = [], []
being_removed = {m.resolve() for m in moves}
for r in sorted(used_images):
    if r.parent.name in IMAGE_DIRS:
        target = FIGDIR / r.name
        # a file already in figures/ with this name is only a problem if it survives step 3
        if target.exists() and target.resolve() != r and target.resolve() not in being_removed:
            collisions.append((r, target))
        else:
            consolidate.append(r)

# ---- report ----------------------------------------------------------------------------------
if moves:
    total = sum(f.stat().st_size for f in moves)
    print(f"UNREFERENCED: {len(moves)} files, {total / 1e6:.1f} MB -> _to_delete/\n")
    by_dir = {}
    for f in moves:
        by_dir.setdefault(f.parent.name, []).append(f)
    for d, files in by_dir.items():
        size = sum(f.stat().st_size for f in files)
        print(f"  {d}/  {len(files)} files, {size / 1e6:.1f} MB")
        for f in files:
            print(f"      {f.name}")
        print()
else:
    print("UNREFERENCED: none\n")

if consolidate:
    print(f"CONSOLIDATE: {len(consolidate)} referenced images -> figures/\n")
    for r in consolidate:
        print(f"      {r.parent.name}/{r.name}")
    print()
else:
    print("CONSOLIDATE: nothing outside figures/\n")

if collisions:
    print("COLLISION: figures/ already holds a different file with this name.")
    print("Resolve by hand, then re-run.\n")
    for r, t in collisions:
        print(f"      {r.parent.name}/{r.name}  vs  figures/{t.name}")
    raise SystemExit(1)

if not APPLY:
    print("dry run. Re-run with --apply to carry this out.")
    raise SystemExit(0)

# ---- apply -------------------------------------------------------------------------------------
for f in moves:
    dest = ROOT / "_to_delete" / f.parent.name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(f), str(dest / f.name))
print(f"moved {len(moves)} unreferenced files into _to_delete/")

rewrites = {}
for r in consolidate:
    shutil.move(str(r), str(FIGDIR / r.name))
    rewrites[used_images[r]] = f"figures/{r.name}"
print(f"moved {len(consolidate)} referenced images into figures/")

changed = 0
for f in source_files:
    text = original = read(f)
    for old, new in rewrites.items():
        text = text.replace("{" + old + "}", "{" + new + "}")
    if text != original:
        f.write_text(text, encoding="utf-8")
        changed += 1
        print(f"  rewrote paths in {f.relative_to(ROOT)}")
print(f"rewrote {changed} source files")

for d in IMAGE_DIRS:
    folder = ROOT / d
    if folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()
        print(f"removed empty {d}/")

print("\nrebuild the document, check it, then delete _to_delete/")
