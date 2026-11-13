"""Execute the canonical probing notebooks with this Python environment.

Full execution saves outputs in the canonical notebooks; smoke copies live under
notebooks/_run/probing/validation so a diagnostic never overwrites full results.
Each completed cell is checkpointed, including when a later cell fails.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import traceback

import nbformat
from nbclient import NotebookClient

if __package__:
    from . import checkpointing as cp
else:
    import checkpointing as cp


def _execute_ipython(notebook, checkpoint, validation, stem):
    """Execute synchronous Python cells without Jupyter's local TCP sockets.

    This is useful in restricted Windows environments where asyncio socketpair
    creation blocks. Rich display outputs are still captured through IPython.
    """
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    shell = InteractiveShell.instance()
    shell.reset(new_session=False)
    shell.autoawait = False
    execution_count = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        execution_count += 1
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=False)
        cell.execution_count = execution_count
        cell.outputs = []
        for name, value in (("stdout", captured.stdout), ("stderr", captured.stderr)):
            if not value:
                continue
            lines = value.splitlines(keepends=True)
            if len(lines) > 60:
                log = validation / f"{stem}_cell{index}_{name}.log"
                cp.atomic_write(log, lambda p, text=value: p.write_text(text, encoding="utf-8"))
                value = "".join(lines[:15]) + f"\n[{len(lines)-60} lines omitted; full log: {log}]\n" + "".join(lines[-45:])
            cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=value))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data,
                                                      metadata=output.metadata or {}))
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            cell.outputs.append(nbformat.v4.new_output(
                "error", ename=type(error).__name__, evalue=str(error),
                traceback=traceback.format_exception(type(error), error, error.__traceback__)))
        checkpoint(cell=cell, cell_index=index)
        if error is not None:
            raise error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--kind", choices=("pure", "complex", "both"), default="both")
    parser.add_argument("--engine", choices=("ipython", "nbclient"), default="ipython",
                        help="IPython avoids local TCP sockets; nbclient uses a separate Jupyter kernel")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    validation = repo / "notebooks" / "_run" / "probing" / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    status_path = validation / f"notebook_execution_{args.mode}.json"
    state = {"mode": args.mode, "kind": args.kind, "engine": args.engine,
             "status": "running", "completed": [],
             "pid": os.getpid(), "python": sys.executable,
             "started_utc": datetime.now(timezone.utc).isoformat()}
    cp.atomic_json(status_path, state)
    try:
        for kind, filename in (("pure", "probing_pure_tones.ipynb"),
                               ("complex", "probing_complex_signals.ipynb")):
            if args.kind not in (kind, "both"):
                continue
            source = repo / "notebooks" / "signal_analysis" / filename
            destination = source if args.mode == "full" else validation / f"smoke_{filename}"
            notebook = nbformat.read(source, as_version=4)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.outputs, cell.execution_count = [], None
            notebook.metadata["probing_execution"] = {
                "mode": args.mode, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
            def save():
                cp.atomic_write(destination, lambda path: nbformat.write(notebook, path))
            save()
            state.update(current_notebook=filename, output=str(destination))
            cp.atomic_json(status_path, state)
            print(f"Executing {filename} ({args.mode}) -> {destination}", flush=True)
            def checkpoint(cell, cell_index, **_kwargs):
                save()
                state.update(current_cell=cell_index,
                             updated_utc=datetime.now(timezone.utc).isoformat())
                cp.atomic_json(status_path, state)
                print(f"{filename}: cell {cell_index} complete", flush=True)
            try:
                if args.engine == "ipython":
                    old_env = {key: os.environ.get(key) for key in
                               ("PATCHALIASING_PROBING_MODE", "MPLBACKEND")}
                    try:
                        os.environ.update(PATCHALIASING_PROBING_MODE=args.mode, MPLBACKEND="Agg")
                        _execute_ipython(notebook, checkpoint, validation, f"{args.mode}_{Path(filename).stem}")
                    finally:
                        for key, value in old_env.items():
                            if value is None:
                                os.environ.pop(key, None)
                            else:
                                os.environ[key] = value
                else:
                    client = NotebookClient(notebook, timeout=None, kernel_name="python3",
                                            resources={"metadata": {"path": str(repo)}})
                    manager = client.create_kernel_manager()
                    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
                    client.km = manager
                    client.on_cell_executed = checkpoint
                    client.execute(env={**os.environ, "PATCHALIASING_PROBING_MODE": args.mode,
                                        "PYTHONUTF8": "1", "MPLBACKEND": "Agg"})
            except BaseException as exc:
                notebook.metadata["probing_execution"].update(status="error", error=str(exc))
                raise
            else:
                notebook.metadata["probing_execution"].update(
                    status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
                state["completed"].append(filename)
            finally:
                save()
            # HTML is a local inspection artifact, not a report deliverable.
            from nbconvert import HTMLExporter
            html, _resources = HTMLExporter().from_notebook_node(notebook)
            preview = validation / f"{args.mode}_{Path(filename).stem}.html"
            cp.atomic_write(preview, lambda p: p.write_text(html, encoding="utf-8"))
        state.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        state.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        cp.atomic_json(status_path, state)


if __name__ == "__main__":
    main()
