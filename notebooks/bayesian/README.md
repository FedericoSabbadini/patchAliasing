# Final implementation: the Deliverable 2 Bayesian models

| File | What it is |
|---|---|
| [`deliverable3_bayesian_models.ipynb`](deliverable3_bayesian_models.ipynb) | The notebook. Open it in Colab and choose **Run all**. |
| [`implementation_report.pdf`](implementation_report.pdf) | The companion report: what is implemented, every choice and its reason, the fitting procedure, the gates, and the limitations. Read it first. |
| [`figure_ppc_dispersion.py`](figure_ppc_dispersion.py) | Draws Figure 4 of Deliverable 3 (Appendix E) from the `tables/05_ppc.csv` of a finished run. |

## What this fits

The five model families of `coursework/deliverable2_v0`, stated in full in
`coursework/deliverable3/sections/E_bayesian_appendix.tex`, on the fifteen retrained patch/stride
geometries of `tab:hfModels`.

| Claim | Model | Estimand | Predicted |
|---|---|---|---|
| H1 behavioural | A | `exp(beta_bar)`, recovery at a candidate over its controls | `< 1` |
| M1 mitigation | A′ | `delta_O`, the overlap slope | `> 0` |
| H1 representational | B | `exp(theta_lock)`, codelength at a lock over elsewhere | `> 1` |
| H2 phase invariance | C | `sigma_phase`, the spread across phase | `≈ 0` |
| H3 location | D1 | `theta_S`, `theta_P`, dip depth on each grid | both `< 0` |
| H3a, H3b movement | D2 | `kappa_S`, `kappa_P`, measured over predicted spacing | `= 1` |

## Running it

0. **Push first.** On Colab the notebook clones this repository and imports
   `notebooks/support_scripts/` from the clone, so the scripts on GitHub must be the current ones.
   Part 0.7 stops with a named message if the clone is older than the notebook.
1. **Runtime > Run all.** The first cell installs a pinned environment and restarts the runtime
   once; Colab reports a crashed session, which is that restart. Choose **Run all** again.
2. Every setting is in one form, **Part 0.4**. Nothing elsewhere needs editing; the defaults are
   the final, reportable run.
3. The notebook decides for itself whether it needs to collect observations or only analyse them,
   resumes any stage already finished, and stops with a named reason if a gate fails.
4. **If the session drops**, open the notebook again and choose **Run all**: the run folder is found
   through its pointer on Drive and every finished stage reloads from its checkpoint.

### Where the run is written

On Colab everything goes to Google Drive under `MyDrive/patchAliasing_D3_final/`
(`DRIVE_ROOT`). The first launch creates a folder with a name no earlier run can have,
`<mode>_snr<amplitude>_<UTC date-time>_<random suffix>`, and writes the pointer
`ACTIVE_RUN_<mode>_snr<amplitude>.json` beside it; every later launch resumes that folder.
`NEW_RUN = True` starts another run from zero in another new folder, and `RUN_NAME = "<folder>"`
opens a named one. Nothing is ever written into an older folder and nothing is deleted. The folder
records its creation, the repository commit it started from and one entry per session in
`run_info.json`; a resumed session checks the clone out at that commit before importing anything.

`REGENERATE_SIGNALS = True` (the default) generates every background again from its seed in every
collecting session. In a new folder that is how the signal archive `data/signals/` is written; on a
resume each regenerated draw is compared with the archived one and the largest difference is
recorded in the collection manifest.

### Cost

From an empty folder a full run collects 1,116,000 forecasts, 3,000 of them the background-only
arm, and then fits about sixty-six posteriors, so it is a day of wall time on a CPU runtime and rather less on a GPU one. Every stage
is checkpointed; an interrupted session costs only the table or the pair of chains it was writing.

The contrast collection is also checkpointed block by block (`data/raw/_partial/`), so a
disconnect in the middle of a geometry costs one block of candidate frequencies; posterior
predictive checks are checkpointed per fit, and every fit per pair of chains.

`MODE = "preflight"` runs everything up to and including the parity gates and parameter recovery,
which are themselves short fits, then stops before the reportable ones.
`MODE = "smoke"` rehearses the whole chain in minutes and can never produce a verdict.
`STANDALONE_FIGURES = True` with `RUN_NAME` set to a finished run regenerates Parts 6 and 7
from the saved artifacts alone.

`RECOVERY_DENOMINATOR` chooses what the amplitude recovery is measured against:
`true_continuation`, the default, which is how Deliverable 2 and Appendix E define it
(R = A_pred / A_true), or `injected`, the amplitude the tone was injected at, kept as a robustness
reading. Both are computed from the same collection and Part 2.4 prints the
difference, so the choice can be changed without collecting again.

Part 0.8 prints, and saves as `00_specification_notes.csv`, every place the implementation
interprets a clause of the documents, corrects one, or departs from one. Section 3.9 of the report
repeats that list with the reasoning.

## What it writes

Into `MyDrive/patchAliasing_D3_final/<run id>/`:

| Where | What |
|---|---|
| `run_info.json`, `analysis_manifest.json` | creation, commit, sessions; every artifact with its SHA-256 |
| `data/` | the five collected tables, their shards, the collection manifest and the signal archive |
| `04_*.nc`, `05_sens_*.nc`, `03_*.nc` | every posterior (ArviZ NetCDF), one file per fit and per pair of chains |
| `tables/*.csv` | every result table; `bayesGates.tex` (every claim) and `bayesFitSummary.tex` (every fit) |
| `figures/*.png`, `figures/report/*.pdf` | every figure; the prior-posterior figures of Part 7.3 also as vector PDF |
| `export/` | `az.summary` of every fit, every estimand draw and the prior draws in long form, Model A's configuration effects, the environment, a README |
| `05_run_summary.json` | every gate result, every verdict, the measured timings and the limitations |

The notebook ends with **Part 7**: one table of every fit with its convergence and the gates it
feeds, one table of every claim with its worst convergence, each gate and the verdict, and, as the
last output, prior against posterior for every claim with the probability of each decision rule
before and after the data.

## Changes to the shared scripts

Three properties the analysis needed belong to the collection library rather than to any one
notebook, so they were fixed in `support_scripts/` instead of patched at run time. Section 8.1 of
the report states them and the checks they were verified against.

| File | Change |
|---|---|
| `probe_lib.py` | `build_context` takes an optional `amp`; without it the behaviour is unchanged. |
| `collect.py` | `Config.tone_snr`, so the tone amplitude enters the design fingerprint. |
| `collect.py` | `Config.sites_per_block`: the contrast collector forwards candidate frequencies in blocks, with resident memory reported. The table it produces is identical, row for row. |
| `collect.py` | `check_design`, `merge`, `load_collection` and `collect_all` take `response`, either `localisation` or `contrast`. The default is `localisation`, so every existing caller is unaffected. |
| `collect.py` | `derive_sites` records `first_harmonic`, and a candidate with no spectral peak in band is a miss rather than a validation failure. |
| `collect.py` | `Config.null_arm`: one background-only forecast per realisation, and the net recovery `a_net` per arm. |
| `collect.py` | The contrasts of a geometry are checkpointed block by block while they are collected; the package versions are recorded per session instead of gating a resume. |
| `collect.py`, `probe_lib.py` | `collect_all(regenerate_signals=True)` / `save_signal_pool(regenerate=True)`: every background regenerated from its seed and checked against the archive. |

Both files are hashed into every collection's design fingerprint, so a collection made before these
changes will refuse to be resumed or merged, and will say so.

## Notes

- No notebook is modified. This one is new and stands on its own.
- The verdict table reports, per claim, the **pre-gate** reading, **each gate independently**, and
  the **post-gate** verdict. `NOT REPORTABLE` and `NOT IDENTIFIED` are not null results.
- The tone amplitude, 1.5 over a unit-variance background, is the one design constant Deliverable 2
  does not fix; Appendix E of Deliverable 3 reports it. It is set in Part 0.4, recorded beside the
  collected data, and a run at a different amplitude is refused rather than
  merged.
