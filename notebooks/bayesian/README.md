# Final implementation: the Deliverable 2 Bayesian models

| File | What it is |
|---|---|
| [`deliverable2_bayesian_models.ipynb`](deliverable2_bayesian_models.ipynb) | The notebook. Open it in Colab and choose **Run all**. |
| [`implementation_report.pdf`](implementation_report.pdf) | The companion report: what is implemented, every choice and its reason, the fitting procedure, the gates, and the limitations. Read it first. |

## What this fits

The five model families of `coursework/deliverable2_v0`, stated in full in
`coursework/deliverable3/sections/B_bayesian_appendix.tex`, on the fifteen retrained patch/stride
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

1. **Runtime > Run all.** The first cell installs a pinned environment and restarts the runtime
   once; Colab reports a crashed session, which is that restart. Choose **Run all** again.
2. Every setting is in one form, **Part 0.4**. Nothing elsewhere needs editing.
3. The notebook decides for itself whether it needs to collect observations or only analyse them,
   resumes any stage already finished, and stops with a named reason if a gate fails.

From an empty folder a full run collects 1,113,000 forecasts and then fits about sixty-five
posteriors, so it is a day of wall time on a CPU runtime and rather less on a GPU one. Every stage
is checkpointed; an interrupted session costs only the table or the pair of chains it was writing.

`MODE = "preflight"` runs everything up to and including the parity gates and parameter recovery,
which are themselves short fits, then stops before the reportable ones.
`MODE = "smoke"` rehearses the whole chain in minutes and can never produce a verdict.
`STANDALONE_FIGURES = True` regenerates Part 6 from the saved artifacts alone.

`RECOVERY_DENOMINATOR` chooses what the amplitude recovery is measured against: `injected`, which
is how Appendix B defines it and the default, or `true_continuation`, which is what the shared
estimator has always returned. Both are computed from the same collection and Part 2.4 prints the
difference, so the choice can be changed without collecting again.

Part 0.8 prints, and saves as `00_specification_notes.csv`, every place the implementation
interprets a clause of the documents, corrects one, or departs from one. Section 3.9 of the report
repeats that list with the reasoning.

## What it writes

Into `patchAliasing/final_implementation/<run id>/` on Drive: `analysis_manifest.json`, the five
collected tables under `data/`, every result table as CSV under `tables/`, about thirty figures
under `figures/`, one posterior checkpoint per fit, and `05_run_summary.json` with every gate
result, every verdict, the measured timings and the limitations.

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

Both files are hashed into every collection's design fingerprint, so a collection made before these
changes will refuse to be resumed or merged, and will say so.

## Notes

- No notebook is modified. This one is new and stands on its own.
- The verdict table reports, per claim, the **pre-gate** reading, **each gate independently**, and
  the **post-gate** verdict. `NOT REPORTABLE` and `NOT IDENTIFIED` are not null results.
- The tone amplitude is the one design constant the report does not fix. It is set in Part 0.4,
  recorded beside the collected data, and a run at a different amplitude is refused rather than
  merged.
