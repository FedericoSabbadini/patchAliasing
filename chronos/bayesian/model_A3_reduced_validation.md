# A3 reduced: validation and restart guide

**Final screening outcome, 2026-09-10: NOT VALIDATED for sparse inference.**
The notebook implementation is complete and its software checks pass. The sparse synthetic
screen with 20 backgrounds and Numba reached five retained divergences within the first
676 posterior iterations of chain 1, after 1500 warmup iterations. The predefined
rejection-only rule stopped sampling. No complete sparse chain, R-hat, ESS, recovery
coverage or sparse PPC was produced. The exact record is
`output/a3_reduced_sparse_numba_v1/early_rejection.json`.

This is a sampling failure. It does not establish that the reduced likelihood fits or
fails the empirical data. The moderate result is insufficient to approve FULL. Removing
the additive baseline null space alone has not solved the sparse sampling problem.
The next diagnostic needs states/trajectories near divergences to locate the remaining
curvature; the interrupted screen did not preserve these, so attributing the failure to
a particular scale would be speculation. A further change should preserve the declared
likelihood while testing the relevant parameterization, before proposing another model.

## Deliverable and statistical change

`model_A3_reduced_localisation_standalone_colab.ipynb` is a separate standalone candidate.
The historical A2 and A3 notebooks are preserved. It removes A3's local baseline term
`r_site` and its scale, while retaining the common harmonic baseline and the local lock
and side contrasts. This is a restriction of the statistical model, not merely a faster
parameterization of the full A3 prior. Direct Normal geometry intercepts and zero-sum
harmonic effects replace the reference-frequency coordinates.

The full likelihood, all priors, effect definition, decision thresholds and gates appear
in notebook section 2.2. Detection still means at least one of the top three forecast
spectral peaks within plus/minus 1 Hz of the injected frequency in each arm. There is no
amplitude-recovery response. Lo and hi remain the registered controls.

## Saved evidence

The earlier moderate synthetic direct-coordinate fit is saved under
`output/a3_no_r_screen_long/`. Its four chains each contain 2000 posterior draws after
1500 warmup iterations. Reported diagnostics are max R-hat 1.00337, minimum bulk ESS
2051.7, minimum tail ESS 3843.3 and zero divergences. Parameter interval coverage is
92.68%, including the headline truth; fine PPC covers 615/615 cells.

This is recovery of one fixed synthetic truth. It does not establish empirical adequacy,
calibration across repeated simulated datasets, or predictive performance on unseen
backgrounds. The sparse run interrupted on September 9 wrote no posterior. Its absence
must not be interpreted as a pass.

Fresh validation artifacts use `output/a3_reduced_*` directories. `validation.json` gives
the actual diagnostic, recovery and all three PPC results. `early_rejection.json`, when
present, records an incomplete synthetic screen rejected for repeated retained
divergences; it supplies no R-hat or ESS and does not certify complete chains.

On September 10, the new notebook helpers revalidated the moderate posterior successfully.
All three PPC sets pass at 100%, with the original diagnostics and parameter coverage
reproduced. See `output/a3_reduced_moderate_validation_v1/validation.json`.

The native-compiler software smoke passes exact Bernoulli/Binomial equivalence, complete-posterior loading,
per-chain resume with the sampler disabled, rejection of empty posterior draws and changed
checkpoint settings, and execution of recovery plus all three PPC sets. See
`output/a3_reduced_smoke_v1/software_smoke.json`. Its eight draws per chain are deliberately
insufficient for inference and the smoke's diagnostic gate correctly remains false.

The same software checks also pass with the delivered Numba compiler configuration:
`output/a3_reduced_numba_smoke_v1/software_smoke.json`. Both complete chains were written
and reloaded without sampling; the tiny smoke remains non-reportable.

## Run and resume

From the repository root, run the software check:

```powershell
.\.venv\Scripts\python.exe -u chronos\bayesian\tests\validate_A3_reduced.py --smoke --output output\a3_reduced_numba_smoke_v1
```

This tests small actual NUTS fits, full-posterior loading, completed-chain loading without
sampling, posterior validation and PPC execution. Its tiny draw count cannot certify
convergence. It also verifies exact Bernoulli/Binomial likelihood equivalence.

Run the sparse screening with the notebook's sampler and checkpoints:

```powershell
.\.venv\Scripts\python.exe -u chronos\bayesian\tests\validate_A3_reduced.py --scenario sparse --draws 2000 --tune 1500 --chains 4 --early-reject --output output\a3_reduced_sparse_numba_v1
```

The screening uses ten backgrounds per generator, whereas notebook PILOT recovery uses
the backgrounds available in its own collected groups. Always use the actual recorded
background counts when interpreting either result. The local screening checks ESS >1000;
the notebook retains ESS >400 for PILOT and >1000 for FULL.

Numba is selected for the notebook when installed; otherwise it uses native PyTensor.
The standalone screening runner defaults to Numba, with `--compile-mode native` available.
Three-point native/Numba logp and gradient comparison passed at rtol 1e-10 for logp and
1e-8 for the gradient. Median evaluator time was 3.792 ms native versus 1.365 ms Numba
(2.78x). This microbenchmark is not a claim of equivalent whole-run acceleration.
The sampler performs an additional evaluator comparison before sampling. Compiler choice
is recorded in each fit fingerprint. Use a separate output directory for a different
compiler; do not mix chains from the native and Numba screening runs.

The optional early rejection rule stops after at least 200 retained iterations with at
least five divergences. It can only reject, never accept, and never discards a failed
chain to obtain a passing result. The notebook itself retains all completed chains.
The notebook enables this guard by default (`EARLY_REJECT_ENABLED=True`); its settings
are fingerprinted. Disabling it requires a fresh RUN_ID and does not relax the final
zero-divergence gate. The command-line validation runner controls early rejection with
`--early-reject` instead. A focused callback check verified that warmup divergences are
excluded, both thresholds apply, the failure artifact is non-reportable, and the disable
switch works. This check does not establish statistical convergence.

Rerun the identical command and output directory to load compatible completed chains.
Every chain uses its own deterministic seed and full warmup. An incomplete chain restarts
from warmup; this is not continuation of its sampler state. Data/settings/code fingerprints,
artifact hashes and posterior dimensions prevent mixing incompatible or empty results.

For a diagnostic rerun of the standalone notebook, keep the default fresh
`RUN_ID=model_A3_reduced_direct_v1` and `RUN_MODE=PILOT`, restart the kernel, then run all.
This is a reproduction/diagnosis route, not a recommendation for an expensive FULL run.
Only a matching successful PILOT
allows switching to FULL. Do not copy old A2/A3 posterior checkpoints into this run.
Compatible measurement data may be reused by the notebook's validated source loader.

## Scope of validation

The complete empirical notebook, forecast collection, FULL inference and sensitivity fits
have not been executed as part of the synthetic screen. Recovery failure blocks the
empirical fit. No failure is repaired by lowering gates, increasing predictive intervals
or selecting chains. Completed chains and failure diagnostics remain available for review.
