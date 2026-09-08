# A2 reference-frequency revision: validation report

This report belongs to `model_A2_localisation_standalone_colab_improved.ipynb`. The delivered notebook is an unexecuted new copy of the supplied A2 notebook. The supplied notebook and its run directories were not modified.

## What changed

The scientific response remains the A2 localisation indicator: a hit is one when one of the three retained forecast peaks is within ±1 Hz of the injected frequency. The lo, lock, and hi arms remain separate, and repeated Bernoulli trials are pooled as exact Binomial counts. There is no amplitude-recovery response.

The sampler now works in reference-frequency coordinates. At the fixed registered 64 Hz frequency, it samples the baseline at that frequency and the 30 frequency differences from it. The original `beta` and zero-sum `u_harm` are reconstructed as deterministics. The original conditional Normal prior on `beta` is retained as a Potential, so this is a reparameterisation of the same posterior rather than a new independent flat-prior model. The induced covariance of the difference vector is `sigma_harm² (I + 11')`, the covariance corresponding to the original zero-sum Normal in difference coordinates.

The inference runner uses PyMC NUTS with `jitter+adapt_full`, one independently seeded chain at a time, and a checkpoint for each completed chain. A chain is considered complete only after its posterior and sampler statistics have the requested dimensions, finite values, unique coordinates, required variables, and matching data/settings fingerprint. The serialized file is loaded and checked again before it is recorded in the manifest. A returned empty or partial trace is retained under `checkpoints/partial/` for inspection and is never combined with complete chains.

The notebook retains the two existing PPC sets and adds a 615-stratum check over geometry × lock frequency × role. It also adds a fixed sparse synthetic recovery scenario intended to exercise the frequency and background effects. These are additional gates; they cannot make an unconverged posterior reportable.

## Bounded verification performed locally

The validation harness used the existing local FULL count table only as a model-shape and likelihood fixture. It did not treat it as a new empirical fit and did not write into the notebook run directory.

| Check | Result |
|---|---:|
| Notebook syntax, cleared outputs, and nbformat structure | PASS |
| Original and reference-coordinate joint log density, after the constant coordinate Jacobian | maximum absolute error `4.81e-12` |
| Bernoulli versus pooled Binomial likelihood | PASS |
| Reject zero-draw, partial-draw, duplicate-coordinate, and non-finite fixtures | PASS |
| ArviZ/NetCDF and DataTree/NetCDF round-trip fixtures | PASS |
| ArviZ 1 compatibility constructor, including root attributes | PASS |
| Simulated interruption during chain 3 | PASS: chains 1–2 reload; only chains 3–4 run |
| Changed draw setting under an existing checkpoint namespace | PASS: rejected |
| Empty sampler return | PASS: no completed checkpoint is written; partial inspection file only |
| Fine PPC partition | PASS: 615 disjoint strata and all input trials accounted for |
| Sparse recovery generator | PASS: fixed truth generated; hit rate `0.07005` in the validation fixture |

The local environment emitted a PyTensor warning that no BLAS installation was available. This is an environment-performance warning, not a statistical result. A bounded one-chain FULL benchmark was stopped before it produced retained samples, so no speed-up claim is made. The actual pilot is the required performance and convergence test.

A second tiny smoke fit reached NUTS initialisation but was stopped before retaining samples because another local Python process held PyTensor's shared compilation lock. This is an environment contention result, not a model failure. It does not alter the deterministic equivalence and checkpoint tests above.

## How to run

Open the new notebook in Colab or VS Code and run it from a clean runtime. It starts with `RUN_MODE = "PILOT"`, `RUN_ID = "model_A2_reference64_v1"`, four chains, 2,000 warmup iterations, and 2,000 retained draws per chain. The FULL setting is 3,000 retained draws per chain with the same warmup. The notebook automatically reuses compatible measurement artifacts but never imports old posterior samples.

If a runtime stops, rerun the notebook with exactly the same settings and RUN_ID. Completed files are:

```text
checkpoints/<label>.chain_00.nc
checkpoints/<label>.chain_01.nc
checkpoints/<label>.chain_02.nc
checkpoints/<label>.chain_03.nc
```

The first missing chain is run; completed chains are loaded and verified. Changing draws, warmup, seed, model, data population, or other fit settings requires a new RUN_ID. Do not point this revision at the old zero-draw `primary_A2_logit.nc`; it is intentionally isolated under the new namespace.

The PILOT must pass measurement, both synthetic recoveries, convergence, all three PPC sets, and the chain-completeness checks before changing `RUN_MODE` to `FULL`. The FULL result additionally requires prior/link sensitivity fits and the stricter ESS gate. Any failed gate remains `NOT REPORTABLE`.

Source notebook SHA-256 used to build the copy: `4eb6d0fc6ee09bdfd51bd47e0a90c3d3b07a79cf48fb5cde7e0445e95b79523b`.
