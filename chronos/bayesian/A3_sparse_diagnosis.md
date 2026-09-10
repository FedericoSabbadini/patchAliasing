# Reduced A3: sparse recovery and measurement diagnosis

Reviewed 2026-09-10. Source: the executed notebook in Downloads, SHA256
`5b285740ce33753b861a625242917a327afbc68909afd5199bc105e17d814b42`.
Its code cells exactly match the repository's reduced A3 notebook at review time.

## What failed in the attached run

Moderate synthetic recovery completed: maximum R-hat 1.00320795, minimum bulk ESS
2019.3368, minimum tail ESS 2855.9418, zero divergences, recovery coverage 0.92786885,
all three PPC coverage summaries 1.0. These are synthetic results.

Sparse synthetic recovery stopped during chain 1 after repeated retained divergences.
At iteration 1800 (1500 warmup plus 300 retained), four divergences were printed;
the next rejection means at least five were subsequently reached. The exact final
retained iteration is not included in the attached output. Empirical inference and
empirical PPC cells were not executed. This is not an empirical PPC rejection.

Regenerating the sparse data using the attached functions, its seed and the same
six-background design gives 3690 groups, 33390 trials, hit rate 0.07349506,
14 of 31 frequencies with zero hits, and 456 of 615 geometry-frequency-role cells
with zero hits. Moderate recovery has zero empty frequencies and four empty cells.
Many zero cells provide upper bounds, with little information about how extremely
negative their individual logits are. A large trial count alone does not resolve this.

## Controlled computational intervention

The three geometry-level Normal hierarchies were still centered. The diagnostic copy
changes only their sampling coordinates:

```
z_beta, z_gamma, z_kappa ~ independent Normal(0, 1)
beta = beta_mu + tau * z_beta
gamma_config = gamma_bar + sigma_gamma * z_gamma
kappa_config = kappa_bar + sigma_kappa * z_kappa
```

Their conditional Normal distributions, hyperpriors, likelihood, estimands and gates
are unchanged. Harmonic/background coordinates and local lock/side coordinates remain
as before. Exact joint log-density equality including the 45-dimensional Jacobian was
verified at three parameter points. This is a test of hierarchical sampling geometry,
not a new likelihood or a claim that every divergence has the same cause.

`tests/diagnose_A3_sparse.py` compares centered and non-centered coordinates on identical
synthetic data, with seed 3351273248, 1500 warmup, 500 retained draws, target_accept 0.95,
diagonal adaptation and Numba. Each version uses ONE chain. It saves complete posterior
states, sampler health and scale summaries for divergent/non-divergent transitions in
`output/a3_sparse_coordinate_diagnosis_v1`. These states are not complete trajectories.
Numerical results are in `comparison.json`. One chain cannot establish R-hat or approve
PILOT/FULL. Local PyMC is 5.28.5 / ArviZ 0.23.4; the attachment used PyMC 6.0.1 / ArviZ
1.1.0, so the controlled comparison is within the local environment, not cross-version proof.

Observed comparison: centered 7 divergences / 500 retained draws, BFMI 0.57221434;
non-centered geometry blocks 0 / 500, BFMI 0.97835664. Wall time including compilation
was about 123 and 133 seconds respectively, so this is not evidence of a speedup.
For the centered chain, median sigma_gamma was 0.03076445 among seven divergent
transitions and 0.25337125 among the other 493 transitions. This is consistent with
a narrow hierarchical region near small sigma_gamma. It is not definitive isolation of
the sole responsible parameter, since three blocks changed and only one seed was tested.

## The stress truth is deliberately far from the prior center

Sparse truth uses beta_bar=-8 and sigma_harm=5, while the fitted priors are
StudentT(4,0,1.5) and HalfStudentT(4,scale=0.5). Under these priors,
P(beta_bar <= -8)=0.00297595 and P(sigma_harm >= 5)=0.00056200.
The stress case is possible under the model, but not a typical prior draw. It jointly
tests sparse information, extreme heterogeneity and prior/likelihood tension. Retain
it as a declared stress case; do not tune a prior around its known answer just to pass.
A single fixed-truth interval-coverage check is not simulation-based calibration.

## A separate scientific limitation

The response records whether any top-three peak is within +/-1 Hz of the arm's target.
It does not establish that injection caused that peak. In the attached calibration table,
kernel-synth/TSMixup lock hit rates are 16.0647%/17.3405%, versus lo 6.0916%/6.6128%
and hi 2.6954%/2.3001%. Background-only lock hit rates are already 6.7385%/4.7799%.
These descriptive numbers do not establish a causal effect, but they rule out treating
all observed lock hits as recovered injected tones. They also do not support assuming
in advance that lock detection must be worse.

For p16-s16 the attached rates are lock 16.0819%, lo 1.7544%, hi 0.8772%.
Background-only rates are lock 7.6023%, lo 2.9240%, hi 0%. The lock increase remains
positive after simple background subtraction. Subtraction is descriptive here, not a
replacement Bayesian analysis and not an estimate of sensitivity/specificity.

If the scientific target is injection-specific reconstruction, define and validate a
matched tone-minus-background contrast before fitting an expanded model. A background-only
forecast is reused across phases: it must not be treated as many independent controls.
Likewise, Binomial aggregation is exact for the assumed common-p independent Bernoulli
likelihood; this algebra does not prove that phase trials sharing a background are independent.
Do not change the measured target or the model to force agreement with an expected sign.

## Next run and verification limits

Use `model_A3_reduced_ncp_localisation_standalone_colab.ipynb`, fresh
RUN_ID `model_A3_reduced_ncp_config_v1`, PILOT, restart kernel and run all. Retain
2000 draws, the existing warmup, all four chains, and all gates. Do not reuse original
posterior checkpoints. Validated measurement caches can be reused through the loader.
Selected retained states are saved before early rejection to make another failure diagnosable.

JSON/code parsing, cleared outputs and saving a diagnostic state before rejection were
checked. The standard nbformat validator could not import because the local rpds DLL is
inaccessible; its schema file was also inaccessible. No complete empirical notebook run
or four-chain non-centered recovery was executed during this diagnosis.

Reference: Stan User's Guide, Efficiency Tuning, hierarchical non-centered parameterization:
https://mc-stan.org/docs/stan-users-guide/efficiency-tuning.html
