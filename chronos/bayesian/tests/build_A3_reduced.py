"""Build the standalone reduced candidate from the reviewed A3 pipeline.

The historical notebooks and coordination exchange are read-only inputs.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BAYES = ROOT / 'chronos/bayesian'
TARGET = BAYES / 'model_A3_reduced_localisation_standalone_colab.ipynb'


def function_source(text, name):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(text, node)


class Standalone(ast.NodeTransformer):
    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'g':
            return ast.copy_location(ast.Name(id=node.slice.value, ctx=ast.Load()), node)
        return self.generic_visit(node)

    def visit_If(self, node):
        # Select the previously screened direct beta / zero-sum harmonic coordinates.
        if isinstance(node.test, ast.Name) and node.test.id == 'direct_harmonic':
            return [self.visit(n) for n in node.body]
        if (isinstance(node.test, ast.UnaryOp) and isinstance(node.test.operand, ast.Name)
                and node.test.operand.id == 'direct_harmonic'):
            return [self.visit(n) for n in node.orelse]
        return self.generic_visit(node)


def main():
    nb = json.loads((BAYES / 'model_A3_localisation_standalone_colab.ipynb').read_text(encoding='utf-8'))
    src = [''.join(c['source']) for c in nb['cells']]
    screened = (BAYES / 'coordination/codex_a3_no_r_screen.py').read_text(encoding='utf-8')
    model = function_source(screened, 'candidate_model')
    start = model.index('    import pandas as pd')
    model = ('def model_A3(frame, scale=None, baseline_scale=None, nu=None,\n'
             '             link="logit", encoding="counts", local_scale=None):\n' + model[start:])
    model = ast.unparse(ast.fix_missing_locations(Standalone().visit(ast.parse(model)))) + '\n'
    # Direct coordinates have no reference-frequency dependency.
    begin, end = model.index('    reference = '), model.index('    side = ')
    model = model[:begin] + model[end:]
    shared = '\n\n'.join(function_source(src[12], name) for name in
                          ('_codes', '_overlap_scaled', '_logP_centred', 'aggregate_trials',
                           'site_design', 'site_effect_table'))
    src[12] = shared + '\n\n' + model + '\n\ndef model_A3_generative(*args, **kwargs):\n    return model_A3(*args, **kwargs)\n'
    recovery = function_source(screened, 'no_r_recovery').replace(
        'def no_r_recovery(frame, g, sparse=False):', 'def _a3_recovery(frame, sparse=False):')
    recovery = ast.unparse(ast.fix_missing_locations(Standalone().visit(ast.parse(recovery))))
    old_recovery = function_source(src[14], '_a3_recovery')
    src[14] = src[14].replace(old_recovery, recovery)
    src[14] = src[14].replace("'sigma_local_baseline',", '').replace("'r_site',", '')
    src[14] = src[14].replace("local_family=['baseline','lock','side']", "local_family=['lock','side']")
    src[14] = src[14].replace("parameterization='reference64-local-weighted-v1'", "parameterization='direct-zero-sum-no-r-v1'")
    src[14] = src[14].replace('return dict(r=posterior_array(idata,"r_site","site"),', 'return dict(')
    src[14] = src[14].replace('+arrays["r"][i,arrays["si"]]', '')
    src[14] = src[14].replace('+arrays["r"][i, si]', '')
    for i in (22, 26):
        src[i] = src[i].replace('"sigma_local_baseline",', '').replace(',"sigma_local_baseline"', '')
    src[6] = src[6].replace('model_A3_local_frequency_diag_v2', 'model_A3_reduced_direct_v1')
    src[6] = src[6].replace('A3-local-baseline-lock-side-v1', 'A3-reduced-lock-side-v1')
    src[6] = src[6].replace('A3-local-frequency-diagonal-checkpoints-v2', 'A3-reduced-direct-checkpoints-v1')
    src[6] = src[6].replace('from importlib import metadata as importlib_metadata',
        'from importlib import metadata as importlib_metadata\nimport importlib.util')
    src[6] = src[6].replace('PROGRESS_EVERY = 200',
        'PROGRESS_EVERY = 200\nEARLY_REJECT_ENABLED = True\nEARLY_REJECT_AFTER = 200\nEARLY_REJECT_DIVERGENCES = 5')
    src[6] = src[6].replace('NUTS_INIT = "jitter+adapt_diag"',
        'NUTS_INIT = "jitter+adapt_diag"\n'
        '# Numba changes only the evaluator; use native compilation when unavailable.\n'
        'COMPILE_MODE = "NUMBA" if importlib.util.find_spec("numba") is not None else None')
    src[6] += '\nprint("Log-density compiler:", COMPILE_MODE or "native PyTensor")\n'
    src[14] = src[14].replace("backend='pymc', init=NUTS_INIT, analysis=ANALYSIS_FINGERPRINT)",
                             "backend='pymc', init=NUTS_INIT, compile_mode=COMPILE_MODE, early_reject=(EARLY_REJECT_ENABLED, EARLY_REJECT_AFTER, EARLY_REJECT_DIVERGENCES), analysis=ANALYSIS_FINGERPRINT)")
    src[14] = src[14].replace("callback=progress, progressbar=False, return_inferencedata=True,",
        "callback=progress, progressbar=False, return_inferencedata=True,\n"
        "                               compile_kwargs={'mode': COMPILE_MODE} if COMPILE_MODE else {},")
    anchor = "                posterior_divergences[0] += int(stats.get('diverging', False))\n"
    assert src[14].count(anchor) == 1, 'Sampling callback anchor changed'
    src[14] = src[14].replace(anchor, anchor +
        "                if (EARLY_REJECT_ENABLED and posterior_divergences[0] >= EARLY_REJECT_DIVERGENCES\n"
        "                        and (iteration - tune) >= EARLY_REJECT_AFTER):\n"
        "                    failure = dict(status='SYNTHETIC_OR_PRIMARY_EARLY_REJECTION', passed=False,\n"
        "                                   reason='Zero-divergence criterion already violated repeatedly',\n"
        "                                   label=label, fit_fingerprint=fit_hash, chain=int(chain_id), retained_draws=int(iteration - tune),\n"
        "                                   divergences=int(posterior_divergences[0]), complete=False,\n"
        "                                   reportable=False)\n"
        "                    cp.atomic_json(OUTPUT_ROOT/'early_rejection.json', failure)\n"
        "                    record_artifact(OUTPUT_ROOT/'early_rejection.json')\n"
        "                    raise RuntimeError('Early rejection: repeated retained divergences; '\n"
        "                                       'inspect early_rejection.json and do not report this fit.')\n")
    src[14] = src[14].replace("            print(f'  Sampled scalar dimensions:",
        "            if COMPILE_MODE:\n"
        "                np.testing.assert_allclose(model.compile_logp(mode=COMPILE_MODE)(point),\n"
        "                                           model.compile_logp()(point), rtol=1e-10, atol=1e-8)\n"
        "                np.testing.assert_allclose(model.compile_dlogp(mode=COMPILE_MODE)(point),\n"
        "                                           model.compile_dlogp()(point), rtol=1e-8, atol=1e-8)\n"
        "                print('  Compiler logp/gradient parity: PASS', flush=True)\n"
        "            print(f'  Sampled scalar dimensions:")
    src[16] = src[16].replace('backend=NUTS_BACKEND,nuts_init=NUTS_INIT,',
                              'backend=NUTS_BACKEND,nuts_init=NUTS_INIT,compile_mode=COMPILE_MODE,early_reject=(EARLY_REJECT_ENABLED, EARLY_REJECT_AFTER, EARLY_REJECT_DIVERGENCES),')
    src[16] = src[16].replace('"numpy","scipy","pandas","pymc"', '"numpy","scipy","pandas","numba","pymc"')
    src[16] = src[16].replace('A2 global priors plus weighted-centred local baseline/lock/side',
                              'A2 global priors plus weighted-centred local lock/side; no local baseline')
    src[16] = src[16].replace('reference64-local-weighted-v1', 'direct-zero-sum-no-r-v1')
    src[16] = src[16].replace('A3-local-effects-sparse-v1', 'A3-reduced-sparse-v1')
    src[5] = ('### 1.3 Settings\nDefault RUN_ID: `model_A3_reduced_direct_v1`. Keep it to resume '
              'identical settings. Each completed chain is saved. A changed model, settings or '
              'data source requires a new RUN_ID. PILOT and FULL have separate directories.\n')
    src[0] = '''# A3 reduced: common frequency baseline and local lock/side contrasts

**Validation status, 2026-09-10: sparse screening rejected for retained divergences.**
With 20 backgrounds, 1500 warmup iterations and the Numba evaluator, chain 1 reached
5 divergences within 676 retained draws. The rejection-only screening stopped before
a complete chain existed. No sparse R-hat, ESS or PPC pass is available. This candidate
is retained for diagnosis and reproducibility; it is not approved for FULL inference.

Standalone candidate for the 15 retrained checkpoints. This removes the local baseline
residual from A3 and samples beta and zero-sum harmonic effects directly. Bernoulli frequency
detection, lo/lock/hi design, priors for retained parameters and acceptance thresholds remain.

The moderate synthetic screen passed (4 chains, 1500 warmup + 2000 retained draws):
max R-hat 1.00337, minimum bulk/tail ESS 2051.7/3843.3, zero divergences,
92.68% parameter interval coverage and 615/615 fine PPC cells covered.
These are synthetic screening results, not evidence of empirical fit or a FULL approval.
Sparse recovery and the empirical PILOT must pass in this notebook before FULL.

The sampler has a rejection-only guard: five retained divergences after 200 posterior draws
write `early_rejection.json` and stop the current chain. This cannot accept a run early and
does not replace the complete-chain diagnostics.

CPU inference; GPU is used only for forecast collection when available. Complete-chain
checkpoints, settings fingerprints and text progress support local VS Code and Colab.
Keep the output directory to resume. A running incomplete chain restarts its warmup.
'''
    src[11] = src[11].replace('Complete A3 specification', 'Complete A3 reduced specification')
    src[11] = src[11].replace('+r_{cj}', '').replace('{r, ell, s}', '{ell, s}')
    src[11] = src[11].replace('The three positive scales', 'The two positive local scales')
    begin = src[11].index('For global frequency effects,')
    end = src[11].index('The site contrast', begin)
    src[11] = src[11][:begin] + (
        'The sampler uses direct Normal beta and ZeroSumNormal harmonic coordinates. '
        'All priors are proper; there is no reference-frequency Flat node or local baseline residual. '
        'On this design, removing r eliminates the extra 30-dimensional baseline null space. '
        'This restricts the baseline to additive geometry and frequency effects. Conditional '
        'PPC must check whether that restriction is adequate. The pilot with six total backgrounds '
        'has 472 sampled scalars (full with 200 backgrounds: 666).\n\n') + src[11][end:]
    src[21] = '''## 5. Reduced-model recovery: moderate and sparse scenarios
Both fixed synthetic truths use the declared reduced model, including local lock/side effects.
Moderate uses up to 3 backgrounds per generator; sparse uses up to 10 available backgrounds
per generator with low detection rates. The actual numbers and hit rate are printed.
Both must pass convergence, >=80% parameter interval coverage and headline coverage.
Synthetic PPC for all three sets is also required. Failed recovery stops before the empirical fit.
'''
    # Exercise the same predictive functions during recovery as for empirical inference.
    needle = '    recovery_status[scenario] = dict('
    insertion = '''    synthetic_ppc = predictive_check(fitted, synthetic_frame)
    synthetic_fine = fine_predictive_check(fitted, synthetic_frame)
    ppc_coverage = synthetic_ppc.groupby("check_set").passed.mean().to_dict()
    ppc_coverage["geometry_frequency_role"] = float(synthetic_fine.passed.mean())
    synthetic_ppc_ok = bool(all(v >= PPC_MIN_COVERAGE for v in ppc_coverage.values()))
    recovery_gates[scenario] = bool(recovery_gates[scenario] and synthetic_ppc_ok)
    save_table(OUTPUT_ROOT/f"synthetic_{scenario}_ppc.parquet", synthetic_ppc)
    save_table(OUTPUT_ROOT/f"synthetic_{scenario}_fine_ppc.parquet", synthetic_fine)
    print("Synthetic PPC coverage:", ppc_coverage, "| final recovery gate:", recovery_gates[scenario])
'''
    src[22] = src[22].replace(needle, insertion + needle)
    src[22] = src[22].replace('headline_covered=headline_covered, passed=recovery_gates[scenario])',
                             'headline_covered=headline_covered, ppc_coverage=ppc_coverage, passed=recovery_gates[scenario])')
    src[33] = ('This is an experimental reduced candidate, delivered without empirical outputs. '
               'The local sparse screening failed on retained divergences. A diagnostic rerun '
               'starts in PILOT from a clean kernel; a matching successful PILOT remains required '
               'for FULL. All complete chains survive interruption; no failed diagnostic is waived.\n')
    for cell, source in zip(nb['cells'], src):
        cell['source'] = source.splitlines(keepends=True)
        if cell['cell_type'] == 'code':
            ast.parse(source)
            cell['outputs'], cell['execution_count'] = [], None
    TARGET.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(TARGET)


if __name__ == '__main__':
    main()
