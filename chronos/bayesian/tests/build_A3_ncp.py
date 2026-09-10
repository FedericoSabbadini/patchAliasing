"""Build a separate, distribution-preserving diagnostic copy of reduced A3."""
import ast
import json
from pathlib import Path
from diagnose_A3_sparse import ncp_source

BAYES = Path(__file__).resolve().parents[1]


def main():
    nb = json.loads((BAYES/'model_A3_reduced_localisation_standalone_colab.ipynb').read_text(encoding='utf8'))
    cells = nb['cells']
    source = [''.join(c['source']) for c in cells]
    source[12] = ncp_source(source[12])
    # The equivalence preflight must perturb free sampling coordinates, not deterministics.
    for old, new in [('gamma_config', 'z_gamma'), ('kappa_config', 'z_kappa')]:
        assert source[20].count(f'candidate["{old}"]') == 2
        source[20] = source[20].replace(f'candidate["{old}"]', f'candidate["{new}"]')
    source = [s.replace('model_A3_reduced_direct_v1', 'model_A3_reduced_ncp_config_v1')
               .replace('A3-reduced-direct-checkpoints-v1', 'A3-reduced-ncp-config-checkpoints-v1')
               .replace('direct-zero-sum-no-r-v1', 'ncp-config-zero-sum-no-r-v1') for s in source]
    source[0] = '''# Reduced A3: controlled non-centered geometry parameterization

This diagnostic copy changes ONLY the coordinates of beta, gamma_config and kappa_config.
The likelihood, priors, observations, local effects, recovery truths and acceptance gates
are identical to reduced A3. It is not a new statistical model and is not approved for FULL.
Run PILOT with the fresh RUN_ID `model_A3_reduced_ncp_config_v1`.

The original attached run passed moderate synthetic recovery but stopped during sparse
recovery, before empirical inference. The sparse PILOT design has 14/31 frequencies and
456/615 geometry-frequency-role cells with zero successes. The local comparison and its
limits are documented in `A3_sparse_diagnosis.md`; a one-chain test cannot establish R-hat
or successful recovery. Run all four chains and the existing gates to validate this copy.

Local controlled comparison: centered 7/500 retained divergences, BFMI 0.572;
non-centered geometry blocks 0/500, BFMI 0.978. Both used 1500 warmup iterations,
seed 3351273248 and target_accept 0.95. This supports the coordinate intervention,
but does not establish four-chain convergence or empirical adequacy.

Completed chains are checkpointed individually. The rejection-only guard stays enabled.
Selected retained sampler states (every 100th iteration and divergent transitions) are
saved separately as non-reportable diagnostics, including before an early rejection.
They are states associated with transitions, not the complete divergent trajectories.
'''
    source[11] += '''

### Equivalent sampling coordinates
For independent standard Normal vectors z_beta, z_gamma and z_kappa:

    beta = beta_mu + tau * z_beta
    gamma_config = gamma_bar + sigma_gamma * z_gamma
    kappa_config = kappa_bar + sigma_kappa * z_kappa

Thus beta|hyperparameters, gamma_config|hyperparameters and kappa_config|hyperparameters
retain exactly the Normal distributions specified above. Harmonic and background effects
retain their existing centered zero-sum coordinates; local lock/side effects retain their
existing non-centered coordinates. No prior was widened to make the stress test pass.
'''
    anchor = "                posterior_divergences[0] += int(stats.get('diverging', False))\n"
    assert source[14].count(anchor) == 1
    source[14] = source[14].replace(anchor, anchor + '''                if stats.get('diverging', False) or (iteration-tune) % 100 == 0:
                    state_dir = OUTPUT_ROOT/'sampler_states'
                    state_dir.mkdir(exist_ok=True)
                    state_path = state_dir/f'{label}.chain_{chain_id:02d}.draw_{iteration-tune:06d}.json'
                    cp.atomic_json(state_path, dict(label=label, chain=int(chain_id),
                        retained_iteration=int(iteration-tune), reportable=False,
                        fit_fingerprint=fit_hash, divergent=bool(stats.get('diverging', False)),
                        point={key: np.asarray(value).tolist() for key,value in draw.point.items()}))
                    record_artifact(state_path)
''')
    for c,s in zip(cells,source):
        c['source'] = s.splitlines(keepends=True)
        if c['cell_type']=='code':
            ast.parse(s)
            c['outputs'] = []
            c['execution_count'] = None
    nb['metadata'].pop('widgets',None)
    target = BAYES/'model_A3_reduced_ncp_localisation_standalone_colab.ipynb'
    target.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf8')
    print(target)


if __name__=='__main__':
    main()
