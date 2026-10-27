"""Check executed teaching notebooks and their from-scratch outputs.

This records parity with the frozen research, not new-seed confirmation.
No training is launched and no bundled reference is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import torch
from pagr import runtime as rt
from build_research_notebooks import NAMES, cells_from_script


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_state(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


def collect(root, analytic_run, toy_run):
    notebooks = []
    for name in NAMES:
        source = root/'notebooks'/f'{name}.py'
        path = source.with_suffix('.ipynb')
        nb = nbformat.read(path, as_version=4)
        expected = cells_from_script(source)
        assert [(c.cell_type, c.source) for c in nb.cells] == [(c.cell_type, c.source) for c in expected]
        assert nb.metadata.pagr.source_sha256 == sha(source)
        cells = [c for c in nb.cells if c.cell_type == 'code']
        assert all(c.execution_count is not None for c in cells)
        assert not any(o.output_type == 'error' for c in cells for o in c.outputs)
        # Portable, public-facing outputs: no machine-specific home or credentials.
        for cell in cells:
            for output in cell.outputs:
                plain = output.get('text', '') + '\n'.join(
                    str(v) for k, v in output.get('data', {}).items() if k.startswith('text/'))
                assert '/Users/' not in plain and '/home/' not in plain
        notebooks.append(dict(name=name, code_cells=len(cells), error_outputs=0,
                              source_sha256=sha(source), notebook_sha256=sha(path)))
    analytic = json.loads((analytic_run/'summary.json').read_text())
    toy = json.loads((toy_run/'summary.json').read_text())
    assert all(c['passed'] for c in analytic['assertions'])
    assert all(c['passed'] for c in toy['assertions'])
    arrays, models, actors, runs = 0, 0, 0, []
    for seed in (18001, 18002, 18003):
        current = toy_run/f'seed_{seed}'
        original = rt.ASSETS/'original/pilot/navigation'/f'seed_{seed}'/'critics'
        generated = np.load(current/'data.npz', allow_pickle=False)
        reference = np.load(original/'data.npz', allow_pickle=False)
        assert set(generated.files) == set(reference.files)
        for key in generated.files:
            assert np.array_equal(generated[key], reference[key]), (seed, key)
            arrays += 1
        for activation in ('relu', 'silu'):
            ref_folder = (original if activation == 'relu' else
                          rt.ASSETS/'anchored/models/navigation'/f'seed_{seed}'/'silu')
            for member in range(5):
                g = torch.load(current/'critics'/activation/f'critic_{member}.pt', weights_only=True)
                r = torch.load(ref_folder/f'critic_{member}.pt', weights_only=True)
                assert same_state(g['state_dict'], r['state_dict']), (seed, activation, member)
                assert g['fit']['best_step'] == r['fit']['best_step']
                assert g['fit']['noisy_validation_mse'] == r['fit']['noisy_validation_mse']
                models += 1
        for method in ('am_adam', 'am_2x', 'nominal_guard', 'robust_guard', 'silu'):
            folder = current/'actors'/method
            g = json.loads((folder/'result.json').read_text())
            r = rt.reference('navigation', seed, method)
            errors = {k: abs(g['final'][k]-r['final'][k]) for k in r['final']}
            fine_error = max(abs(g['fine_grid'][k]-r['fine_grid'][k]) for k in r['fine_grid'])
            a = torch.load(folder/'actor.pt', weights_only=True)
            b = torch.load(rt.ASSETS/'reference/followup/navigation'/f'seed_{seed}'/method/'actor.pt',
                           weights_only=True)
            assert same_state(a, b), (seed, method, 'actor')
            assert max(errors.values()) == 0 and fine_error == 0, (seed, method, 'metrics')
            actors += 1
            runs.append(dict(seed=seed, method=method, cost=g['final']['cost'],
                             max_final_metric_error=max(errors.values()), fine_grid_metric_error=fine_error,
                             actor_state_exact=True))
    return dict(
        notebooks=notebooks, environment=toy['versions'], bundled_integrity=rt.verify(),
        analytic=dict(numerical_checks=analytic['checks_total'], passed=analytic['checks_passed'],
                      scalar_one_step=analytic['scalar_one_step'], feedback_models=analytic['feedback_models'],
                      k200=analytic['k200'], outcomes=analytic['outcomes']),
        navigation=dict(from_scratch_offline_arrays_exact=arrays, fitted_critic_states_exact=models,
                        final_actor_states_exact=actors, runs=runs, aggregate=toy['aggregate'],
                        diagnostics=toy['diagnostics'], numerical_checks=toy['assertions']),
        boundary='Local CPU reconstruction of three existing development seeds, not independent confirmation. '
                 'Analytic evidence is finite-budget and conditional, not a universal AM theorem. '
                 'No candidate superiority or new HJBI equation is established. '
                 'Historical six support-certificate warnings remain unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analytic-run', required=True, type=Path)
    parser.add_argument('--toy-run', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('provenance/notebook_validation.json'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = collect(root, args.analytic_run, args.toy_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(notebooks=result['notebooks'],
                          analytic_checks=result['analytic']['numerical_checks'],
                          regenerated_arrays=result['navigation']['from_scratch_offline_arrays_exact'],
                          exact_critics=result['navigation']['fitted_critic_states_exact'],
                          exact_actors=result['navigation']['final_actor_states_exact']), indent=2))
