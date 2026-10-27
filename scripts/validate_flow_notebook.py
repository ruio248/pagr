"""Validate a completed Flow AM run and replay its saved checkpoints, read-only.

Optionally write a NEW audit JSON (--output); never overwrite prior audits.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import nbformat
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError('Choose a new audit path; historical audits are immutable')
    directory = args.run.resolve()
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = ROOT/'notebooks/02_flow_am_terminal_gradient.py'
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    n = {'__name__': 'flow_validation_definitions'}
    exec(compile(tree, str(source), 'exec'), n)
    config = json.loads((directory/'config.json').read_text())
    c = n['configuration'](config['profile'])
    checks = []

    def check(name, condition, **data):
        checks.append(dict(name=name, passed=bool(condition), **data))
        if not condition:
            raise AssertionError(name)

    check('run config equals current frozen protocol', config == json.loads(json.dumps(n['asdict'](c))))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    completion = json.loads((directory/'completion.json').read_text())
    environment = json.loads((directory/'environment.json').read_text())
    check('source hash identical at start, completion and validation', digest == completion['source_sha256'] == environment['source_sha256'])
    notebook = nbformat.read(source.with_suffix('.ipynb'), as_version=4)
    nbformat.validate(notebook)
    spec = importlib.util.spec_from_file_location('flow_builder', ROOT/'scripts/build_research_notebooks.py')
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    expected = builder.cells_from_script(source)
    check('notebook source and metadata match', [(a.cell_type, a.source) for a in notebook.cells] ==
          [(a.cell_type, a.source) for a in expected] and notebook.metadata.pagr.source_sha256 == digest)
    code = [cell for cell in notebook.cells if cell.cell_type == 'code']
    check('all notebook code cells executed without errors', all(cell.execution_count is not None for cell in code)
          and not any(o.output_type == 'error' for cell in code for o in cell.outputs), code_cells=len(code))
    events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    check('formal suite completed without failures',
          any(e['event'] == 'suite_completed' for e in events) and
          events[-1]['event'] in ('suite_completed', 'oracle_diagnostic_complete') and
          not any(e['event'] == 'failure' for e in events))
    audits = json.loads((directory/'implementation-tests.json').read_text())
    check('all runtime implementation checks passed', all(a['passed'] for a in audits), count=len(audits))
    results = pd.read_csv(directory/'final.csv')
    expected_rows = 12*len(c.seeds)+(2 if c.fine_retrain_first_seed else 0)
    check('all planned result rows present', len(results) == expected_rows, rows=len(results))
    replays = []
    for seed in c.seeds:
        run = directory/f'seed-{seed}'
        base = n['freeze'](n['Velocity'](c.width))
        base.load_state_dict(torch.load(run/'base.pt', weights_only=True))
        metadata = json.loads((run/'pretraining-selection.json').read_text())
        check(f'{seed} base hash', n['model_hash'](base) == metadata['base_hash'])
        models = []
        for j in range(c.members):
            model = n['freeze'](n['RewardModel'](c.width))
            model.load_state_dict(torch.load(run/f'reward-{j}.pt', weights_only=True))
            models.append(model)
        check(f'{seed} reward hashes', [n['model_hash'](m) for m in models] == metadata['reward_hashes'])
        data = n['make_data'](c, seed)
        saved_data = np.load(run/'data.npz')
        check(f'{seed} all observed data replay exactly', all(np.array_equal(t.numpy(), saved_data[key]) for key, t in data.items()))
        for (method, experiment), rows in results[results.seed.eq(seed)].groupby(['method', 'experiment']):
            fine = experiment == 'retrain-2K'
            model_dir = run/(method+('-retrain-2K' if fine else ''))
            actor = n['actor_copy'](base)
            if method != 'Frozen Flow':
                actor.load_state_dict(torch.load(model_dir/'final.pt', weights_only=True))
                initial = torch.load(model_dir/'checkpoint-0.pt', weights_only=True)
                check(f'{seed} {method} {experiment} exact initialization', all(torch.equal(initial['actor'][key], t) for key, t in base.state_dict().items()))
                budget = json.loads((model_dir/'budget.json').read_text())
                check(f'{seed} {method} {experiment} frozen and oracle-isolated', budget['frozen_hashes_before'] == budget['frozen_hashes_after']
                      and budget['actor_true_reward_queries'] == 0 and
                      (method == 'AM-oracle' or budget['actor_true_gradient_points'] == 0))
                updates = pd.read_csv(model_dir/'updates.csv')
                check(f'{seed} {method} {experiment} all updates finite', len(updates) == c.actor_updates and
                      np.isfinite(updates.to_numpy()).all())
            for _, row in rows.iterrows():
                z, xi = n['noise'](c, c.final_paths, n['seed_key'](seed, 700000),
                                   steps=int(row.eval_K) if row.sampler == 'SDE' else c.steps)
                if row.sampler == 'SDE':
                    actual, arrays = n['evaluate'](actor, base, n['replace'](c, steps=int(row.eval_K)), z, xi)
                    saved = np.load(model_dir/'eval.npz' if fine else run/f'eval-{method.replace(" ", "-")}-K{int(row.eval_K)}.npz')
                    exact = all(np.array_equal(arrays[key].numpy(), saved[key]) for key in ('reward', 'cost', 'energy'))
                else:
                    action = n['ode_sample'](actor, z, c.ode_steps)
                    reward = n['true_reward'](action, c)
                    mean, se = n['mean_se'](reward)
                    actual = dict(reward=mean, reward_mc_se=se)
                    saved = np.load(run/f'eval-{method.replace(" ", "-")}-ODE.npz')
                    exact = np.array_equal(action.numpy(), saved['terminal']) and np.array_equal(reward.numpy(), saved['reward'])
                gap = max(abs(actual[key]-row[key]) for key in actual if key in row.index and pd.notna(row[key]))
                check(f'{seed} {method} {experiment} {row.sampler} K{row.eval_K} checkpoint replay', exact and gap < 1e-12, metric_max_abs=gap)
                replays.append(dict(seed=seed, method=method, experiment=experiment, sampler=row.sampler,
                                    eval_K=int(row.eval_K), per_path_exact=exact, metric_max_abs=gap))
    for name in completion['figures']:
        check(f'figure {name}', (directory/name).is_file() and (directory/name).stat().st_size > 1000)
    archive = json.loads((ROOT/'notebooks/archive/manifest.json').read_text())
    for item in archive['files']:
        for key in ('original', 'archived'):
            check(f'archive bytes: {item[key]}', hashlib.sha256((ROOT/item[key]).read_bytes()).hexdigest() == item['sha256'])
    result = dict(run=str(directory.relative_to(ROOT)), status='passed', source_sha256=digest,
                  config_sha256=completion['config_sha256'], checks=checks, checkpoint_replays=replays,
                  paired_seed_summary=json.loads((directory/'paired-seed-summary.json').read_text()),
                  elapsed_seconds=time.monotonic()-started,
                  artifacts=[dict(path=str(p.relative_to(directory)), bytes=p.stat().st_size,
                                  sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                             for p in sorted(directory.rglob('*')) if p.is_file()])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed', checks=len(checks), checkpoint_replays=len(replays),
                          runtime_checks=len(audits), elapsed_seconds=result['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
