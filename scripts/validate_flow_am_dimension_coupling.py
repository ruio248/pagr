"""Read-only replay audit of the fixed dimension/coupling Flow AM run."""
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


def definitions(path):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(
        node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    namespace = {'__name__': 'dimension_audit_definitions', 'ROOT': ROOT,
                 'SPEC_PATH': ROOT/'configs/flow_am_dimension_coupling.json'}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    directory = args.run.resolve()
    if args.output and args.output.exists():
        raise FileExistsError('Choose a fresh audit output path')
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = ROOT/'notebooks/02_flow_am_dimension_coupling.py'
    n = definitions(source)
    completion = json.loads((directory/'completion.json').read_text())
    environment = json.loads((directory/'environment.json').read_text())
    config = json.loads((directory/'config.json').read_text())
    c = n['configuration']('formal')
    checks, replays = [], []

    def check(name, condition, **metadata):
        checks.append(dict(name=name, passed=bool(condition), **metadata))
        if not condition:
            raise AssertionError(name)

    check('formal config exactly frozen', config == n['config_dict'](c))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    config_hash = hashlib.sha256((ROOT/'configs/flow_am_dimension_coupling.json').read_bytes()).hexdigest()
    check('source hash start/completion/current',
          source_hash == completion['source_sha256'] == environment['source_sha256'])
    check('config hash start/completion/current',
          config_hash == completion['formal_spec_sha256'] == environment['formal_spec_sha256'])
    notebook = nbformat.read(source.with_suffix('.ipynb'), as_version=4)
    nbformat.validate(notebook)
    module_spec = importlib.util.spec_from_file_location('builder',
                         ROOT/'scripts/build_research_notebooks.py')
    builder = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(builder)
    expected = builder.cells_from_script(source)
    check('notebook cells and source hash',
          [(cell.cell_type, cell.source) for cell in notebook.cells] ==
          [(cell.cell_type, cell.source) for cell in expected] and
          notebook.metadata.pagr.source_sha256 == source_hash)
    code = [cell for cell in notebook.cells if cell.cell_type == 'code']
    check('all notebook code cells executed without errors',
          len(code) == 5 and all(cell.execution_count is not None for cell in code)
          and not any(output.output_type == 'error' for cell in code for output in cell.outputs))
    events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    check('suite completed without failure',
          any(item['event'] == 'suite_completed' for item in events)
          and not any(item['event'] == 'failure' for item in events))
    check('nine seed completions', sum(item['event'] == 'seed_completed' for item in events) == 9)
    check('eighteen actor completions', sum(item['event'] == 'actor_completed' for item in events) == 18)
    check('formal count metadata', completion['base_runs'] == 9 and
          completion['reward_models'] == 45 and completion['actor_runs'] == 18 and
          completion['diagnostic_repetitions'] == 144)
    final = pd.read_csv(directory/'final.csv')
    diagnostics = pd.read_csv(directory/'diagnostics.csv')
    gains = pd.read_csv(directory/'oracle-gains-by-seed.csv')
    check('27 final method rows', len(final) == 27 and
          all(len(frame) == 3 for _, frame in final.groupby(['task', 'seed'])))
    check('144 diagnostic repeats', len(diagnostics) == 144 and len(gains) == 9)

    for task_index, task in enumerate(c.tasks):
        for seed in c.seeds:
            location = directory/task.name/f'seed-{seed}'
            raw_data = np.load(location/'data.npz')
            data, calibration = n['make_data'](c, task, task_index, seed)
            check(f'{task.name}/{seed}: all data replay exactly',
                  all(np.array_equal(value.numpy(), raw_data[key]) for key, value in data.items()))
            check(f'{task.name}/{seed}: calibration replay',
                  calibration == json.loads((location/'calibration.json').read_text()))
            base = n['freeze'](n['Velocity'](task.dim, c.width))
            base.load_state_dict(torch.load(location/'base.pt', weights_only=True))
            metadata = json.loads((location/'pretraining.json').read_text())
            models = []
            for member in range(c.ensemble_members):
                model = n['freeze'](n['RewardModel'](task.dim, c.width))
                model.load_state_dict(torch.load(location/f'reward-{member}.pt', weights_only=True))
                models.append(model)
            hashes = [n['model_hash'](model) for model in [base, *models]]
            check(f'{task.name}/{seed}: frozen model hashes', hashes == metadata['model_hashes'])
            check(f'{task.name}/{seed}: data hashes',
                  all(n['tensor_hash'](value) == metadata['data_hashes'][key]
                      for key, value in data.items()))
            z, xi = n['noise'](c, task.dim, c.final_paths,
                               n['seed_key'](task_index, seed, 700000))
            for method in ('Frozen Base', 'learned', 'oracle'):
                actor = n['actor_copy'](base)
                if method != 'Frozen Base':
                    actor_dir = location/method
                    actor.load_state_dict(torch.load(actor_dir/'final.pt', weights_only=True))
                    initial = torch.load(actor_dir/'checkpoint-0.pt', weights_only=True)
                    check(f'{task.name}/{seed}/{method}: exact Base initialization',
                          all(torch.equal(initial['actor'][key], value)
                              for key, value in base.state_dict().items()) and
                          len(initial['optimizer']['state']) == 0)
                    for step in c.actor_checkpoints:
                        check(f'{task.name}/{seed}/{method}: checkpoint {step}',
                              (actor_dir/f'checkpoint-{step}.pt').exists())
                    budget = json.loads((actor_dir/'budget.json').read_text())
                    check(f'{task.name}/{seed}/{method}: fixed budget and frozen models',
                          budget['updates'] == c.actor_updates and
                          budget['training_rollouts'] == c.actor_updates*c.actor_batch and
                          budget['frozen_hashes_before'] == budget['frozen_hashes_after'] == hashes and
                          budget['oracle_gradient_points'] ==
                          (c.actor_updates*c.actor_batch if method == 'oracle' else 0))
                    updates = pd.read_csv(actor_dir/'updates.csv')
                    check(f'{task.name}/{seed}/{method}: all updates finite',
                          len(updates) == c.actor_updates and
                          np.isfinite(updates.select_dtypes(include='number').to_numpy()).all())
                metrics, arrays = n['evaluate'](actor, base, c, z, xi)
                saved = np.load(location/f'eval-{method.replace(" ", "-")}.npz')
                exact = all(np.array_equal(arrays[key].numpy(), saved[key])
                            for key in ('terminal', 'reward', 'energy', 'cost'))
                row = final[(final.task == task.name) & (final.seed == seed) &
                            (final.method == method)].iloc[0]
                gap = max(abs(float(row[key])-value) for key, value in metrics.items())
                check(f'{task.name}/{seed}/{method}: fixed checkpoint replay',
                      exact and gap < 1e-12, metric_max_abs=gap)
                replays.append(dict(task=task.name, seed=int(seed), method=method,
                                    per_path_exact=exact, metric_max_abs=gap))
            triple = final[(final.task == task.name) & (final.seed == seed)].set_index('method')
            observed = float(gains[(gains.task == task.name) & (gains.seed == seed)].iloc[0]
                             .oracle_gain_J_per_dim)
            actual = (float(triple.loc['learned'].cost)-float(triple.loc['oracle'].cost))/task.dim
            check(f'{task.name}/{seed}: gain formula', abs(observed-actual) < 1e-12)
    for item in completion['figures']:
        check(f'figure {item}', (directory/item).stat().st_size > 1000)
    archive = json.loads((ROOT/'notebooks/archive/flow_am_four_mode/manifest.json').read_text())
    for item in archive['files']:
        for field in ('original', 'archived'):
            check(f'historical archive unchanged: {item[field]}',
                  hashlib.sha256((ROOT/item[field]).read_bytes()).hexdigest() == item['sha256'])
    result = dict(status='passed', run=str(directory.relative_to(ROOT)),
                  source_sha256=source_hash, config_sha256=config_hash,
                  checks=checks, checkpoint_replays=replays,
                  elapsed_seconds=time.monotonic()-started)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed', checks=len(checks),
                          checkpoint_replays=len(replays),
                          elapsed_seconds=result['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
