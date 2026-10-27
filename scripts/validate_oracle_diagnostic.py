"""Read-only validation of the executed Oracle AM diagnostic; optional new JSON audit."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


def definitions():
    path = ROOT/'notebooks/02_flow_am_terminal_gradient.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(
        node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    namespace = {'__name__': 'oracle_validation', 'ROOT': ROOT}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError('Audit output must be new')
    torch.set_num_threads(1)
    n = definitions()
    run = args.run.resolve()
    diagnostic = run/'oracle-diagnostic'
    c = n['configuration']('formal')
    d = n['oracle_diagnostic_configuration'](c)
    checks = []

    def check(name, condition, **details):
        checks.append(dict(name=name, passed=bool(condition), **details))
        if not condition:
            raise AssertionError(name)

    configuration = json.loads((diagnostic/'config.json').read_text())
    completion = json.loads((diagnostic/'completion.json').read_text())
    source_hash = hashlib.sha256((ROOT/'notebooks/02_flow_am_terminal_gradient.py').read_bytes()).hexdigest()
    spec_hash = hashlib.sha256((ROOT/'configs/oracle_am_diagnostic.json').read_bytes()).hexdigest()
    check('full formal config and source hashes',
          configuration['flow'] == json.loads(json.dumps(n['asdict'](c))) and
          configuration['diagnostic'] == json.loads(json.dumps(n['asdict'](d))) and
          configuration['source_sha256'] == completion['source_sha256'] == source_hash and
          configuration['formal_spec_sha256'] == spec_hash)
    check('full scope completed', completion['status'] == 'complete' and
          completion['seeds'] == list(c.seeds) and completion['repeats'] == d.repeats and
          completion['total_probe_groups'] == len(c.seeds)*len(d.checkpoint_methods)*len(d.batches) and
          completion['total_probe_pairs'] == len(c.seeds)*len(d.checkpoint_methods)*len(d.batches)*d.repeats)
    events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    check('no failures and final event complete',
          events[-1]['event'] == 'oracle_diagnostic_complete' and
          not any(event['event'] == 'failure' for event in events))
    replay = json.loads((diagnostic/'historical-replay-audit.json').read_text())
    check('historical baseline replay across all three seeds and methods',
          len(replay) == 9 and
          all(row['status'] == 'retrained and exact-replay-verified' for row in replay))
    for method in ('AM-learned', 'AM-oracle'):
        new_fine = run/f'seed-{c.seeds[0]}'/f'{method}-retrain-2K'
        old_fine = ROOT/d.previous_run/f'seed-{c.seeds[0]}'/f'{method}-retrain-2K'
        for step in (0, 150, 300):
            current = torch.load(new_fine/f'checkpoint-{step}.pt', weights_only=False)
            reference = torch.load(old_fine/f'checkpoint-{step}.pt', weights_only=False)
            check(f'historical first-seed K64 {method} step{step} exact actor and Adam replay',
                  n['exact_tree_equal'](current['actor'], reference['actor']) and
                  n['exact_tree_equal'](current['optimizer'], reference['optimizer']))
        current, reference = np.load(new_fine/'eval.npz'), np.load(old_fine/'eval.npz')
        check(f'historical first-seed K64 {method} exact paired evaluation',
              all(np.array_equal(current[key], reference[key])
                  for key in ('terminal', 'reward', 'cost', 'energy')))
    hashes = json.loads((diagnostic/'frozen-hash-audit.json').read_text())
    check('base and reward models frozen for every seed',
          len(hashes) == 3 and all(row['frozen_hashes_before'] == row['frozen_hashes_after']
                                   for row in hashes))
    frame = pd.read_csv(diagnostic/'checkpoint-evaluations.csv')
    budgets = pd.read_csv(diagnostic/'budgets.csv')
    probes = pd.read_csv(diagnostic/'probe-repetitions.csv')
    summary = pd.read_csv(diagnostic/'probe-summary.csv')
    grid = pd.read_csv(diagnostic/'grid-sensitivity.csv')
    effects = pd.read_csv(diagnostic/'paired-effects-by-seed.csv')
    check('all four arms at all fixed checkpoints',
          len(frame) == 3*4*3 and set(frame['update']) == {0, 150, 300} and
          set(frame['method']) == {'AM-learned-B64', 'AM-oracle-B64',
                                   'AM-learned-B128', 'AM-oracle-B128'})
    check('frozen reference present', len(pd.read_csv(diagnostic/'frozen-reference.csv')) == 3)
    check('complete gradient probe size and unique random streams',
          len(probes) == 768 and len(summary) == 24 and
          probes['train_noise_seed'].nunique() == 3*4*32 and
          probes['eval_noise_seed'].nunique() == 3*4*32)
    for seed in c.seeds:
        source, target = run/f'seed-{seed}', diagnostic/f'seed-{seed}'
        base = n['load_velocity'](source/'base.pt', c, frozen=True)
        base_hash = n['model_hash'](base)
        for method in ('AM-learned-B64', 'AM-oracle-B64', 'AM-learned-B128', 'AM-oracle-B128'):
            location = n['oracle_method_location'](source, target, method)
            budget = json.loads((location/'budget.json').read_text())
            source_name = 'oracle' if '-oracle-' in method else 'learned'
            batch = 128 if method.endswith('B128') else 64
            check(f'{seed} {method} explicit route and budget',
                  budget['gradient_source'] == source_name and budget['batch_size'] == batch and
                  budget['training_paths'] == 300*batch and
                  budget['actor_true_gradient_points'] == (300*batch if source_name == 'oracle' else 0) and
                  budget['frozen_hashes_before'] == budget['frozen_hashes_after'] and
                  budget['frozen_hashes_before'][0] == base_hash)
            initial, opt, _ = n['load_actor_checkpoint'](location/'checkpoint-0.pt', c, base)
            check(f'{seed} {method} exact actor and Adam initialization',
                  n['model_hash'](initial) == base_hash and len(opt.state) == 0)
            final, _, _ = n['load_actor_checkpoint'](location/'checkpoint-300.pt', c, base)
            z, xi = n['noise'](c, c.final_paths, n['seed_key'](seed, 700000))
            actual, arrays = n['evaluate'](final, base, c, z, xi)
            row = frame[frame.seed.eq(seed) & frame.method.eq(method) & frame['update'].eq(300)].iloc[0]
            saved = np.load(target/f'eval-{method}-step300-K32.npz')
            check(f'{seed} {method} final paired arrays and finite-chain objective',
                  all(np.array_equal(arrays[key].numpy(), saved[key]) for key in ('reward', 'cost', 'energy')) and
                  max(abs(actual[key]-row[key]) for key in ('reward', 'cost', 'energy')) < 1e-12 and
                  abs(row.cost-(row.energy-c.beta*row.reward)) < 1e-6)
        for cp_index, (name, step) in enumerate(d.checkpoint_methods):
            for repetition in range(d.repeats):
                role = d.train_noise_offset+cp_index*100000+4*repetition
                a = n['noise'](c, 64, n['seed_key'](seed, role))
                b = n['noise'](c, 128, n['seed_key'](seed, role))
                check(f'{seed} checkpoint{cp_index} repetition{repetition} B64 prefix',
                      all(torch.equal(x, y[:64]) for x, y in zip(a, b)))
            for batch in d.batches:
                row = summary[summary.seed.eq(seed) & summary.checkpoint_actor.eq(name) &
                              summary.checkpoint_step.eq(step) & summary.batch_size.eq(batch)].iloc[0]
                saved = np.load(target/f'probe-{name}-{step}-B{batch}.npz')
                ql = torch.from_numpy(saved['q_learned'])
                qo = torch.from_numpy(saved['q_oracle'])
                difference = ql-qo
                vl = n['trace_covariance'](ql)
                vo = n['trace_covariance'](qo)
                vd = n['trace_covariance'](difference)
                cross = n['trace_covariance'](qo, difference)
                check(f'{seed} {name}/{step} B{batch} covariance with cross term',
                      abs(vl-(vo+vd+2*cross)) < 1e-8*max(1., abs(vl)) and
                      max(abs(row[col]-value) for col, value in [
                          ('trace_cov_qL', vl), ('trace_cov_qO', vo),
                          ('trace_cov_d', vd), ('trace_cross_cov_qO_d', cross)]) < 1e-9)
                subset = probes[probes.seed.eq(seed) & probes.checkpoint_actor.eq(name) &
                                probes.checkpoint_step.eq(step) & probes.batch_size.eq(batch)]
                check(f'{seed} {name}/{step} B{batch} paired virtual-step MC SE',
                      len(subset) == 32 and
                      np.allclose(saved['D_J_path'].mean(axis=1), subset.D_J, atol=1e-9) and
                      np.allclose(saved['D_R_path'].mean(axis=1), subset.D_R, atol=1e-9) and
                      np.allclose(saved['D_J_path'].std(axis=1, ddof=1)/np.sqrt(1024),
                                  subset.D_J_mc_se, atol=1e-9) and
                      np.allclose(saved['D_R_path'].std(axis=1, ddof=1)/np.sqrt(1024),
                                  subset.D_R_mc_se, atol=1e-9))
    for metric in ('cost', 'reward', 'energy'):
        for schedule in ('equal_updates', 'equal_rollouts'):
            detail, _ = n['seed_effects'](frame, metric, schedule, c.actor_updates)
            expected = pd.DataFrame(detail).sort_values(['seed', 'effect']).reset_index(drop=True)
            observed = effects[effects.metric.eq(metric) & effects.schedule.eq(schedule)].sort_values(
                ['seed', 'effect']).reset_index(drop=True)
            check(f'{metric} {schedule} signed paired effects', np.allclose(expected.value, observed.value))
    check('K32/K64 full three-seed B64 grid',
          len(grid[grid.train_K.eq(64) & grid.eval_K.eq(64) &
                   grid.method.isin(('AM-learned-B64', 'AM-oracle-B64'))]) == 6 and
          len(budgets[budgets.train_K.eq(64) & budgets['update'].eq(300)]) == 6)
    check('all four figures and report saved',
          all((diagnostic/name).stat().st_size > 1000 for name in completion['figures']) and
          (diagnostic/'report.md').stat().st_size > 1000)
    result = dict(status='passed', run=str(run.relative_to(ROOT)), checks=checks,
                  summary=dict(checks=len(checks), main_rows=len(frame), probe_pairs=len(probes),
                               gradient_groups=len(summary), grid_rows=len(grid),
                               source_sha256=source_hash, formal_spec_sha256=spec_hash))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
