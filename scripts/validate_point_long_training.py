"""Read-only structural and paired-evaluation checks for the long QAM run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def validate(run, replay=False):
    run = Path(run)
    frozen = json.loads((run/'protocol.json').read_text())
    p, c = frozen['protocol'], frozen['config']
    payload = {key: frozen[key] for key in ('protocol', 'config', 'source_hashes')}
    actual_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert actual_hash == frozen['config_hash'], 'Protocol was edited'
    for name, expected in frozen['source_hashes'].items():
        assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest() == expected, name
    assert (run/'complete.json').exists(), 'Suite incomplete'
    assert len(p['seeds']) == 3 and len(set(p['seeds'])) == 3
    assert p['actor_updates'] == 300
    assert not p['q_value_consistency_is_blocking']
    audit_count = 0
    replay_count = 0
    final_rows = []
    if replay:
        from point_reference_accelerated import M
        torch.set_num_threads(4)
        replay_config = M.Config(**c)
        task = M.PointObstacle(replay_config)
    for seed in p['seeds']:
        sub = run/f'seed-{seed}'
        assert (sub/'complete.json').exists()
        logs = pd.read_csv(sub/'train.csv')
        curves = pd.read_csv(sub/'curves.csv')
        assert not logs[['method', 'N']].duplicated().any()
        final = pd.read_csv(sub/'final-independent.csv').set_index('method')
        base_hashes = []
        initial_actors = []
        for method in ('learned', 'reference'):
            rows = logs[logs.method == method]
            assert set(rows.N) == set(range(1, 301))
            assert len(rows) == 300
            assert np.isfinite(rows[['td_loss', 'matching_loss', 'update_seconds']]).all().all()
            assert set(curves[curves.method == method].N) == set(p['curve_updates'])
            saved = torch.load(sub/f'{method}-N300.pt', map_location='cpu', weights_only=True)
            assert saved['N'] == 300 and saved['config_hash'] == frozen['config_hash']
            base_hashes.append(saved['base_hash'])
            for key in ('actor_optimizer', 'critic_optimizer'):
                assert saved[key]['state']
                assert all(int(value['step']) == 300 for value in saved[key]['state'].values())
            initial = torch.load(sub/f'{method}-N0.pt', map_location='cpu', weights_only=True)
            initial_actors.append(initial['actor'])
            for N in p['audit_updates']:
                if method == 'reference' and N == 0:
                    shared = json.loads((sub/'audit-reference-N0-shared.json').read_text())
                    assert shared['passed']
                    ordinary = json.loads((sub/'audit-learned-N0/summary.json').read_text())
                    assert shared['actor_hash'] == ordinary['actor_hash']
                    continue
                audit = json.loads((sub/f'audit-{method}-N{N}/summary.json').read_text())
                assert audit['passed'] and all(audit['checks'].values())
                assert not audit['q_value_gate']
                assert audit['points'] == len(p['audit_geometry_ids'])+p['audit_endpoint_count']
                progress = json.loads((sub/f'audit-{method}-N{N}/mc-progress.json').read_text())
                assert progress['completed'] == progress['total'] == audit['points']
                assert len(progress['rows']) == 2*audit['points']
                audit_count += 1
            with np.load(sub/f'final-{method}-N300.npz') as data:
                assert len(data['status']) == p['final_episodes']
                metrics = dict(success_rate=float(np.mean(data['status'] == 'success')),
                               collision_rate=float(np.mean(data['status'] == 'collision')),
                               timeout_rate=float(np.mean(data['status'] == 'timeout')),
                               discounted_return=float(data['returns'].mean()))
                for name, value in metrics.items():
                    assert np.isclose(final.loc[method, name], value, atol=1e-12), name
                lengths = np.isfinite(data['paths'][:, :, 0]).sum(0)-1
                np.testing.assert_array_equal(lengths, data['lengths'])
                signs = (data['status'] == 'success').astype(float)-(data['status'] == 'collision').astype(float)
                np.testing.assert_allclose(data['returns'], signs*c['gamma']**(lengths-1), atol=1e-12)
                if replay:
                    init = torch.load(sub/'shared-initialization.pt', map_location='cpu', weights_only=True)
                    base = M.core.ConditionalVelocity(replay_config)
                    base.load_state_dict(init['base'])
                    actor = M.core.ConditionalVelocity(replay_config)
                    actor.load_state_dict(saved['actor'])
                    bank = M.noise_bank(replay_config, p['final_episodes'], M.seed_for(seed, 502))
                    stats, paths, status = M.evaluate(task, actor, base, bank, trajectories=True)
                    np.testing.assert_array_equal(status, data['status'])
                    np.testing.assert_allclose(paths, data['paths'], atol=1e-7, rtol=0., equal_nan=True)
                    for name, value in metrics.items():
                        assert np.isclose(stats[name], value, atol=1e-12)
                    replay_count += 1
            final_rows.append(dict(seed=seed, method=method, **metrics))
        assert len(set(base_hashes)) == 1
        assert all(torch.equal(initial_actors[0][key], initial_actors[1][key]) for key in initial_actors[0])
        a = curves[(curves.method == 'learned') & (curves.N == 0)].iloc[0]
        b = curves[(curves.method == 'reference') & (curves.N == 0)].iloc[0]
        for metric in ('success_rate', 'collision_rate', 'discounted_return'):
            assert a[metric] == b[metric]
    assert audit_count == len(p['seeds'])*(2*len(p['audit_updates'])-1)
    return dict(passed=True, seeds=p['seeds'], actor_updates_per_branch=300,
                td_updates_per_branch=300, actual_gradient_audits=audit_count,
                replayed_final_checkpoints=replay_count,
                final_episodes_per_branch=p['final_episodes'],
                estimator=p.get('reference_estimator', 'policy-table'), final=final_rows,
                interpretation='Integrity and sampled gradient checks; not a universal accuracy or algorithm superiority proof.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replay', action='store_true', help='Re-evaluate all six final Actor checkpoints')
    args = parser.parse_args()
    result = validate(args.run, replay=args.replay)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f'Refusing to overwrite {args.output}')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
