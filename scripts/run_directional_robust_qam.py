"""Frozen dev/formal PointObstacle comparison; no reference in either stage.

Run ``dev`` first, inspect its selected-alpha.json, then run ``formal``.
The only difference between formal branches is the AM terminal gradient.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import directional_robust_qam as D

spec = importlib.util.spec_from_file_location('point_obstacle_directional_runner',
                                              ROOT/'notebooks/02_qam_point_obstacle.py')
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False))


def protocol():
    return json.loads((ROOT/'configs/directional_robust_qam_v1.json').read_text())


def source_hashes():
    files = ('configs/qam_point_obstacle.json', 'configs/directional_robust_qam_v1.json',
             'notebooks/02_qam_point_obstacle.py', 'notebooks/02_qam_td_terminal_error.py',
             'scripts/directional_robust_qam.py', 'scripts/run_directional_robust_qam.py')
    return {name: digest(ROOT/name) for name in files}


def effective_config(p):
    original, _ = M.load_config()
    return replace(original, actor_updates=p['formal_updates'],
                   curve_updates=tuple(p['formal_checkpoints']),
                   curve_episodes=p['curve_episodes'], final_episodes=p['final_episodes'])


def ensure_manifest(output, p, c):
    output.mkdir(parents=True, exist_ok=True)
    info = json.loads(json.dumps(dict(protocol=p, config=asdict(c), source_hashes=source_hashes())))
    path = output/'frozen-protocol.json'
    if path.exists() and json.loads(path.read_text()) != info:
        raise RuntimeError('Frozen protocol/source differs; choose a new run directory')
    if not path.exists():
        write_json(path, info)
    return digest(path)


def load_or_create_initial(c, seed, folder, source=None):
    """Exactly one initial offline dataset/Base/Critic per seed, shared by branches."""
    folder.mkdir(parents=True, exist_ok=True)
    init = folder/'shared-initialization.pt'
    if not init.exists() and source is not None:
        source = Path(source)/f'seed-{seed}'
        names = ('shared-initialization.pt', 'train-transitions.npz', 'valid-transitions.npz',
                 'base.pt', 'base-log.csv', 'pretrain-critic.csv', 'data-summary.json')
        if not all((source/name).exists() for name in names):
            raise FileNotFoundError(f'Incomplete frozen initialization: {source}')
        prior = json.loads((source.parent/'protocol.json').read_text())['config']
        for key, value in asdict(c).items():
            if key in ('actor_updates', 'curve_updates', 'curve_episodes', 'final_episodes') or key.startswith('reference_'):
                continue
            if json.loads(json.dumps(value)) != prior[key]:
                raise RuntimeError(f'Initialization config mismatch: {key}')
        for name in names:
            shutil.copy2(source/name, folder/name)
        write_json(folder/'initialization-source.json', dict(source=str(source),
                   hashes={name: digest(folder/name) for name in names}))
    if not init.exists():
        task = M.PointObstacle(c)
        train_raw, train_summary = M.collect_episodes(task, c.train_episodes, M.seed_for(seed, 1))
        valid_raw, valid_summary = M.collect_episodes(task, c.valid_episodes, M.seed_for(seed, 2))
        np.savez_compressed(folder/'train-transitions.npz', **train_raw)
        np.savez_compressed(folder/'valid-transitions.npz', **valid_raw)
        write_json(folder/'data-summary.json', dict(train=train_summary, valid=valid_summary))
        M.pretrain(task, M.as_torch(train_raw), c, seed, folder)
    archive = np.load(folder/'train-transitions.npz')
    train = M.as_torch({key: archive[key] for key in archive.files})
    saved = torch.load(init, map_location='cpu', weights_only=True)
    base = M.core.ConditionalVelocity(c); base.load_state_dict(saved['base'])
    for parameter in base.parameters(): parameter.requires_grad_(False)
    critic = M.core.Critic(c); critic.load_state_dict(saved['critic'])
    target = M.core.Critic(c); target.load_state_dict(saved['target'])
    return train, base, critic, target


def new_state(base, critic, target, c, seed):
    actor = copy.deepcopy(base)
    for parameter in actor.parameters(): parameter.requires_grad_(True)
    cr = copy.deepcopy(critic); tg = copy.deepcopy(target)
    return dict(actor=actor, critic=cr, target=tg,
                aopt=torch.optim.Adam(actor.parameters(), lr=c.actor_lr),
                copt=torch.optim.Adam(cr.parameters(), lr=c.critic_lr),
                rng=M.core.generator(M.seed_for(seed, 701)))


def save_checkpoint(folder, method, N, state, base, frozen_hash):
    path = folder/f'{method}-N{N}.pt'
    torch.save(dict(N=N, method=method, actor=state['actor'].state_dict(),
          critic=state['critic'].state_dict(), target=state['target'].state_dict(),
          actor_optimizer=state['aopt'].state_dict(), critic_optimizer=state['copt'].state_dict(),
          rng=state['rng'].get_state(), base_hash=M.model_hash(base),
          frozen_protocol_sha256=frozen_hash), path)


def train_branch(task, train, base, initial_critic, initial_target, c, seed,
                 method, alpha, updates, checkpoints, bank, folder, frozen_hash,
                 diagnostic_interval=10, scale=1.):
    folder.mkdir(parents=True, exist_ok=True)
    state = new_state(base, initial_critic, initial_target, c, seed)
    actor, critic, target = state['actor'], state['critic'], state['target']
    aopt, copt, rng = state['aopt'], state['copt'], state['rng']
    curves, logs, eigen_rows = [], [], []
    for N in range(updates+1):
        if N in checkpoints:
            stats = M.evaluate(task, actor, base, bank)
            curves.append(dict(seed=seed, method=method, N=N, critic_updates=N, **stats))
            save_checkpoint(folder, method, N, state, base, frozen_hash)
            print('CURVE', seed, method, N, json.dumps(stats), flush=True)
        if N == updates: break
        idx = torch.randint(len(train['k']), (c.critic_batch,), generator=rng)
        td_loss, q_mean = M.td_step(critic, target, copt, train, idx, rng, c, actor, base)
        idx = torch.randint(len(train['k']), (c.actor_batch,), generator=rng)
        k, y = train['k'][idx], train['y'][idx]
        z = torch.randn((len(idx), 2), generator=rng)
        flow = torch.randn((len(idx), c.flow_steps, 2), generator=rng)
        if method == 'learned':
            result = M.core.actor_step(actor, aopt, base, critic, task, c, k, y, z, flow, 'learned')
        elif method == 'directional_robust':
            result = D.directional_robust_actor_step(actor, aopt, base, critic, c, k, y, z, flow,
                   alpha, M.core, diagnostic=(N+1)%diagnostic_interval==0)
        elif method == 'global_scale':
            result = D.global_scale_actor_step(actor, aopt, base, critic, c, k, y, z, flow,
                                               scale, M.core)
        else:
            raise ValueError(method)
        if (N+1)%diagnostic_interval == 0:
            eig = result.pop('_eigen', None)
            logs.append(dict(seed=seed, method=method, N=N+1, critic_updates=N+1,
                             td_loss=td_loss, q_mean=q_mean, **result))
            if eig is not None:
                for b in range(len(eig['eigenvalues'])):
                    eigen_rows.append(dict(seed=seed, method=method, N=N+1, batch_index=b,
                        lambda_1=eig['eigenvalues'][b][0], lambda_2=eig['eigenvalues'][b][1],
                        principal_direction_1=[eig['eigenvectors'][b][0][0],eig['eigenvectors'][b][1][0]],
                        principal_direction_2=[eig['eigenvectors'][b][0][1],eig['eigenvectors'][b][1][1]],
                        projection_on_v1=eig['projections'][b][0],
                        projection_on_v2=eig['projections'][b][1],
                        shrink_1=eig['shrink_factors'][b][0], shrink_2=eig['shrink_factors'][b][1]))
    pd.DataFrame(curves).to_csv(folder/f'{method}-curves.csv', index=False)
    pd.DataFrame(logs).to_csv(folder/f'{method}-train.csv', index=False)
    if eigen_rows:
        pd.DataFrame(eigen_rows).to_csv(folder/f'{method}-eigen.csv', index=False)
    return state, curves


def dev(output, source=None):
    p = protocol(); c = effective_config(p)
    frozen_hash = ensure_manifest(output, p, c)
    seed = p['development_seed']; folder = output/f'seed-{seed}'
    train, base, critic, target = load_or_create_initial(c, seed, folder, source)
    bank = M.noise_bank(c, p['curve_episodes'], M.seed_for(seed, p['development_bank_role']))
    results = []
    for alpha in p['alpha_candidates']:
        name = 'learned' if alpha == 0 else f'development_alpha_{str(alpha).replace(".","p")}'
        state, curves = train_branch(M.PointObstacle(c), train, base, critic, target, c, seed,
            name if alpha == 0 else 'directional_robust', alpha,
            p['development_updates'], (p['development_updates'],), bank,
            folder/name, frozen_hash, p['diagnostic_interval'])
        # Each candidate has a separate output folder; its only selection
        # signal is the held-out environment return, never a reference gradient.
        row = dict(alpha=alpha, **curves[-1]); results.append(row)
        pd.DataFrame(results).to_csv(output/'development-alpha-sweep.csv', index=False)
    chosen = sorted(results, key=lambda x:(-x['success_rate'],x['collision_rate'],
                                             -x['discounted_return'],x['alpha']))[0]
    write_json(output/'selected-alpha.json', dict(alpha=chosen['alpha'],
               selection_rule=p['alpha_selection'], selected_from='seed-59000 held-out development bank N=50',
               development_results=results, frozen_protocol_sha256=frozen_hash))
    print('ALPHA FROZEN', chosen['alpha'], flush=True)


def trajectory_records(task, actor, base, bank):
    stats, paths, status = M.evaluate(task, actor, base, bank, trajectories=True)
    lengths = np.isfinite(paths[:, :, 0]).sum(0)-1
    signs = (status == 'success').astype(float)-(status == 'collision').astype(float)
    returns = signs*task.c.gamma**(lengths-1)
    if not np.isclose(float(returns.mean()), stats['discounted_return'], atol=1e-12):
        raise RuntimeError('Per-episode return mismatch')
    return stats, paths, status, lengths, returns


def draw_trajectories(task, records, output, subset, title):
    colors = dict(success='#167a43', collision='#d15342', timeout='#6686ad')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), sharex=True, sharey=True)
    for ax, (method, record) in zip(axes, records.items()):
        stats, paths, status, _, _ = record
        ax.add_patch(Circle(task.c.obstacle_center, task.c.obstacle_radius,
                            color='black', alpha=.38))
        ax.add_patch(Circle(task.c.goal_center, task.c.goal_radius,
                            color='gold', alpha=.5))
        for i, label in enumerate(status):
            if subset == 'success' and label != 'success': continue
            if subset == 'failure' and label == 'success': continue
            path = paths[:, i]; path = path[np.isfinite(path[:, 0])]
            ax.plot(path[:, 0], path[:, 1], color=colors[label], lw=.8, alpha=.65)
        ax.scatter([task.c.goal_center[0]], [task.c.goal_center[1]], marker='*', color='black')
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xlabel='x', ylabel='y')
        ax.set_title(('Ordinary QAM' if method == 'learned' else 'Directional Robust QAM') + '\n'
           + f"Success {round(stats['success_rate']*50)}/50  Collision {round(stats['collision_rate']*50)}/50\n"
           + f"Timeout {round(stats['timeout_rate']*50)}/50  Mean return {stats['discounted_return']:.3f}")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(output/f'{subset}_trajectories.png', dpi=160); plt.close(fig)


def paired_test(task, states, base, c, seed, folder, p):
    output = folder/'test50'; output.mkdir(exist_ok=True)
    bank = M.noise_bank(c, p['paired_episodes'], M.seed_for(seed, p['paired_bank_role']))
    np.save(output/'initial_states.npy', bank['start'])
    np.savez_compressed(output/'noise_bank.npz', **bank)
    records = {method: trajectory_records(task, state['actor'], base, bank)
               for method, state in states.items()}
    rows = []
    for method, (_, _, status, lengths, returns) in records.items():
        for i in range(len(status)):
            rows.append(dict(seed=seed, method=method, episode=i, status=status[i],
                             length=int(lengths[i]), discounted_return=float(returns[i])))
    pd.DataFrame(rows).to_csv(output/'episode_results.csv', index=False)
    old, new = records['learned'][2], records['directional_robust'][2]
    paired = pd.DataFrame(dict(seed=seed, episode=np.arange(len(old)),
                               learned_status=old, robust_status=new))
    paired.to_csv(output/'paired_outcomes.csv', index=False)
    summary = dict(seed=seed, ordinary_failure_to_robust_success=int(((old!='success')&(new=='success')).sum()),
        ordinary_success_to_robust_failure=int(((old=='success')&(new!='success')).sum()),
        both_success=int(((old=='success')&(new=='success')).sum()),
        both_failure=int(((old!='success')&(new!='success')).sum()),
        ordinary_collision_to_robust_success=int(((old=='collision')&(new=='success')).sum()))
    write_json(output/'paired_outcome_summary.json', summary)
    np.savez_compressed(output/'trajectories.npz',
        learned=records['learned'][1], directional_robust=records['directional_robust'][1],
        learned_status=old, robust_status=new)
    for subset in ('all', 'success', 'failure'):
        draw_trajectories(task, records, output, subset, f'Seed {seed}: 50 paired physical trajectories')
    return summary


def formal(output, source):
    p = protocol(); c = effective_config(p)
    frozen_hash = ensure_manifest(output, p, c)
    selection = json.loads((output/'selected-alpha.json').read_text())
    if selection['frozen_protocol_sha256'] != frozen_hash:
        raise RuntimeError('Selection/protocol mismatch')
    alpha = selection['alpha']
    if alpha not in p['alpha_candidates']:
        raise RuntimeError('Alpha not predeclared')
    write_json(output/'formal-frozen-selection.json', dict(alpha=alpha,
        selection_sha256=digest(output/'selected-alpha.json'), frozen_protocol_sha256=frozen_hash,
        formal_seeds=p['formal_seeds']))
    all_curves, all_final, all_pair = [], [], []
    for seed in p['formal_seeds']:
        folder = output/f'seed-{seed}'
        train, base, critic, target = load_or_create_initial(c, seed, folder, source)
        task = M.PointObstacle(c)
        curve_bank = M.noise_bank(c, c.curve_episodes, M.seed_for(seed,p['formal_curve_bank_role']))
        final_bank = M.noise_bank(c, c.final_episodes, M.seed_for(seed,p['formal_final_bank_role']))
        states = {}
        for method in ('learned','directional_robust'):
            branch_dir = folder/method; branch_dir.mkdir(exist_ok=True)
            states[method], curves = train_branch(task, train, base, critic, target, c, seed,
                method, alpha, c.actor_updates, c.curve_updates, curve_bank, branch_dir,
                frozen_hash, p['diagnostic_interval'])
            all_curves.extend(curves)
            stats = M.evaluate(task, states[method]['actor'], base, final_bank)
            all_final.append(dict(seed=seed, method=method, N=c.actor_updates, **stats))
            pd.DataFrame(all_curves).to_csv(output/'formal-curves.csv', index=False)
            pd.DataFrame(all_final).to_csv(output/'formal-final.csv', index=False)
        all_pair.append(paired_test(task, states, base, c, seed, folder, p))
        write_json(folder/'complete.json', dict(seed=seed, alpha=alpha,
                   final=[row for row in all_final if row['seed']==seed], paired=all_pair[-1]))
        print('SEED COMPLETE', seed, flush=True)
    write_json(output/'paired-outcome-summary.json', dict(per_seed=all_pair,
        aggregate={key: sum(row[key] for row in all_pair) for key in all_pair[0] if key!='seed'}))
    print('FORMAL COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('dev','formal'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initialization-source', type=Path)
    args = parser.parse_args()
    if args.stage == 'dev': dev(args.output, args.initialization_source)
    else: formal(args.output, args.initialization_source)


if __name__ == '__main__': main()
