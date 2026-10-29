"""Equal-update, three-seed long-training study of obstacle QAM.

Uses the original CPU training functions and an equivalent accelerated
current-policy table. A NEW gradient-focused protocol controls admission;
historical audits remain untouched. Checkpoints include Adam and RNG state.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

from point_reference_accelerated import M, ROOT, AcceleratedPolicyTable, sample_actor_tensor


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temporary.replace(path)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_mc(task, actor, base, k, p, a, episodes, seed, device):
    """Original NumPy simulator and IID RNG; GPU accelerates only the Actor.

    Never reads a numerical table or Critic. The RNG consumption, return and
    sample-covariance estimator match mc_fixed_q, including unused flow draws.
    """
    c = task.c
    rng = np.random.default_rng(seed)
    state = np.broadcast_to(np.asarray(p, dtype=float), (episodes, 2)).copy()
    active = np.ones(episodes, dtype=bool)
    returns = np.zeros(episodes)
    first_noise = rng.normal(size=(episodes, 2))
    policy = actor  # Caller supplies a frozen device copy, reused across points.
    for j in range(k, c.horizon):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        if j == k:
            action = np.broadcast_to(np.asarray(a, dtype=float), (len(ids), 2))
            noise = first_noise[ids]
        else:
            z = rng.normal(size=(len(ids), 2)).astype('f')
            rng.normal(size=(len(ids), c.flow_steps, 2))
            kk = torch.full((len(ids),), j, device=device, dtype=torch.long)
            action = sample_actor_tensor(policy, c, kk,
                       torch.as_tensor(state[ids], device=device, dtype=torch.float32),
                       torch.as_tensor(z, device=device)).cpu().numpy()
            noise = rng.normal(size=(len(ids), 2))
        pn, reward, done, _ = task.transition(j, state[ids], action, noise)
        returns[ids] += c.gamma**(j-k)*reward
        state[ids] = pn
        active[ids[done]] = False
    score = ((first_noise-first_noise.mean(0))*(returns-returns.mean())[:, None]
             *episodes/(episodes-1)/c.sigma_env)
    chain = c.delta*(1-np.tanh(a)**2)
    return dict(q=float(returns.mean()), q_se=float(returns.std(ddof=1)/np.sqrt(episodes)),
                gradient=(chain*score.mean(0)).tolist(),
                gradient_se=(chain*score.std(0, ddof=1)/np.sqrt(episodes)).tolist())


class MonteCarloReference:
    """Local, high-sample current-policy score estimate, without value-grid bias."""
    def __init__(self, task, actor, base, episodes, seed, device):
        self.task, self.c = task, task.c
        self.actor_hash, self.base_hash = M.model_hash(actor), M.model_hash(base)
        self.policy = copy.deepcopy(actor).to(device).eval()
        self.base, self.episodes, self.seed, self.device = base, episodes, seed, device
        self.seconds, self.action_calls = 0., 0
        self.gradient_se, self.q_se = None, None

    def assert_current(self, actor, base):
        if M.model_hash(actor) != self.actor_hash or M.model_hash(base) != self.base_hash:
            raise ValueError('stale MC reference')

    def q_gradient(self, k, p, a):
        k = np.broadcast_to(np.asarray(k, dtype=int), (len(p),))
        start = time.monotonic()
        rows = [independent_mc(self.task, self.policy, self.base, int(k[i]), p[i], a[i],
                              self.episodes, self.seed+7919*i, self.device)
                for i in range(len(p))]
        self.seconds += time.monotonic()-start
        self.gradient_se = np.asarray([row['gradient_se'] for row in rows])
        self.q_se = np.asarray([row['q_se'] for row in rows])
        return np.asarray([row['q'] for row in rows]), np.asarray([row['gradient'] for row in rows])


def gradient_gate(g, higher, mc, se, critic, protocol, reference_se=None):
    """No Q scalar can trigger this gate. Fixed tolerances precede training."""
    gap = rms(critic-g)
    error_limit = max(protocol['gradient_error_absolute_floor'],
                      protocol['gradient_error_to_critic_gap_max']*gap)
    se_limit = max(protocol['mc_se_absolute_floor'],
                   protocol['gradient_error_to_critic_gap_max']*gap)
    diff = np.abs(g-mc)
    reference_se = np.zeros_like(se) if reference_se is None else reference_se
    # Unlike a fixed value table, a stochastic training reference also has MC
    # uncertainty. Independent estimates require the sum of their variances.
    comparison_se = np.hypot(se, reference_se)
    positive = comparison_se >= 1e-8
    zz = diff[positive]/comparison_se[positive]
    z_rms = rms(zz) if len(zz) else 0.
    z_max = float(np.max(zz)) if len(zz) else 0.
    norms = np.linalg.norm(mc, axis=1)
    reliable_direction = norms > np.maximum(protocol['gradient_nondegenerate_norm'],
                                             3*np.linalg.norm(se, axis=1))
    opposite = int(np.sum(((g*mc).sum(1) < 0.) & reliable_direction))
    checks = dict(
        mc_gradient_consistency=(z_rms <= protocol['gradient_z_rms_max']
                                 and z_max <= protocol['gradient_z_max']),
        zero_se_absolute=(bool(np.all(diff[~positive] <= protocol['gradient_zero_se_abs_max']))),
        mc_precision=rms(se) <= se_limit,
        training_reference_precision=rms(reference_se) <= se_limit,
        absolute_or_relative_error=rms(g-mc) <= error_limit,
        higher_precision_stability=rms(g-higher) <= error_limit,
        meaningful_direction=(opposite <= protocol['meaningful_opposite_directions_max']))
    return dict(passed=all(checks.values()), checks=checks, critic_reference_rms=gap,
                reference_mc_rms=rms(g-mc), mc_se_rms=rms(se), gradient_z_rms=z_rms,
                training_reference_se_rms=rms(reference_se),
                gradient_z_max=z_max, higher_precision_drift_rms=rms(g-higher),
                error_limit=error_limit, mc_se_limit=se_limit,
                meaningful_opposite_directions=opposite)


def make_table(task, actor, base, protocol, device, higher=False, sampling_seed=216):
    if protocol.get('reference_estimator') == 'iid-monte-carlo-score':
        return MonteCarloReference(task, actor, base,
                   protocol['audit_reference_mc_episodes'] if higher else protocol['reference_mc_episodes'],
                   sampling_seed, device)
    return AcceleratedPolicyTable(task, actor, base,
             protocol['audit_higher_grid'] if higher else protocol['reference_grid'],
             protocol['audit_higher_policy_samples'] if higher else protocol['reference_policy_samples'],
             protocol['audit_higher_transition_samples'] if higher else protocol['reference_transition_samples'],
             protocol['reference_gradient_samples'], device=device)


def audit_snapshot(task, actor, base, critic, train, protocol, seed, N, method, output,
                   device, primary=None):
    c = task.c
    folder = output/f'audit-{method}-N{N}'
    folder.mkdir(exist_ok=True)
    report_path = folder/'summary.json'
    actor_hash = M.model_hash(actor)
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if existing['actor_hash'] != actor_hash:
            raise RuntimeError('Audit snapshot hash changed')
        return existing, primary
    start = time.monotonic()
    geometry = json.loads((ROOT/'configs/reference_audit_points.json').read_text())['points']
    geometry = [p for p in geometry if p['id'] in protocol['audit_geometry_ids']]
    ids = [p['id'] for p in geometry]
    k = [p['k'] for p in geometry]
    pos = [[p['p_x'], p['p_y']] for p in geometry]
    action = [[p['a_x'], p['a_y']] for p in geometry]
    rng = M.core.generator(M.seed_for(seed, 820+N))
    count = protocol['audit_endpoint_count']
    selected = torch.randint(len(train['k']), (count,), generator=rng)
    ek, ep = train['k'][selected], train['y'][selected]
    z = torch.randn((count, 2), generator=rng)
    xi = torch.randn((count, c.flow_steps, 2), generator=rng)
    paths, _ = M.core.rollout(actor, base, c, ek, ep, z, xi)
    k.extend(ek.tolist()); pos.extend(ep.tolist()); action.extend(paths[:, -1].tolist())
    ids.extend([f'endpoint-{j:02d}' for j in range(count)])
    k, pos, action = np.asarray(k), np.asarray(pos), np.asarray(action)
    write_json(folder/'points.json', dict(ids=ids, k=k.tolist(), p=pos.tolist(), a=action.tolist(),
                                         actor_hash=actor_hash, state_indices=selected.tolist()))
    if primary is None:
        primary = make_table(task, actor, base, protocol, device,
                             sampling_seed=M.seed_for(seed, 700000+N*1000))
    primary.assert_current(actor, base)
    q, g = primary.q_gradient(k, pos, action)
    reference_se = getattr(primary, 'gradient_se', None)
    high = make_table(task, actor, base, protocol, device, higher=True,
                      sampling_seed=M.seed_for(seed, 900000+N*1000))
    qh, gh = high.q_gradient(k, pos, action)
    with torch.enable_grad():
        aa = torch.tensor(action, dtype=torch.float32, requires_grad=True)
        cq = critic(torch.tensor(k), torch.tensor(pos, dtype=torch.float32), aa).mean(-1)
        cg = torch.autograd.grad(cq.sum(), aa)[0].detach().numpy()
    policy = copy.deepcopy(actor).to(device).eval()
    mc_rows, means, ses, qmeans, qses = [], [], [], [], []
    for i, point_id in enumerate(ids):
        split = []
        for rep in range(2):
            mc_seed = M.seed_for(seed, 10000+N*100+rep*100000+i)
            item = independent_mc(task, policy, base, int(k[i]), pos[i], action[i],
                                  protocol['audit_mc_episodes_per_split'], mc_seed, device)
            split.append(item)
            mc_rows.append(dict(point_id=point_id, split=rep, mc_seed=mc_seed, **item))
        means.append((np.asarray(split[0]['gradient'])+split[1]['gradient'])/2)
        ses.append(np.hypot(split[0]['gradient_se'], split[1]['gradient_se'])/2)
        qmeans.append((split[0]['q']+split[1]['q'])/2)
        qses.append(np.hypot(split[0]['q_se'], split[1]['q_se'])/2)
        write_json(folder/'mc-progress.json', dict(completed=i+1, total=len(ids), rows=mc_rows))
    mc, se = np.asarray(means), np.asarray(ses)
    qm, qs = np.asarray(qmeans), np.asarray(qses)
    result = gradient_gate(g, gh, mc, se, cg, protocol, reference_se)
    qvalid = qs >= 1e-8
    qz = np.abs(q[qvalid]-qm[qvalid])/qs[qvalid]
    result.update(seed=seed, N=N, method=method, actor_hash=actor_hash,
                  points=len(ids), endpoint_points=count, seconds=time.monotonic()-start,
                  q_mc_abs_rms=rms(q-qm), q_mc_z_rms=rms(qz) if len(qz) else 0.,
                  q_mc_z_max=float(np.max(qz)) if len(qz) else 0.,
                  q_value_gate=False, primary_table_seconds=primary.seconds,
                  higher_table_seconds=high.seconds,
                  estimator=protocol.get('reference_estimator', 'policy-table'))
    rows = []
    for i, point_id in enumerate(ids):
        rows.append(dict(point_id=point_id, k=int(k[i]),
                         q_table=float(q[i]), q_mc=float(qm[i]), q_mc_se=float(qs[i]),
                         reference_x=g[i, 0], reference_y=g[i, 1],
                         higher_x=gh[i, 0], higher_y=gh[i, 1],
                         mc_x=mc[i, 0], mc_y=mc[i, 1], se_x=se[i, 0], se_y=se[i, 1],
                         critic_x=cg[i, 0], critic_y=cg[i, 1]))
    pd.DataFrame(rows).to_csv(folder/'gradients.csv', index=False)
    write_json(report_path, result)
    print('AUDIT', seed, method, N, json.dumps(result), flush=True)
    return result, primary


def evaluate_saved(task, actor, base, bank, output, method, N, final=False):
    stats, paths, status = M.evaluate(task, actor, base, bank, trajectories=True)
    lengths = np.isfinite(paths[:, :, 0]).sum(0)-1
    signs = (status == 'success').astype(float)-(status == 'collision').astype(float)
    returns = signs*task.c.gamma**(lengths-1)
    if not np.isclose(returns.mean(), stats['discounted_return'], atol=1e-12):
        raise RuntimeError('Per-episode return reconstruction mismatch')
    prefix = 'final' if final else 'curve'
    np.savez_compressed(output/f'{prefix}-{method}-N{N}.npz', paths=paths, status=status,
                        lengths=lengths, returns=returns)
    return stats


def checkpoint(state, base, N, output, method, config_hash, keep=False):
    saved = dict(N=N, method=method, config_hash=config_hash,
                 actor=state['actor'].state_dict(), critic=state['critic'].state_dict(),
                 target=state['target'].state_dict(), actor_optimizer=state['aopt'].state_dict(),
                 critic_optimizer=state['copt'].state_dict(), rng=state['rng'].get_state(),
                 base_hash=M.model_hash(base))
    temporary = output/f'{method}-latest.tmp'
    torch.save(saved, temporary)
    temporary.replace(output/f'{method}-latest.pt')
    if keep:
        torch.save(saved, output/f'{method}-N{N}.pt')


def state_from_initial(base, critic, target, c, seed, output, method, config_hash):
    state = dict(actor=copy.deepcopy(base), critic=copy.deepcopy(critic), target=copy.deepcopy(target),
                 rng=M.core.generator(M.seed_for(seed, 701)))
    for p in state['actor'].parameters():
        p.requires_grad_(True)
    state['aopt'] = torch.optim.Adam(state['actor'].parameters(), lr=c.actor_lr)
    state['copt'] = torch.optim.Adam(state['critic'].parameters(), lr=c.critic_lr)
    path = output/f'{method}-latest.pt'
    N = 0
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['config_hash'] != config_hash or saved['base_hash'] != M.model_hash(base):
            raise RuntimeError('Resume provenance changed')
        for key in ('actor', 'critic', 'target'):
            state[key].load_state_dict(saved[key])
        state['aopt'].load_state_dict(saved['actor_optimizer'])
        state['copt'].load_state_dict(saved['critic_optimizer'])
        state['rng'].set_state(saved['rng'])
        N = saved['N']
    return state, N


def run_seed(c, protocol, seed, output, device, config_hash):
    output.mkdir(exist_ok=True)
    task = M.PointObstacle(c)
    init = output/'shared-initialization.pt'
    initialization_source = protocol.get('initialization_source_run')
    if not init.exists() and initialization_source:
        import shutil
        source = ROOT/initialization_source/f'seed-{seed}'
        if (source/'shared-initialization.pt').exists():
            prior = json.loads((source.parent/'protocol.json').read_text())['config']
            for key, value in asdict(c).items():
                if not key.startswith('reference_') and json.loads(json.dumps(value)) != prior[key]:
                    raise RuntimeError(f'Initialization task/config changed: {key}')
            names = ('shared-initialization.pt', 'train-transitions.npz', 'valid-transitions.npz',
                     'base.pt', 'base-log.csv', 'pretrain-critic.csv', 'data-summary.json')
            for name in names:
                shutil.copy2(source/name, output/name)
            write_json(output/'initialization-source.json', dict(source=str(source),
                         hashes={name: digest(source/name) for name in names}))
    if init.exists():
        archive = np.load(output/'train-transitions.npz')
        train = M.as_torch({key: archive[key] for key in archive.files})
        saved = torch.load(init, map_location='cpu', weights_only=True)
        base = M.core.ConditionalVelocity(c); base.load_state_dict(saved['base'])
        for p in base.parameters():
            p.requires_grad_(False)
        critic = M.core.Critic(c); critic.load_state_dict(saved['critic'])
        target = M.core.Critic(c); target.load_state_dict(saved['target'])
    else:
        raw, summary = M.collect_episodes(task, c.train_episodes, M.seed_for(seed, 1))
        valid, vsummary = M.collect_episodes(task, c.valid_episodes, M.seed_for(seed, 2))
        np.savez_compressed(output/'train-transitions.npz', **raw)
        np.savez_compressed(output/'valid-transitions.npz', **valid)
        write_json(output/'data-summary.json', dict(train=summary, valid=vsummary))
        train = M.as_torch(raw)
        base, critic, target = M.pretrain(task, train, c, seed, output)
    curve_bank = M.noise_bank(c, c.curve_episodes, M.seed_for(seed, 501))
    final_bank = M.noise_bank(c, c.final_episodes, M.seed_for(seed, 502))
    states, completed = {}, {}
    for method in ('learned', 'reference'):
        states[method], completed[method] = state_from_initial(base, critic, target, c, seed,
                                                              output, method, config_hash)
    curves_file, logs_file = output/'curves.csv', output/'train.csv'
    curves = pd.read_csv(curves_file).to_dict('records') if curves_file.exists() else []
    logs = pd.read_csv(logs_file).to_dict('records') if logs_file.exists() else []
    for N in range(min(completed.values()), c.actor_updates+1):
        for method, st in states.items():
            if N < completed[method]:
                continue
            actor, cr, ta, rng = st['actor'], st['critic'], st['target'], st['rng']
            table = None
            if N in c.curve_updates:
                checkpoint(st, base, N, output, method, config_hash, keep=True)
                if not any(row['method'] == method and row['N'] == N for row in curves):
                    stats = evaluate_saved(task, actor, base, curve_bank, output, method, N)
                    curves.append(dict(seed=seed, method=method, N=N, **stats))
                    pd.DataFrame(curves).to_csv(curves_file, index=False)
                    print('CURVE', seed, method, N, json.dumps(stats), flush=True)
            if N in protocol['audit_updates']:
                # Same initial Actor: one audit remains valid for either identical initial branch.
                source = output/'audit-learned-N0/summary.json'
                if N == 0 and method == 'reference' and source.exists():
                    audit = json.loads(source.read_text())
                    if audit['actor_hash'] != M.model_hash(actor):
                        raise RuntimeError('Initial policies differ')
                    write_json(output/'audit-reference-N0-shared.json', dict(
                        source='audit-learned-N0/summary.json', actor_hash=audit['actor_hash'],
                        passed=audit['passed']))
                else:
                    audit, table = audit_snapshot(task, actor, base, cr, train, protocol,
                                                  seed, N, method, output, device)
                if not audit['passed']:
                    checkpoint(st, base, N, output, method, config_hash, keep=True)
                    write_json(output/'paused-gradient-audit.json', dict(seed=seed, N=N,
                                method=method, audit=audit))
                    raise RuntimeError(f'Gradient audit failed: seed={seed} {method} N={N}')
            if N == c.actor_updates:
                continue
            start = time.monotonic()
            indices = torch.randint(len(train['k']), (c.critic_batch,), generator=rng)
            td_loss, qmean = M.td_step(cr, ta, st['copt'], train, indices, rng, c, actor, base)
            indices = torch.randint(len(train['k']), (c.actor_batch,), generator=rng)
            k, pos = train['k'][indices], train['y'][indices]
            z = torch.randn((len(indices), 2), generator=rng)
            flow = torch.randn((len(indices), c.flow_steps, 2), generator=rng)
            table_seconds = 0.
            if method == 'reference':
                # Training MC uses a separate stream from all snapshot audits.
                if table is None or protocol.get('reference_estimator') == 'iid-monte-carlo-score':
                    table = make_table(task, actor, base, protocol, device,
                                       sampling_seed=M.seed_for(seed, 500000+N*1000))
                actor_log = M.reference_actor_step(actor, st['aopt'], base, c, k, pos, z, flow, table)
                table_seconds = table.seconds
                if getattr(table, 'gradient_se', None) is not None:
                    actor_log['reference_gradient_se_rms'] = rms(table.gradient_se)
            else:
                actor_log = M.core.actor_step(actor, st['aopt'], base, cr, task, c, k, pos, z,
                                              flow, 'learned')
            row = dict(seed=seed, method=method, N=N+1, td_loss=td_loss, q_mean=qmean,
                       update_seconds=time.monotonic()-start, reference_seconds=table_seconds,
                       **actor_log)
            logs = [r for r in logs if not (r['method'] == method and r['N'] == N+1)]
            logs.append(row)
            pd.DataFrame(logs).to_csv(logs_file, index=False)
            checkpoint(st, base, N+1, output, method, config_hash)
            write_json(output/'progress.json', dict(seed=seed, method=method, completed_update=N+1,
                        seconds=row['update_seconds'], updated_at=time.time()))
            print('UPDATE', seed, method, N+1, 'seconds', round(row['update_seconds'], 3), flush=True)
    final = []
    for method, st in states.items():
        stats = evaluate_saved(task, st['actor'], base, final_bank, output, method, c.actor_updates, final=True)
        final.append(dict(seed=seed, method=method, N=c.actor_updates, **stats))
    pd.DataFrame(final).to_csv(output/'final-independent.csv', index=False)
    write_json(output/'complete.json', dict(seed=seed, N=c.actor_updates, final=final,
                                          completed_at=time.time()))
    print('SEED COMPLETE', seed, json.dumps(final), flush=True)


def run(output, protocol_path, device):
    protocol = json.loads(protocol_path.read_text())
    c, _ = M.load_config()
    c = replace(c, actor_updates=protocol['actor_updates'],
                 curve_updates=tuple(protocol['curve_updates']),
                 curve_episodes=protocol['curve_episodes'], final_episodes=protocol['final_episodes'],
                 reference_grid=protocol['reference_grid'],
                 reference_policy_samples=protocol['reference_policy_samples'],
                 reference_transition_samples=protocol['reference_transition_samples'],
                 reference_gradient_samples=protocol['reference_gradient_samples'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT/'scripts/point_reference_accelerated.py',
               ROOT/'notebooks/02_qam_point_obstacle.py', M.CORE_PATH,
               ROOT/'configs/qam_point_obstacle.json', ROOT/'configs/reference_audit_points.json']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    payload = dict(protocol=protocol, config=asdict(c), source_hashes=hashes)
    config_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    prov = output/'protocol.json'
    if prov.exists():
        if json.loads(prov.read_text())['config_hash'] != config_hash:
            raise RuntimeError('Frozen protocol or source changed; use a new run directory')
    else:
        write_json(prov, dict(**payload, config_hash=config_hash, frozen_at=time.time(),
                   python=sys.version, torch=torch.__version__, numpy=np.__version__,
                   platform=platform.platform(), device=str(device),
                   gpu=torch.cuda.get_device_name() if device.startswith('cuda') else None))
        archive = output/'source'
        archive.mkdir()
        for src in sources:
            target = archive/src.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(src.read_bytes())
    for seed in protocol['seeds']:
        sub = output/f'seed-{seed}'
        if (sub/'complete.json').exists():
            continue
        run_seed(c, protocol, seed, sub, device, config_hash)
    frames = [pd.read_csv(output/f'seed-{s}/curves.csv') for s in protocol['seeds']]
    pd.concat(frames).to_csv(output/'curves.csv', index=False)
    frames = [pd.read_csv(output/f'seed-{s}/final-independent.csv') for s in protocol['seeds']]
    pd.concat(frames).to_csv(output/'final-independent.csv', index=False)
    write_json(output/'complete.json', dict(seeds=protocol['seeds'], N=c.actor_updates,
                                          completed_at=time.time()))
    print('SUITE COMPLETE', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/point_long_training.json')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    try:
        run(args.output.resolve(), args.protocol.resolve(), args.device)
    except Exception as exc:
        import traceback
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output/'failure.json', dict(error=repr(exc), time=time.time(),
                                                    traceback=traceback.format_exc()))
        raise
    finally:
        if list(args.output.glob('seed-*/curves.csv')):
            from analyze_point_long_training import analyze
            analyze(args.output)
