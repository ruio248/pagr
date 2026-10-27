"""Finite-displacement value guard around unchanged lean AM + Adam proposals.

Not a new HJBI PDE. One shared scalar critic mixture evaluates the difference
between the candidate and incumbent policies. No true terminal values or
gradients are accessed in proposal() or guard(). The known running cost is
unchanged. All additional simulations and critic queries are counted.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import copy
import json
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'am_terminal_consistent_lean'))
import lean_am as la


@dataclass
class Options:
    updates: int = 1000
    batch: int = 64
    block: int = 20
    guard_paths: int = 320
    alphas: tuple = (0., .25, .5, 1.)
    rho: float = .3
    validation_slack: float = .1
    lr: float = .002
    eval_every: int = 100
    final_paths: int = 4096


def vector(actor):
    return torch.nn.utils.parameters_to_vector(actor.parameters()).detach().clone()


@torch.no_grad()
def assign(actor, v):
    torch.nn.utils.vector_to_parameters(v.clone(), actor.parameters())


def proposal(actor, optimizer, paths, critics, task, cfg):
    """Exactly the previously used lean AM + Adam update, including clip 5."""
    return la.update(actor, optimizer, paths, critics, task, cfg,
                     la.Options(method='am_adam'), {}, None, 0)


@torch.no_grad()
def model_costs(actor, task, cfg, critics, seed, n):
    paths, u = task.rollout(actor, cfg, n, task.rng(seed))
    known = .5 * cfg.dt * u.square().sum((-1, -2))
    if task is la.rh.navigation:
        known += cfg.dt * task.running_cost_and_gradient(paths[:, :-1], cfg)[0].sum(-1)
    return torch.stack([known + critic(paths[:, -1]) for critic in critics], -1).double().numpy()


def worst_difference(member_difference, adversary):
    """Maximize a LINEAR finite cost difference; no gradient minimax here."""
    d = np.asarray(member_difference, dtype=np.float64)
    if adversary.rho == 0 or np.ptp(d) < 1e-12:
        return float(d @ adversary.pi), adversary.pi.copy(), 0.
    scale = max(float(np.ptp(d)), 1e-10)
    objective = (d - d.mean()) / scale
    result = minimize(lambda w: -float(w @ objective), adversary.pi,
                      jac=lambda w: -objective, method='SLSQP',
                      bounds=[(1e-12, 1.)] * len(d), constraints=adversary.constraints(),
                      options=dict(ftol=1e-11, maxiter=300))
    if not result.success or adversary.violation(result.x) > 1e-7:
        raise RuntimeError(f'Value adversary failed: {result.message}')
    w = result.x
    # Feasible incumbent lower bound is a necessary, not sufficient, certificate.
    assert w @ d >= adversary.pi @ d - 1e-8
    return float(w @ d), w, adversary.violation(w)


def guard(actor, before, proposed, task, cfg, critics, seed, options, adversary, robust):
    costs = []
    for alpha in options.alphas:
        assign(actor, before + alpha * (proposed - before))
        costs.append(model_costs(actor, task, cfg, critics, seed, options.guard_paths))
    differences = [np.mean(cost - costs[0], axis=0) for cost in costs]
    scores, weights, violations = [], [], []
    for delta in differences:
        if robust:
            score, w, violation = worst_difference(delta, adversary)
        else:
            score, w, violation = float(delta.mean()), adversary.pi.copy(), 0.
        scores.append(score)
        weights.append(w.tolist())
        violations.append(violation)
    selected = int(np.argmin(scores))  # alpha=0 is always feasible, ties prefer 0.
    assert scores[selected] <= 1e-9
    alpha = options.alphas[selected]
    assign(actor, before + alpha * (proposed - before))
    return dict(alpha=alpha, scores=scores, member_differences=[d.tolist() for d in differences],
                weights=weights, max_constraint_violation=max(violations),
                proposed_step_norm=float((proposed-before).norm()))


def train(name, seed, method, folder=None, options=None):
    options = options or Options()
    assert method in ('am_adam', 'am_2x', 'nominal_guard', 'robust_guard')
    folder = Path(folder) if folder else ROOT/'results'/name/f'seed_{seed}'/method
    folder.mkdir(parents=True, exist_ok=True)
    task, cfg, critics, c, manifest = la.load_inputs(name, seed)
    torch.manual_seed(seed+300000)
    actor = task.Controller(cfg.actor_width)
    optimizer = torch.optim.Adam(actor.parameters(), lr=options.lr)
    generator = task.rng(seed+400000)
    adversary = la.Adversary(c, options.rho, options.validation_slack)
    history, checks, blocks = [], [], []
    batch = 2*options.batch if method == 'am_2x' else options.batch
    guarded = method.endswith('_guard')
    start = time.monotonic()
    for step in range(options.updates+1):
        if step % options.eval_every == 0 or step == options.updates:
            history.append(dict(step=step, **task.evaluate(actor, cfg, seed+500000, n=1024)))
        if step == options.updates:
            break
        if guarded and step % options.block == 0:
            before = vector(actor)
        paths, _ = task.rollout(actor, cfg, batch, generator)
        proposal(actor, optimizer, paths, critics, task, cfg)
        if guarded and (step+1) % options.block == 0:
            proposed = vector(actor)
            block = (step+1)//options.block
            info = guard(actor, before, proposed, task, cfg, critics,
                         seed+800000+block*10007, options, adversary, method == 'robust_guard')
            checks.append(dict(step=step+1, **info))
            blocks.append(dict(step=step+1, before=before, proposed=proposed, alpha=info['alpha']))
            # Adam moments advance on ALL proposal minibatches, even if rejected.
    final, arrays = task.evaluate(actor, cfg, seed+600000, n=options.final_paths, keep_paths=True)
    fine = task.evaluate(actor, cfg, seed+700000, n=2048, steps=2*cfg.steps)
    torch.save(actor.state_dict(), folder/'actor.pt')
    torch.save(blocks, folder/'blocks.pt')
    np.savez_compressed(folder/'evaluation.npz', **arrays)
    n_guard = options.updates//options.block * len(options.alphas)*options.guard_paths if guarded else 0
    result = dict(task=name, seed=seed, method=method, options=asdict(options), config=asdict(cfg),
                  final=final, fine_grid=fine, history=history, checks=checks,
                  seconds=time.monotonic()-start, input_manifest=manifest,
                  actor_training_rollouts=options.updates*batch,
                  extra_model_rollouts=n_guard, total_training_model_rollouts=options.updates*batch+n_guard,
                  extra_critic_value_queries=n_guard*len(critics),
                  additional_real_terminal_queries=0, true_gradient_training_calls=0,
                  adjoint='lean', optimizer='Adam',
                  optimizer_state_policy='Moments advance on all proposals; only parameters are interpolated.')
    (folder/'result.json').write_text(json.dumps(result, indent=2))
    print(name, seed, method, 'cost', final['cost'], 'seconds', result['seconds'], flush=True)
    return result


def freeze():
    protocol = dict(options=asdict(Options()), tasks=['navigation', 'arm'], seeds=[18001,18002,18003],
                    methods=['am_adam', 'am_2x', 'nominal_guard', 'robust_guard'],
                    primary='robust_guard vs native lean AM+Adam AND equal-model-rollout AM+Adam batch128',
                    secondary='robust_guard vs nominal_guard isolates critic-mixture uncertainty.',
                    selection='Fixed final checkpoint at 1000 updates; no best true-evaluation selection.',
                    gate='At least 1% mean reduction and 3/3 seed wins against both Adam baselines on either task. This is only a development gate, NOT statistical confirmation.',
                    budget='Guard: 64000 AM rollouts + 64000 model-only check rollouts. am_2x: 128000 AM rollouts. Wall-clock and derivative compute differ.',
                    reuse='Seeds/data are reused developmental data, NOT independent confirmation. No new tuning after seeing these results.',
                    boundaries=['All methods lean adjoint. Same unchanged Adam proposals.',
                                'Frozen original five critics, no new real outcome labels.',
                                'No engineered gradient corruption, altered dynamics, or objective.',
                                'Empirical finite-cost minimax safeguard is not a new HJBI PDE.',
                                'No uncertainty coverage or true-J descent guarantee is assumed.'],
                    sources={str(Path(__file__).resolve()):la.rh.rel.sha(__file__),
                             str(Path(la.__file__)):la.rh.rel.sha(la.__file__)})
    path=ROOT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(protocol))
    else:
        path.write_text(json.dumps(protocol, indent=2))
    return protocol


def audit():
    records={}
    for name in ('navigation','arm'):
        task,cfg,critics,c,_=la.load_inputs(name,18001)
        torch.manual_seed(9312)
        actor=task.Controller(cfg.actor_width)
        other=copy.deepcopy(actor)
        paths,_=task.rollout(actor,cfg,11,task.rng(788))
        saved={k:getattr(task,k) for k in ('true_terminal_value','true_terminal_gradient','observed_terminal_cost')}
        prediction,oldmatch=la.base.prediction,la.base.matching_gradients
        def forbidden(*a,**kw): raise AssertionError('Forbidden truth / full-adjoint access')
        def lean_prediction(*a,**kw):
            assert kw.get('spatial_jacobian') is False
            return prediction(*a,**kw)
        for k in saved:setattr(task,k,forbidden)
        la.base.prediction,la.base.matching_gradients=lean_prediction,forbidden
        try:
            before=vector(actor)
            proposal(actor,torch.optim.Adam(actor.parameters(),lr=.002),paths,critics,task,cfg)
            la.update(other,torch.optim.Adam(other.parameters(),lr=.002),paths,critics,task,cfg,
                      la.Options(method='am_adam'),{},None,0)
            assert torch.equal(vector(actor),vector(other))
            proposed=vector(actor)
            for robust in (False,True):
                info=guard(actor,before,proposed,task,cfg,critics,922,Options(guard_paths=16),la.Adversary(c,.3),robust)
                assert info['scores'][0]==0 and min(info['scores'])<=0
            records[name+'_exact_Adam_proposal_and_no_truth_no_full_adjoint']=True
        finally:
            for k,v in saved.items():setattr(task,k,v)
            la.base.prediction,la.base.matching_gradients=prediction,oldmatch
        # Same finite scalar function weights must evaluate BOTH policies.
        v=np.array([-.4,.2,.7,-.1,.3])
        a=la.Adversary(c,0)
        val,w,_=worst_difference(v,a)
        assert np.allclose(w,np.ones(5)/5) and abs(val-v.mean())<1e-10
        records[name+'_zero_radius_equals_nominal']=True
    result=dict(passed=True,checks=records)
    (ROOT/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--freeze',action='store_true');p.add_argument('--audit',action='store_true')
    p.add_argument('--task',choices=['navigation','arm'])
    a=p.parse_args()
    if a.audit:audit()
    if a.freeze:freeze()
    if a.task:
        protocol=freeze()
        assert json.loads((ROOT/'audit.json').read_text())['passed']
        for seed in protocol['seeds']:
            for method in protocol['methods']:
                folder=ROOT/'results'/a.task/f'seed_{seed}'/method
                if not (folder/'result.json').exists():train(a.task,seed,method,folder)
