"""Terminal-coupled robust Hamiltonian AM (HJBI-inspired, not a new PDE).

All trajectories, adjoints and critic gradients are stop-gradient quantities.
For M frozen scalar critics, A_m=(a_m-mean(a))/sqrt(M-1). An adversary chooses
||xi||_2 <= rho; its terminal error is a linear combination of critic-gradient
contrasts and the SAME xi propagates through every time on a trajectory.

L = mean[.5||u||^2 + u^T sigma^T mean(a)]
    + rho || mean_{batch,time}[ A^T (b + sigma u) ] ||_2.

The global version also shares xi across the minibatch (one scalar-critic
function contrast). Path and rectangular variants deliberately relax that set.
The b term must remain INSIDE the support function despite being fixed in u.
No drift adversary h, no extra scalar outcome queries, no analytic truth in loss.
This is a frozen-trajectory training surrogate, NOT a time-consistent HJBI
solution or a guarantee on true J. Noise Hessian terms are not robustified.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / 'am_propagation_reliability'
sys.path.insert(0, str(LEGACY))
import reliability as rel
# Import both before assigning rel.task; their base infrastructure is toy_core.
import arm_task as arm
import inertial_task as navigation


@dataclass
class Options:
    method: str = 'global'
    rho: float = .3
    updates: int = 1000
    batch: int = 64
    lr: float = .002
    eval_every: int = 100
    final_paths: int = 4096


def get_task(name):
    assert name in ('arm', 'navigation')
    return arm if name == 'arm' else navigation


def config_for(name):
    fname = 'arm_protocol.json' if name == 'arm' else 'formal_protocol.json'
    return get_task(name).Config(**json.loads((LEGACY / fname).read_text())['config'])


@torch.no_grad()
def adjoints(task, paths, terminal, cfg):
    """a at the right end of each Euler interval, exactly as legacy AM targets."""
    a = terminal.clone()
    states = []
    if task is arm:
        F = 1 - cfg.dt * cfg.stiffness
        for k in range(cfg.steps - 1, -1, -1):
            states.append(a)
            a = a * F
    else:
        _, F, _ = task.matrices(cfg)
        running = task.running_cost_and_gradient(paths[:, :-1], cfg)[1]
        for k in range(cfg.steps - 1, -1, -1):
            states.append(a)
            a = a @ F + cfg.dt * running[:, k]
    return torch.stack(states[::-1], dim=1)


def coefficients(task, paths, cfg):
    if task is arm:
        b = -cfg.stiffness * (paths[:, :-1] - task.tensor(cfg.start))
        sigma = cfg.sigma * torch.eye(paths.shape[-1], dtype=paths.dtype)
    else:
        b = torch.stack([task.drift(paths[:, k], k * cfg.dt, cfg) for k in range(cfg.steps)], 1)
        sigma = task.matrices(cfg)[2]
    return b, sigma


def support(score, method):
    """score has M x B x T shape; exact Euclidean-ball support functions."""
    if method in ('global', 'no_drift', 'norm_matched'):
        return torch.linalg.vector_norm(score.mean((1, 2)))
    if method == 'path':
        return torch.linalg.vector_norm(score.mean(2), dim=0).mean()
    if method == 'rectangular':
        return torch.linalg.vector_norm(score, dim=0).mean()
    raise ValueError(method)


def objective(control, member_adjoints, drift, sigma, rho, method):
    mean = member_adjoints.mean(0)
    nominal = (.5 * control.square().sum(-1) + (control * (mean @ sigma)).sum(-1)).mean()
    if rho == 0 or method in ('am', 'oracle'):
        return nominal, nominal, nominal.new_zeros(())
    contrasts = (member_adjoints - mean) / math.sqrt(len(member_adjoints) - 1)
    total_drift = control @ sigma.T
    if method != 'no_drift':
        total_drift = total_drift + drift
    score = (contrasts * total_drift.unsqueeze(0)).sum(-1)
    penalty = support(score, method)
    return nominal + rho * penalty, nominal, penalty


def actor_loss(actor, paths, critics, task, cfg, options):
    zs = ([task.true_terminal_gradient(paths[:, -1], cfg)] if options.method == 'oracle'
          else [task.raw_gradient(c, paths[:, -1]) for c in critics])
    member_adjoints = torch.stack([adjoints(task, paths, z, cfg) for z in zs])
    drift, sigma = coefficients(task, paths, cfg)
    n, steps = paths.shape[:2]
    tt = (torch.arange(cfg.steps) * cfg.dt)[None, :, None].expand(n, -1, -1)
    control = actor(paths[:, :-1].reshape(-1, paths.shape[-1]), tt.reshape(-1, 1))
    control = control.reshape(n, cfg.steps, -1)
    return objective(control, member_adjoints, drift, sigma, options.rho, options.method)


def update(actor, optimizer, paths, critics, task, cfg, options):
    loss, nominal, penalty = actor_loss(actor, paths, critics, task, cfg, options)
    if options.method == 'norm_matched':
        robust_grad = torch.autograd.grad(loss, actor.parameters(), retain_graph=True)
        plain_grad = torch.autograd.grad(nominal, actor.parameters())
        nr = torch.cat([g.flatten() for g in robust_grad]).norm()
        np_ = torch.cat([g.flatten() for g in plain_grad]).norm().clamp_min(1e-12)
        optimizer.zero_grad()
        for p, g in zip(actor.parameters(), plain_grad):
            p.grad = g * (nr / np_)
    else:
        optimizer.zero_grad()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.)
    optimizer.step()
    return dict(loss=loss.item(), nominal=nominal.item(), support=penalty.item(),
                pre_clip_gradient_norm=norm.item())


def train(name, cfg, seed, critics, options, folder):
    task = get_task(name)
    rel.task = task
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed + 300000)
    actor = task.Controller(cfg.actor_width)
    optimizer = torch.optim.Adam(actor.parameters(), lr=options.lr)
    generator = task.rng(seed + 400000)
    history, diagnostics = [], []
    started = time.monotonic()
    for step in range(options.updates + 1):
        if step % options.eval_every == 0 or step == options.updates:
            history.append(dict(step=step, **task.evaluate(actor, cfg, seed + 500000, n=1024)))
        if step in {0, options.updates // 2, options.updates}:
            torch.save(dict(actor=actor.state_dict(), optimizer=optimizer.state_dict(), step=step),
                       folder / f'checkpoint_{step}.pt')
        if step == options.updates:
            break
        paths, _ = task.rollout(actor, cfg, options.batch, generator)
        info = update(actor, optimizer, paths, critics, task, cfg, options)
        if step % 25 == 0:
            diagnostics.append(dict(step=step + 1, **info))
    final, arrays = task.evaluate(actor, cfg, seed + 600000, n=options.final_paths, keep_paths=True)
    fine = task.evaluate(actor, cfg, seed + 700000, n=2048, steps=2 * cfg.steps)
    torch.save(actor.state_dict(), folder / 'actor.pt')
    np.savez_compressed(folder / 'evaluation.npz', **arrays)
    row = dict(task=name, seed=seed, config=asdict(cfg), options=asdict(options),
               history=history, diagnostics=diagnostics, final=final, fine_grid=fine,
               seconds=time.monotonic() - started, additional_scalar_cost_queries=0,
               actor_training_rollouts=options.updates * options.batch,
               evaluation_only_truth=True)
    (folder / 'result.json').write_text(json.dumps(row, indent=2))
    print(name, seed, options.method, options.rho, 'cost', round(final['cost'], 5),
          'fine', round(fine['cost'], 5), 'seconds', round(row['seconds'], 1), flush=True)
    return row
