"""Finite-budget Adjoint Matching with naturally learned terminal critics.

All learner-facing terminal derivatives come from a fitted scalar network.
The clean terminal cost/gradient is reserved for data generation, evaluation,
and the explicitly labelled oracle baseline. No gradient corruption is injected.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)


@dataclass
class Config:
    name: str = "single_obstacle"
    steps: int = 32
    updates: int = 500
    batch: int = 64
    eval_every: int = 50
    eval_paths: int = 1024
    sigma: float = 0.4
    start: tuple = (-2.0, 0.0)
    goal: tuple = (2.0, 0.6)
    base_velocity: tuple = (3.6, 0.4)
    base_arch: float = 0.0
    drift_matrix: tuple = ((0.0, 1.0), (0.0, 0.25))
    obstacles: tuple = ((0.0, 0.0, 0.45),)
    obstacle_weight: float = 22.0
    terminal_weight: float = 6.0
    outcome_std: float = 0.4
    success_radius: float = 0.65
    critic_train: int = 1024
    critic_valid: int = 512
    critic_mc: int = 4
    critic_updates: int = 1800
    critic_width: int = 96
    critic_lr: float = 0.002
    actor_width: int = 64
    actor_lr: float = 0.002
    actor_lr_schedule: str = "constant"
    secant_h: float = 0.3
    consistency_fraction: float = 0.25

    @property
    def dt(self):
        return 1.0 / self.steps


def environment_manifest():
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads())


def rng(seed):
    return torch.Generator(device="cpu").manual_seed(int(seed))


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


class TerminalCritic(nn.Module):
    def __init__(self, width=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        # Fixed output scale shared across every method and seed.
        return 20.0 * self.net(x / 3.0).squeeze(-1)


class Controller(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        if not torch.is_tensor(t):
            t = torch.full_like(x[..., :1], float(t))
        return self.net(torch.cat([x / 3.0, 2.0 * t - 1.0], dim=-1))


def true_terminal_value(x, cfg):
    """Expected squared miss under an independent stochastic terminal outcome."""
    return cfg.terminal_weight * (((x - tensor(cfg.goal)) ** 2).sum(-1)
                                  + 2 * cfg.outcome_std ** 2)


def true_terminal_gradient(x, cfg):
    return 2 * cfg.terminal_weight * (x - tensor(cfg.goal))


def observed_terminal_cost(x, cfg, generator):
    """Monte Carlo task observations, not artificial errors added to a critic."""
    disturbance = cfg.outcome_std * torch.randn(
        len(x), cfg.critic_mc, 2, generator=generator)
    outcomes = x[:, None, :] + disturbance
    return cfg.terminal_weight * ((outcomes - tensor(cfg.goal)) ** 2).sum(-1).mean(-1)


def offline_positions(n, cfg, generator):
    """Mixed behavior-controller endpoints with broad coverage around the goal.

    Independent Gaussian endpoint targets stand in for diverse offline goals.
    Both training and validation distributions are fixed before method comparison.
    """
    x = tensor(cfg.goal) + torch.randn(n, 2, generator=generator) * tensor([1.35, 1.15])
    return x


def fit_critic(cfg, seed, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed + 100000)
    critic = TerminalCritic(cfg.critic_width)
    g = rng(seed + 200000)
    xtrain = offline_positions(cfg.critic_train, cfg, g)
    ytrain = observed_terminal_cost(xtrain, cfg, g)
    xvalid = offline_positions(cfg.critic_valid, cfg, g)
    yvalid = observed_terminal_cost(xvalid, cfg, g)
    optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    best = float("inf")
    best_state = None
    history = []
    start = time.monotonic()
    for k in range(1, cfg.critic_updates + 1):
        ix = torch.randint(len(xtrain), (128,), generator=g)
        loss = ((critic(xtrain[ix]) - ytrain[ix]) / 20.0).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if k % 50 == 0 or k == 1:
            with torch.no_grad():
                valid = (critic(xvalid) - yvalid).square().mean().item()
            history.append(dict(step=k, validation_noisy_mse=valid))
            if valid < best:
                best, best_state, best_step = valid, copy.deepcopy(critic.state_dict()), k
    critic.load_state_dict(best_state)
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    torch.save(dict(state_dict=best_state, config=asdict(cfg), seed=seed), directory / "critic.pt")
    np.savez_compressed(directory / "critic_data.npz", train_x=xtrain.numpy(), train_y=ytrain.numpy(),
                        valid_x=xvalid.numpy(), valid_y=yvalid.numpy())
    info = dict(best_step=best_step, validation_noisy_mse=best, history=history,
                seconds=time.monotonic()-start)
    (directory / "critic_fit.json").write_text(json.dumps(info, indent=2))
    return critic, info


def raw_gradient(critic, x):
    with torch.enable_grad():
        z = x.detach().requires_grad_(True)
        return torch.autograd.grad(critic(z).sum(), z)[0].detach()


@torch.no_grad()
def secant_gradient(critic, x, h):
    eye = h * torch.eye(2)
    plus = critic((x[:, None, :] + eye).reshape(-1, 2)).reshape(-1, 2)
    minus = critic((x[:, None, :] - eye).reshape(-1, 2)).reshape(-1, 2)
    return (plus - minus) / (2.0 * h)


def propagation_matrices(cfg):
    """B[k] maps terminal gradient error to the Euler AM target at step k.

    Discrete lean adjoint: a[k] = (I+dt A)^T a[k+1] + dt grad ell(x[k]).
    Control at step k uses -sigma a[k+1].
    """
    F = torch.eye(2) + cfg.dt * tensor(cfg.drift_matrix)
    B = torch.stack([cfg.sigma * torch.linalg.matrix_power(F.T, cfg.steps-1-k)
                     for k in range(cfg.steps)])
    G = cfg.dt * torch.einsum("kji,kjl->il", B, B)
    return B, G


@torch.no_grad()
def project_box_metric(raw, lower, upper, G):
    """Exact 2-D box-constrained positive-definite quadratic projection."""
    candidates = [raw.clamp(lower, upper)]
    for dim in (0, 1):
        other = 1 - dim
        for edge in (lower[:, dim], upper[:, dim]):
            q = raw.clone()
            q[:, dim] = edge
            q[:, other] = (raw[:, other] - G[other, dim] / G[other, other]
                           * (edge - raw[:, dim])).clamp(lower[:, other], upper[:, other])
            candidates.append(q)
    c = torch.stack(candidates, dim=1)
    d = c - raw[:, None, :]
    score = torch.einsum("nci,ij,ncj->nc", d, G, d)
    return c[torch.arange(len(raw)), score.argmin(1)]


def terminal_signal(method, critic, x, cfg, G):
    if method == "oracle":
        return true_terminal_gradient(x, cfg)
    raw = raw_gradient(critic, x)
    if method in ("am", "am_large_batch"):
        return raw
    if method == "clip":
        return raw / (raw.norm(dim=-1, keepdim=True) / 8.0).clamp_min(1.0)
    fine = secant_gradient(critic, x, cfg.secant_h)
    if method == "secant":
        return fine
    if method in ("project", "project_euclidean"):
        coarse = secant_gradient(critic, x, 2.0 * cfg.secant_h)
        # Empirical multiscale consistency interval; no claimed calibration.
        # The conditional projection theorem is audited separately.
        radius = (fine - coarse).abs() + cfg.consistency_fraction * (1.0 + fine.abs())
        metric = G if method == "project" else torch.eye(2)
        return project_box_metric(raw, fine-radius, fine+radius, metric)
    raise ValueError(method)


def drift(x, t, cfg):
    mean_path = tensor(cfg.start) + t * tensor(cfg.base_velocity)
    mean_path = mean_path + tensor([0., cfg.base_arch * math.sin(math.pi*t)])
    velocity = tensor(cfg.base_velocity) + tensor([0., cfg.base_arch*math.pi*math.cos(math.pi*t)])
    return velocity + (x - mean_path) @ tensor(cfg.drift_matrix).T


def running_cost_and_gradient(x, cfg):
    cost = torch.zeros(x.shape[:-1])
    grad = torch.zeros_like(x)
    for cx, cy, radius in cfg.obstacles:
        d = x - tensor([cx, cy])
        # Smooth obstacle is part of the unchanged true task, never gradient noise.
        scale = radius * 0.95
        bump = cfg.obstacle_weight * torch.exp(-d.square().sum(-1) / (2 * scale**2))
        cost = cost + bump
        grad = grad - bump[..., None] * d / scale**2
    return cost, grad


@torch.no_grad()
def rollout(actor, cfg, n, generator, steps=None):
    steps = cfg.steps if steps is None else steps
    dt = 1.0 / steps
    xs, us = [tensor(cfg.start).expand(n, 2).clone()], []
    for k in range(steps):
        u = actor(xs[-1], k * dt)
        noise = torch.randn(n, 2, generator=generator)
        xs.append(xs[-1] + dt*(drift(xs[-1], k*dt, cfg) + cfg.sigma*u)
                  + cfg.sigma * math.sqrt(dt) * noise)
        us.append(u)
    return torch.stack(xs, 1), torch.stack(us, 1)


@torch.no_grad()
def lean_targets(paths, terminal, cfg):
    F = torch.eye(2) + cfg.dt * tensor(cfg.drift_matrix)
    running_grad = running_cost_and_gradient(paths[:, :-1], cfg)[1]
    a = terminal.clone()
    targets = []
    for k in range(cfg.steps-1, -1, -1):
        targets.append(-cfg.sigma*a)
        a = a @ F + cfg.dt * running_grad[:, k]
    return torch.stack(targets[::-1], 1)


@torch.no_grad()
def evaluate(actor, cfg, seed, n=None, steps=None, keep_paths=False):
    n = cfg.eval_paths if n is None else n
    paths, controls = rollout(actor, cfg, n, rng(seed), steps)
    dt = 1.0 / controls.shape[1]
    energy = 0.5 * controls.square().sum(-1).sum(-1) * dt
    running = running_cost_and_gradient(paths[:, :-1], cfg)[0].sum(-1) * dt
    terminal = true_terminal_value(paths[:, -1], cfg)
    collided = torch.zeros(n, dtype=torch.bool)
    for cx, cy, r in cfg.obstacles:
        # Segment-distance collision, not just visits to grid points.
        a, b = paths[:, :-1]-tensor([cx, cy]), paths[:, 1:]-tensor([cx, cy])
        direction = b-a
        z = (-(a*direction).sum(-1) / direction.square().sum(-1).clamp_min(1e-12)).clamp(0, 1)
        closest = a + z[..., None] * direction
        collided |= (closest.square().sum(-1) < r*r).any(-1)
    distance = (paths[:, -1]-tensor(cfg.goal)).norm(dim=-1)
    success = (~collided) & (distance < cfg.success_radius)
    total = energy+running+terminal
    result = dict(cost=total.mean().item(), cost_mc_se=total.std().item()/math.sqrt(n),
                  terminal=terminal.mean().item(), energy=energy.mean().item(),
                  running=running.mean().item(), success=success.float().mean().item(),
                  collision=collided.float().mean().item(), distance=distance.mean().item())
    if keep_paths:
        return result, dict(paths=paths.numpy(), controls=controls.numpy(), per_path_cost=total.numpy(),
                            success=success.numpy(), collision=collided.numpy())
    return result


def train_controller(method, critic, cfg, seed, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed + 300000)
    actor = Controller(cfg.actor_width)
    optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    g = rng(seed + 400000)
    B, G = propagation_matrices(cfg)
    records = []
    start = time.monotonic()
    batch = cfg.batch * (5 if method == "am_large_batch" else 1)
    times = torch.arange(cfg.steps).float()[None, :, None].expand(batch, -1, -1) * cfg.dt
    for k in range(cfg.updates + 1):
        if k % cfg.eval_every == 0 or k == cfg.updates:
            metrics = evaluate(actor, cfg, seed + 500000)
            records.append(dict(step=k, seconds=time.monotonic()-start, **metrics))
            if k in (0, cfg.updates//2, cfg.updates):
                torch.save(actor.state_dict(), directory / f"actor_step_{k}.pt")
        if k == cfg.updates:
            break
        paths, _ = rollout(actor, cfg, batch, g)
        z = terminal_signal(method, critic, paths[:, -1], cfg, G)
        target = lean_targets(paths, z, cfg)
        pred = actor(paths[:, :-1].reshape(-1, 2), times.reshape(-1, 1)).reshape(batch, cfg.steps, 2)
        loss = 0.5*(pred-target).square().sum(-1).mean()
        if cfg.actor_lr_schedule == "cosine":
            for group in optimizer.param_groups:
                group['lr']=cfg.actor_lr*(.15+.85*.5*(1+math.cos(math.pi*k/cfg.updates)))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        optimizer.step()
    seconds = time.monotonic()-start
    final, arrays = evaluate(actor, cfg, seed + 600000, n=4096, keep_paths=True)
    fine = evaluate(actor, cfg, seed + 700000, n=2048, steps=2*cfg.steps)
    torch.save(actor.state_dict(), directory / "actor.pt")
    np.savez_compressed(directory / "evaluation.npz", **arrays)
    out = dict(method=method, seed=seed, seconds=seconds, history=records, final=final,
               fine_grid=fine, training_paths=batch*cfg.updates,
               critic_forward_points_per_terminal={"am":1,"am_large_batch":1,"oracle":0,
                                                   "clip":1,"secant":5,"project":9,
                                                   "project_euclidean":9}[method])
    (directory / "result.json").write_text(json.dumps(out, indent=2))
    print(f"{cfg.name} seed={seed} {method}: cost={final['cost']:.3f} "
          f"success={final['success']:.3f} collision={final['collision']:.3f} ({seconds:.1f}s)", flush=True)
    return actor, out


def critic_diagnostics(critic, cfg, seed, directory):
    x = offline_positions(4096, cfg, rng(seed + 800000))
    B, G = propagation_matrices(cfg)
    truth = true_terminal_gradient(x, cfg)
    with torch.no_grad():
        actual = true_terminal_value(x, cfg)
        error = critic(x)-actual
    metrics = dict(value_rmse=error.square().mean().sqrt().item(),
                   value_std=actual.std().item(), value_q95_error=error.abs().quantile(.95).item())
    rows = []
    for method in ("am", "secant", "project", "project_euclidean", "clip"):
        z = terminal_signal(method, critic, x, cfg, G)
        err = z-truth
        rows.append(dict(method=method, gradient_rmse=err.square().sum(-1).mean().sqrt().item(),
                         propagated_mse=torch.einsum("ni,ij,nj->n", err, G, err).mean().item()))
    fine, coarse = secant_gradient(critic, x, cfg.secant_h), secant_gradient(critic,x,2*cfg.secant_h)
    radius=(fine-coarse).abs()+cfg.consistency_fraction*(1+fine.abs())
    metrics["empirical_interval_coverage"] = ((truth-fine).abs() <= radius).all(-1).float().mean().item()
    metrics["gradient_reference_rms"] = truth.square().sum(-1).mean().sqrt().item()
    out = dict(**metrics, gradient_methods=rows)
    Path(directory, "critic_diagnostics.json").write_text(json.dumps(out, indent=2))
    return out


def run_suite(cfg, seeds, methods, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(config=asdict(cfg), seeds=list(seeds), methods=list(methods), environment=environment_manifest(),
                  core_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest = json.dumps(config, sort_keys=True, indent=2)
    config["configuration_sha256"] = hashlib.sha256(manifest.encode()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(config, indent=2))
    results = []
    for seed in seeds:
        folder=output/f"seed_{seed}"
        if (folder/"critic.pt").exists():
            saved=torch.load(folder/"critic.pt",weights_only=True)
            if saved["config"] != asdict(cfg):
                raise RuntimeError("Existing critic uses a different configuration")
            critic=TerminalCritic(cfg.critic_width)
            critic.load_state_dict(saved["state_dict"])
            for p in critic.parameters(): p.requires_grad_(False)
            critic.eval()
        else:
            critic, _ = fit_critic(cfg, seed, folder)
        diagnostics=critic_diagnostics(critic,cfg,seed,folder)
        print(f"Critic seed={seed}: value RMSE={diagnostics['value_rmse']:.3f}, "
              f"grad RMSE={diagnostics['gradient_methods'][0]['gradient_rmse']:.3f}, "
              f"coverage={diagnostics['empirical_interval_coverage']:.3f}",flush=True)
        for method in methods:
            result_path=folder/method/"result.json"
            if result_path.exists():
                result=json.loads(result_path.read_text())
            else:
                _, result=train_controller(method,critic,cfg,seed,folder/method)
            results.append(result)
    (output/"results.json").write_text(json.dumps(results,indent=2))
    return results


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="development_v1")
    parser.add_argument("--seeds",type=int,nargs="+",default=[101,102])
    parser.add_argument("--methods",nargs="+",default=["am","secant","project","oracle"])
    parser.add_argument("--updates",type=int,default=500)
    parser.add_argument("--critic-updates",type=int,default=1800)
    parser.add_argument("--h",type=float,default=.3)
    args=parser.parse_args()
    cfg=Config(updates=args.updates,critic_updates=args.critic_updates,secant_h=args.h)
    run_suite(cfg,args.seeds,args.methods,Path(__file__).resolve().parent/args.output)
