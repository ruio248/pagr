"""Accelerated implementation of the existing current-policy reference.

The environment, Sobol nodes, deployment ODE, interpolation and Gaussian score
identity match Notebook 2. Geometry/integration use float64; the unchanged
Actor uses float32. This module does not train a reference Critic.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('point_accelerated_source',
                                             ROOT/'notebooks/02_qam_point_obstacle.py')
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def transition_integral(p, a, table, xi, delta, sigma, gamma,
                        obstacle_x, obstacle_y, obstacle_radius,
                        goal_x, goal_y, goal_radius):
    """One Gaussian integration; termination masks match PointObstacle.outcome."""
    mean = p + delta*torch.tanh(a)
    pn = mean[:, None, :] + sigma*xi[None, :, :]
    direction = pn-p[:, None, :]
    center = p.new_tensor([obstacle_x, obstacle_y])
    fraction = (((center-p[:, None, :])*direction).sum(-1)
                /direction.square().sum(-1).clamp_min(1e-15)).clamp(0., 1.)
    closest = p[:, None, :]+fraction[..., None]*direction
    hit = (closest-center).square().sum(-1) <= obstacle_radius**2
    outside = ((pn < 0.) | (pn > 1.)).any(-1)
    bad = hit | outside
    goal = ((pn-p.new_tensor([goal_x, goal_y])).square().sum(-1)
            <= goal_radius**2)
    success = goal & ~bad
    # At the horizon V_T is identically zero, so timeout bootstrap is also zero.
    reward = success.to(p.dtype)-bad.to(p.dtype)
    coord = pn.clamp(0., 1.)*(table.shape[0]-1)
    ij = coord.floor().to(torch.long).clamp_max(table.shape[0]-2)
    d = coord-ij
    x, y = ij[..., 0], ij[..., 1]
    value = ((1-d[..., 0])*(1-d[..., 1])*table[x, y]
             + d[..., 0]*(1-d[..., 1])*table[x+1, y]
             + (1-d[..., 0])*d[..., 1]*table[x, y+1]
             + d[..., 0]*d[..., 1]*table[x+1, y+1])
    h = reward+gamma*(~(bad | success)).to(p.dtype)*value
    q = h.mean(-1)
    grad = delta*(1-torch.tanh(a).square())*(h[..., None]*xi).mean(1)/sigma
    return q, grad


@torch.no_grad()
def sample_actor_tensor(actor, c, k, p, z):
    x = z
    h = 1/c.flow_steps
    for j in range(c.flow_steps):
        x = x+h*actor(k, p, x, (j+.5)*h)
    return x


_COMPILED = None


class AcceleratedPolicyTable:
    def __init__(self, task, actor, base, grid, samples, transition_samples,
                 gradient_samples=8192, seed=216, device='cuda', chunk=8192,
                 compile_kernel=True):
        global _COMPILED
        self.task, self.c = task, task.c
        self.actor_hash, self.base_hash = M.model_hash(actor), M.model_hash(base)
        self.device = torch.device(device)
        self.grid, self.samples = grid, samples
        self.transition_samples, self.gradient_samples = transition_samples, gradient_samples
        self.chunk = chunk
        self.kernel = transition_integral
        if compile_kernel and self.device.type == 'cuda':
            if _COMPILED is None:
                _COMPILED = torch.compile(transition_integral, fullgraph=True, dynamic=False)
            self.kernel = _COMPILED
        c = self.c
        self.kernel_args = (c.delta, c.sigma_env, c.gamma, *c.obstacle_center,
                            c.obstacle_radius, *c.goal_center, c.goal_radius)
        self.quad = M.normal_quadrature(transition_samples)
        self.gradient_quad = M.normal_quadrature(gradient_samples, seed=1994)
        self.nodes = np.stack(np.meshgrid(np.linspace(0, 1, grid),
                             np.linspace(0, 1, grid), indexing='ij'), -1).reshape(-1, 2)
        self.values_tensor = torch.zeros((c.horizon+1, grid, grid), dtype=torch.float64,
                                         device=self.device)
        policy = copy.deepcopy(actor).to(self.device).eval()
        z, _ = M.policy_noise(c, samples, seed)
        nodes = torch.as_tensor(np.repeat(self.nodes, samples, axis=0),
                                dtype=torch.float64, device=self.device)
        z_full = torch.as_tensor(np.tile(z, (len(self.nodes), 1)), device=self.device)
        xi = torch.as_tensor(self.quad[0], device=self.device)
        self.action_calls = 0
        start = time.monotonic()
        with torch.no_grad():
            for k in range(c.horizon-1, -1, -1):
                kk = torch.full((len(nodes),), k, device=self.device, dtype=torch.long)
                action = sample_actor_tensor(policy, c, kk, nodes.float(), z_full).double()
                q, _ = self._integrate(nodes, action, self.values_tensor[k+1], xi)
                self.values_tensor[k] = q.reshape(len(self.nodes), samples).mean(-1).reshape(grid, grid)
                self.action_calls += len(nodes)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.seconds = time.monotonic()-start
        self.values = self.values_tensor.cpu().numpy()

    def _integrate(self, p, a, table, xi):
        qs, gs = [], []
        for begin in range(0, len(p), self.chunk):
            q, g = self.kernel(p[begin:begin+self.chunk], a[begin:begin+self.chunk],
                               table, xi, *self.kernel_args)
            qs.append(q)
            gs.append(g)
        return torch.cat(qs), torch.cat(gs)

    def assert_current(self, actor, base):
        if M.model_hash(actor) != self.actor_hash or M.model_hash(base) != self.base_hash:
            raise ValueError('stale policy table')

    @torch.no_grad()
    def q_gradient(self, k, p, a):
        k = np.broadcast_to(np.asarray(k, dtype=int), (len(p),))
        q = np.empty(len(p)); g = np.empty((len(p), 2))
        # Queries contain only tens of endpoints. Keep the original NumPy score
        # integral verbatim; acceleration is needed for the full-grid recursion.
        for j in np.unique(k):
            mask = k == j
            q[mask], g[mask] = M.integral(self.task, int(j), np.asarray(p)[mask],
                                         np.asarray(a)[mask], self.values[j+1],
                                         self.gradient_quad, True)
        return q, g


def benchmark(output, device='cuda', grid=81, policy_samples=32, transition_samples=512):
    import json
    from dataclasses import replace
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    c, _ = M.load_config()
    sub = ROOT/'runs/02-qam-point-obstacle-smoke-e6eff27267/seed-59000'
    actor = M.load_actor(sub/'learned-N0.pt', c)
    base = M.load_base(sub/'shared-initialization.pt', c)
    task = M.PointObstacle(c)
    points = json.loads((ROOT/'configs/reference_audit_points.json').read_text())['points']
    k = np.array([p['k'] for p in points])
    p = np.array([[p['p_x'], p['p_y']] for p in points])
    a = np.array([[p['a_x'], p['a_y']] for p in points])
    # Exact cross-backend check on a short chain, including policy and geometry.
    short = replace(c, horizon=3)
    small_cpu = M.PolicyTable(M.PointObstacle(short), actor, base, 17, 4, 32, 2048)
    small_gpu = AcceleratedPolicyTable(M.PointObstacle(short), actor, base, 17, 4, 32,
                                       2048, device=device)
    small_table_max = float(np.max(np.abs(small_cpu.values-small_gpu.values)))
    # Actor.c is the original task config on both backends; both receive the same k.
    qcpu, gcpu = small_cpu.q_gradient(k % 3, p, a)
    qgpu, ggpu = small_gpu.q_gradient(k % 3, p, a)
    checks = dict(table_max=small_table_max, q_max=float(np.max(np.abs(qcpu-qgpu))),
                  gradient_max=float(np.max(np.abs(gcpu-ggpu))))
    print('Backend equivalence', checks, flush=True)
    if max(checks.values()) > 2e-4:
        raise RuntimeError(f'CPU/CUDA equivalence failed: {checks}')
    for repeat in range(2):
        table = AcceleratedPolicyTable(task, actor, base, grid, policy_samples,
                                        transition_samples, 8192, device=device)
        start = time.monotonic()
        q, g = table.q_gradient(k, p, a)
        query_seconds = time.monotonic()-start
        result = dict(repeat=repeat, device=device, grid=grid, samples=policy_samples,
                      transition_samples=transition_samples, seconds=table.seconds,
                      query_seconds=query_seconds, equivalence=checks, q=q.tolist(),
                      gradient=g.tolist())
        (output/f'benchmark-{repeat}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ('q', 'gradient')}), flush=True)
        np.savez_compressed(output/f'table-{repeat}.npz', values=table.values)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--grid', type=int, default=81)
    parser.add_argument('--policy-samples', type=int, default=32)
    parser.add_argument('--transition-samples', type=int, default=512)
    args = parser.parse_args()
    benchmark(args.output, args.device, args.grid, args.policy_samples, args.transition_samples)
