import copy
import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

robust = load("terminal_consistent_robust_qam_test_module",
              ROOT / "scripts/terminal_consistent_robust_qam.py")
point = load("terminal_robust_point_obstacle_test",
             ROOT / "notebooks/02_qam_point_obstacle.py")


def test_quadratic_solver_matches_dense_circle_search_and_singular_case():
    torch.manual_seed(17)
    a = torch.randn(8, 2, dtype=torch.float64)
    raw = torch.randn(8, 2, 2, dtype=torch.float64)
    b = raw.transpose(-1, -2) @ raw
    kappa = .7
    q = robust.solve_unit_ball_quadratic(a, b, kappa)
    angles = torch.arange(16384, dtype=torch.float64) * (2 * torch.pi / 16384)
    circle = torch.stack((angles.cos(), angles.sin()), -1)
    candidates = torch.cat((circle, torch.zeros(1, 2, dtype=torch.float64)), 0)
    values = torch.einsum("nd,md->nm", a, candidates) - .5*kappa*torch.einsum(
        "mi,nij,mj->nm", candidates, b, candidates)
    exact = (a*q).sum(-1) - .5*kappa*torch.einsum("ni,nij,nj->n", q, b, q)
    assert bool((q.norm(dim=-1) <= 1. + 1e-9).all())
    assert bool((exact >= values.max(-1).values - 2e-7).all())

    # B is singular and A has a null-space component, so the maximizer lies
    # on the boundary and the multiplier must regularize the null direction.
    a1 = torch.tensor([[.2, 1.]], dtype=torch.float64)
    b1 = torch.tensor([[[2., 0.], [0., 0.]]], dtype=torch.float64)
    q1 = robust.solve_unit_ball_quadratic(a1, b1, 1.)
    theta = torch.arange(65536, dtype=torch.float64) * (2 * torch.pi / 65536)
    circle = torch.stack((theta.cos(), theta.sin()), -1)
    values = circle @ a1[0] - (circle[:, 0] ** 2)
    value1 = (a1[0] @ q1[0]) - q1[0, 0] ** 2
    assert torch.isclose(q1.norm(), torch.tensor(1., dtype=q1.dtype), atol=1e-8)
    assert value1 >= values.max() - 2e-7


def test_discrete_propagator_matches_lean_adjoint_recurrence():
    torch.manual_seed(21)
    c = point.Config(flow_steps=4, warmup_steps=1, horizon=5)
    base = point.core.ConditionalVelocity(c)
    for p in base.parameters():
        p.requires_grad_(False)
    k = torch.tensor([0, 2, 4])
    y = torch.randn(3, 2)
    paths = torch.randn(3, c.flow_steps + 1, 2)
    terminal = torch.randn(3, 2)
    propagated = point.core.lean_adjoint(base, c, k, y, paths, terminal)
    P = robust.adjoint_propagators(base, c, k, y, paths)
    rebuilt = (P @ terminal[:, None, :, None]).squeeze(-1)
    assert torch.allclose(rebuilt, propagated, atol=2e-6, rtol=2e-6)


def test_zero_radius_matches_ordinary_lean_am_actor_update():
    torch.manual_seed(27)
    c = point.Config(flow_steps=3, warmup_steps=1, actor_batch=4, horizon=5)
    base = point.core.ConditionalVelocity(c)
    for p in base.parameters():
        p.requires_grad_(False)
    ordinary, candidate = copy.deepcopy(base), copy.deepcopy(base)
    for model in (ordinary, candidate):
        for p in model.parameters():
            p.requires_grad_(True)
    critic = point.core.Critic(c)
    opt0 = torch.optim.Adam(ordinary.parameters(), lr=c.actor_lr)
    opt1 = torch.optim.Adam(candidate.parameters(), lr=c.actor_lr)
    k = torch.tensor([0, 1, 2, 3])
    y, z = torch.randn(4, 2), torch.randn(4, 2)
    xi = torch.randn(4, c.flow_steps, 2)
    point.core.actor_step(ordinary, opt0, base, critic, None, c, k, y, z, xi, "learned")
    robust.terminal_consistent_actor_step(candidate, opt1, base, critic, None, c,
                                          k, y, z, xi, 0., point.core)
    assert torch.allclose(point.core.parameter_vector(ordinary),
                          point.core.parameter_vector(candidate), atol=1e-7, rtol=1e-6)


def test_positive_radius_uses_one_terminal_perturbation_across_the_path():
    torch.manual_seed(31)
    c = point.Config(flow_steps=3, warmup_steps=1, actor_batch=5, horizon=5)
    base = point.core.ConditionalVelocity(c)
    for p in base.parameters():
        p.requires_grad_(False)
    actor = copy.deepcopy(base)
    for p in actor.parameters():
        p.requires_grad_(True)
    critic = point.core.Critic(c)
    k = torch.tensor([0, 1, 2, 3, 4])
    y, z = torch.randn(5, 2), torch.randn(5, 2)
    xi = torch.randn(5, c.flow_steps, 2)
    paths, _ = point.core.rollout(actor, base, c, k, y, z, xi)
    terminal = point.core.terminal_label(None, critic, c, k, y, paths[:, -1], "learned")
    p_hat = point.core.lean_adjoint(base, c, k, y, paths, terminal)
    p_robust, q, _ = robust.pathwise_terminal_robust_target(base, c, k, y,
                                                              paths, p_hat, .08)
    P = robust.adjoint_propagators(base, c, k, y, paths)
    expected = p_hat + c.beta * .08 * (P @ q[:, None, :, None]).squeeze(-1)
    assert torch.allclose(p_robust, expected, atol=1e-6, rtol=1e-6)

    opt = torch.optim.Adam(actor.parameters(), lr=c.actor_lr)
    result = robust.terminal_consistent_actor_step(
        actor, opt, base, critic, None, c, k, y, z, xi, .08, point.core)
    assert torch.isfinite(point.core.parameter_vector(actor)).all()
    assert 0. <= result["q_norm_mean"] <= 1. + 1e-6
