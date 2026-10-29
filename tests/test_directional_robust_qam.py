import importlib.util
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("directional_robust_qam", ROOT/"scripts/directional_robust_qam.py")
robust = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = robust
spec.loader.exec_module(robust)


def test_identical_and_zero_alpha_are_exact_mean():
    g = torch.tensor([[[1., 2.], [1., 2.], [1., 2.]],
                      [[0., 0.], [0., 0.], [0., 0.]]])
    filtered, sigma = robust.directional_robust_gradient(g, 10.)
    assert torch.equal(filtered, g.mean(1))
    assert torch.equal(sigma, torch.zeros_like(sigma))
    h = torch.randn(7, 3, 2)
    assert torch.equal(robust.directional_robust_gradient(h, 0.)[0], h.mean(1))


def test_axis_specific_disagreement():
    g = torch.tensor([[[1., 1.], [1., -1.], [1., .7], [1., -.7]]])
    # Put a useful y mean into the batch to test attenuation, while retaining
    # zero x disagreement and a large y disagreement.
    g[:, :, 1] += .2
    filtered, _ = robust.directional_robust_gradient(g, 3.)
    assert abs(float(filtered[0, 0])-1.) < 1e-6
    assert 0 < float(filtered[0, 1]) < .06


def test_rotated_covariance_is_rotation_equivariant():
    g = torch.tensor([[[2., 0.], [1.7, .8], [2.3, -.8], [2.1, .2]]])
    theta = torch.tensor(.43)
    rotation = torch.stack((torch.stack((theta.cos(), -theta.sin())),
                            torch.stack((theta.sin(), theta.cos()))))
    original, _ = robust.directional_robust_gradient(g, 3.)
    rotated, _ = robust.directional_robust_gradient(g@rotation.T, 3.)
    assert torch.allclose(rotated, original@rotation.T, atol=1e-6)
    assert torch.allclose(robust.eigen_analysis(g, 3.)['shrink_factors'],
                          robust.eigen_analysis(g@rotation.T, 3.)['shrink_factors'], atol=1e-6)


def test_fail_loudly_and_member_grads_are_real():
    with pytest.raises(ValueError):
        robust.directional_robust_gradient(torch.zeros(2, 1, 2), 1.)
    bad = torch.zeros(2, 3, 2); bad[0, 0, 0] = float('nan')
    with pytest.raises(FloatingPointError):
        robust.directional_robust_gradient(bad, 1.)
    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.members = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
        def forward(self, k, y, a):
            return torch.stack((a[:, 0], 2*a[:, 1], a[:, 0]+a[:, 1]), -1)
    g = robust.critic_member_action_gradients(Critic(), torch.zeros(2), torch.zeros(2, 2),
                                                torch.zeros(2, 2))
    assert torch.equal(g[0], torch.tensor([[1., 0.], [0., 2.], [1., 1.]]))


def test_alpha_zero_actor_update_matches_untouched_ordinary_step():
    point_spec = importlib.util.spec_from_file_location('point_obstacle_zero_alpha_test',
                                                        ROOT/'notebooks/02_qam_point_obstacle.py')
    point = importlib.util.module_from_spec(point_spec)
    sys.modules[point_spec.name] = point
    point_spec.loader.exec_module(point)
    torch.manual_seed(5)
    c = point.Config(flow_steps=3, warmup_steps=2, actor_batch=4)
    base = point.core.ConditionalVelocity(c)
    for p in base.parameters(): p.requires_grad_(False)
    ordinary = copy.deepcopy(base)
    filtered = copy.deepcopy(base)
    for model in (ordinary, filtered):
        for p in model.parameters(): p.requires_grad_(True)
    critic = point.core.Critic(c)
    old_opt = torch.optim.Adam(ordinary.parameters(), lr=c.actor_lr)
    new_opt = torch.optim.Adam(filtered.parameters(), lr=c.actor_lr)
    k = torch.tensor([0, 1, 2, 3])
    y = torch.randn(4, 2)
    z = torch.randn(4, 2)
    flow = torch.randn(4, c.flow_steps, 2)
    point.core.actor_step(ordinary, old_opt, base, critic, None, c, k, y, z, flow, 'learned')
    robust.directional_robust_actor_step(filtered, new_opt, base, critic, c, k, y, z,
                                          flow, 0., point.core)
    assert torch.allclose(point.core.parameter_vector(ordinary),
                          point.core.parameter_vector(filtered), atol=1e-7, rtol=1e-6)
