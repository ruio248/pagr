"""Backend checks against the original, independently implemented NumPy code."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('point_accel_test_source',
                                             ROOT/'scripts/point_reference_accelerated.py')
A = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = A
SPEC.loader.exec_module(A)


class AcceleratedReferenceTests(unittest.TestCase):
    def test_integral_matches_numpy_at_obstacle_goal_and_bounds(self):
        c = A.M.Config()
        rng = np.random.default_rng(10944)
        p = np.vstack((rng.uniform(0., 1., (30, 2)),
                       [[.3, .5], [.5, .32], [.85, .5], [.98, .6]]))
        a = rng.normal(size=p.shape)
        table = rng.normal(size=(17, 17))
        quad = A.M.normal_quadrature(2048)
        q, g = A.M.integral(A.M.PointObstacle(c), 10, p, a, table, quad, True)
        tq, tg = A.transition_integral(torch.tensor(p), torch.tensor(a), torch.tensor(table),
                                       torch.tensor(quad[0]), c.delta, c.sigma_env, c.gamma,
                                       *c.obstacle_center, c.obstacle_radius,
                                       *c.goal_center, c.goal_radius)
        np.testing.assert_allclose(tq.numpy(), q, atol=2e-14)
        np.testing.assert_allclose(tg.numpy(), g, atol=2e-13)

    def test_actor_sampler_matches_original(self):
        c = A.M.Config()
        torch.manual_seed(91)
        actor = A.M.core.ConditionalVelocity(c)
        rng = np.random.default_rng(30)
        p = rng.uniform(0, 1, (32, 2)).astype('f')
        z = rng.normal(size=(32, 2)).astype('f')
        k = np.arange(32) % c.horizon
        old = A.M.sample_actor(actor, actor, c, k, p, z, None)
        new = A.sample_actor_tensor(actor, c, torch.tensor(k), torch.tensor(p), torch.tensor(z))
        np.testing.assert_array_equal(new.numpy(), old)

    def test_value_table_matches_numpy_and_checks_actor_freshness(self):
        from dataclasses import replace
        c = replace(A.M.Config(), horizon=2, flow_steps=2)
        torch.manual_seed(119)
        actor = A.M.core.ConditionalVelocity(c)
        base = A.M.core.ConditionalVelocity(c)
        task = A.M.PointObstacle(c)
        old = A.M.PolicyTable(task, actor, base, 9, 2, 16, 128)
        new = A.AcceleratedPolicyTable(task, actor, base, 9, 2, 16, 128,
                                       device='cpu', compile_kernel=False)
        np.testing.assert_allclose(new.values, old.values, atol=1e-12)
        new.assert_current(actor, base)
        with torch.no_grad():
            next(actor.parameters()).add_(.001)
        with self.assertRaisesRegex(ValueError, 'stale'):
            new.assert_current(actor, base)


if __name__ == '__main__':
    unittest.main()
