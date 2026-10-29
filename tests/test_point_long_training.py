"""Tests for scientific admission and independent-MC equivalence."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
SPEC = importlib.util.spec_from_file_location('point_long_test_source',
                                             ROOT/'scripts/run_point_long_training.py')
L = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = L
SPEC.loader.exec_module(L)


class LongTrainingProtocolTests(unittest.TestCase):
    def test_mc_matches_original_with_identical_iid_stream(self):
        from dataclasses import replace
        c = replace(L.M.Config(), horizon=4, flow_steps=2)
        task = L.M.PointObstacle(c)
        torch.manual_seed(911)
        actor = L.M.core.ConditionalVelocity(c)
        p, a = np.array([.72, .5]), np.array([1.4, -.2])
        original = L.M.mc_fixed_q(task, actor, actor, 1, p, a, 4096, 412)
        accelerated = L.independent_mc(task, actor, actor, 1, p, a, 4096, 412, 'cpu')
        for key in ('q', 'q_se', 'gradient', 'gradient_se'):
            np.testing.assert_allclose(accelerated[key], original[key], atol=1e-14)

    def test_wrong_direction_or_imprecise_mc_cannot_pass(self):
        import json
        protocol = json.loads((ROOT/'configs/point_long_training.json').read_text())
        g = np.array([[.2, .3], [.3, -.4], [.4, .7]])
        se = np.full_like(g, .001)
        good = L.gradient_gate(g, g+.001, g+.001, se, g+.3, protocol)
        self.assertTrue(good['passed'])
        wrong = L.gradient_gate(g, g, -g, se, g+.3, protocol)
        self.assertFalse(wrong['passed'])
        uncertain = L.gradient_gate(g, g, g, np.ones_like(g), g+.3, protocol)
        self.assertFalse(uncertain['passed'])
        self.assertFalse(uncertain['checks']['mc_precision'])

    def test_resume_restores_optimizers_rng_and_update(self):
        from dataclasses import replace
        import tempfile
        c = replace(L.M.Config(), horizon=3, flow_steps=2)
        base = L.M.core.ConditionalVelocity(c)
        critic = L.M.core.Critic(c)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            st, N = L.state_from_initial(base, critic, critic, c, 59001, output, 'learned', 'fixed')
            st['aopt'].zero_grad()
            sum(p.square().sum() for p in st['actor'].parameters()).backward()
            st['aopt'].step()
            torch.randn((17, 2), generator=st['rng'])
            L.checkpoint(st, base, 12, output, 'learned', 'fixed')
            restored, N = L.state_from_initial(base, critic, critic, c, 59001, output, 'learned', 'fixed')
            self.assertEqual(N, 12)
            self.assertEqual(L.M.model_hash(st['actor']), L.M.model_hash(restored['actor']))
            self.assertTrue(torch.equal(torch.randn(10, generator=st['rng']),
                                        torch.randn(10, generator=restored['rng'])))
            old = next(iter(st['aopt'].state.values()))
            new = next(iter(restored['aopt'].state.values()))
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(old[key], new[key]))
            with self.assertRaisesRegex(RuntimeError, 'provenance'):
                L.state_from_initial(base, critic, critic, c, 59001, output, 'learned', 'changed')


if __name__ == '__main__':
    unittest.main()
