"""Numerical audits of the self-contained Flow AM notebook; no training on import."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def definitions():
    path = ROOT/'notebooks/02_flow_am_terminal_gradient.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(
        n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    namespace = {'__name__': 'flow_notebook_definitions'}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


@unittest.skipUnless(importlib.util.find_spec('pandas'), 'Install .[notebook]')
class FlowAMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.n = definitions()

    def test_all_numerical_identities(self):
        n = self.n
        torch.manual_seed(82004)  # separate unit-test seed, not formal experiment
        c = n['configuration']('smoke')
        base = n['freeze'](n['Velocity'](c.width))
        models = [n['freeze'](n['RewardModel'](c.width)) for _ in range(c.members)]
        checks = n['implementation_audit'](base, models, c)
        self.assertGreaterEqual(len(checks), 19)
        self.assertTrue(all(x['passed'] for x in checks))

    def test_protocol_and_finite_endpoints(self):
        n = self.n
        c = n['configuration']()
        self.assertEqual(c.seeds, (28001, 28002, 28003))
        self.assertTrue(set(c.seeds).isdisjoint(n['configuration']('smoke').seeds))
        t, nu, h = n['grid'](c)
        self.assertGreater(float(t.min()), 0.)
        self.assertLess(float(t.max()), 1.)
        self.assertGreater(float(nu.min()), 0.)
        self.assertAlmostEqual(c.epsilon+c.steps*h, 1.)
        config = ROOT/'configs/flow_am.json'
        if config.exists():
            self.assertEqual(json.loads(config.read_text()), json.loads(json.dumps(n['asdict'](c))))

    def test_data_replay_scalar_observations_only(self):
        n = self.n
        c = n['configuration']('smoke')
        a, b = n['make_data'](c, 82005), n['make_data'](c, 82005)
        self.assertEqual(set(a), {'behavior', 'heldout', 'reward_x', 'valid_x', 'reward_y', 'valid_y'})
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]))
        self.assertEqual(a['reward_y'].shape, (c.reward_train,))
        self.assertGreater(float((a['reward_y']-n['true_reward'](a['reward_x'], c)).std()), 0.)

    def test_history_archive_bytes(self):
        import hashlib
        manifest = ROOT/'notebooks/archive/manifest.json'
        self.assertTrue(manifest.exists())
        for item in json.loads(manifest.read_text())['files']:
            for path in (item['original'], item['archived']):
                self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), item['sha256'])

    def test_training_seed_namespaces_do_not_overlap(self):
        n = self.n
        c = n['configuration']()
        offsets = list(range(10, 23))+list(range(100, 105))+list(range(200, 205))
        offsets += list(range(100000, 100000+2*c.actor_updates))
        offsets += [300000, 300001, 700000, 700001]
        for step in c.diagnostic_steps:
            offsets += [400000+2*step, 400001+2*step, 500000+2*step, 500001+2*step]
            for fraction in c.conditional_fractions:
                for state in range(c.conditional_states):
                    offset = 600000+step*100+1000*int(c.steps*fraction)+2*state
                    offsets += [offset, offset+1]
        self.assertEqual(len(offsets), len(set(offsets)))
        streams = [{n['seed_key'](seed, offset) for offset in offsets} for seed in c.seeds]
        for i in range(len(streams)):
            for j in range(i):
                self.assertTrue(streams[i].isdisjoint(streams[j]))


if __name__ == '__main__':
    unittest.main()
