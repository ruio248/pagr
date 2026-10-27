"""Focused tests for the exploratory Oracle AM gain-space extension."""
import ast
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


def notebook_definitions():
    path = ROOT/'notebooks/02_flow_am_terminal_gradient.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(
        node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    namespace = {'__name__': 'oracle_notebook_definitions', 'ROOT': ROOT}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


class OracleDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.n = notebook_definitions()
        cls.c = replace(cls.n['configuration']('smoke'), actor_updates=2,
                        diagnostic_steps=(0, 1, 2), curve_every=1, curve_paths=8,
                        final_paths=16)
        torch.manual_seed(834791)
        cls.base = cls.n['freeze'](cls.n['Velocity'](cls.c.width))
        cls.models = [cls.n['freeze'](cls.n['RewardModel'](cls.c.width))
                      for _ in range(cls.c.members)]

    def test_oracle_b128_reaches_true_gradient_and_learned_b128_cannot(self):
        n, c = self.n, self.c
        true_gradient = n['true_gradient']
        calls = []

        def counted(x, cfg):
            calls.append(len(x))
            return true_gradient(x, cfg)

        n['true_gradient'] = counted
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                oracle_dir = root/'oracle'
                oracle_dir.mkdir()
                n['train_actor'](self.base, self.models, c, c, 91001, 'explicit-oracle',
                                 oracle_dir, root, gradient_source='oracle', batch_size=128)
                self.assertEqual(calls, [128, 128])
                budget = json.loads((oracle_dir/'budget.json').read_text())
                self.assertEqual(budget['actor_true_gradient_points'], 256)
                self.assertEqual(budget['gradient_source'], 'oracle')
                calls.clear()
                learned_dir = root/'learned'
                learned_dir.mkdir()
                n['train_actor'](self.base, self.models, c, c, 91001, 'explicit-learned',
                                 learned_dir, root, gradient_source='learned', batch_size=128)
                self.assertEqual(calls, [])
                self.assertEqual(json.loads((learned_dir/'budget.json').read_text())[
                    'actor_true_gradient_points'], 0)
                self.assertEqual(torch.load(oracle_dir/'checkpoint-0.pt', weights_only=False)['actor'].keys(),
                                 torch.load(learned_dir/'checkpoint-0.pt', weights_only=False)['actor'].keys())
                initial_oracle = torch.load(oracle_dir/'checkpoint-0.pt', weights_only=False)['actor']
                initial_learned = torch.load(learned_dir/'checkpoint-0.pt', weights_only=False)['actor']
                self.assertTrue(n['exact_tree_equal'](initial_oracle, initial_learned))
                self.assertTrue(n['exact_tree_equal'](initial_oracle, self.base.state_dict()))
        finally:
            n['true_gradient'] = true_gradient

    def test_frozen_models_and_prefix_substreams(self):
        n, c = self.n, self.c
        hashes = [n['model_hash'](m) for m in [self.base, *self.models]]
        d = n['oracle_diagnostic_configuration'](c)
        offsets = []
        for cp in range(len(d.checkpoint_methods)):
            for repetition in range(d.repeats):
                role = d.train_noise_offset+cp*100000+4*repetition
                offsets.extend([role, role+1, role+2, role+3])
                a = n['noise'](c, 64, n['seed_key'](91001, role))
                b = n['noise'](c, 128, n['seed_key'](91001, role))
                self.assertTrue(all(torch.equal(x, y[:64]) for x, y in zip(a, b)))
        self.assertEqual(len(offsets), len(set(offsets)))
        self.assertTrue(min(offsets) > 700001)
        self.assertEqual(hashes, [n['model_hash'](m) for m in [self.base, *self.models]])

    def test_virtual_adam_does_not_mutate_formal_state_or_rng(self):
        n, c = self.n, self.c
        actor = n['actor_copy'](self.base)
        optimizer = n['adam_for'](actor, c)
        frozen_hashes = [n['model_hash'](m) for m in [self.base, *self.models]]
        z, xi = n['noise'](c, 8, n['seed_key'](91001, 1840000))
        n['actor_update'](actor, optimizer, self.base, self.models, c, z, xi, oracle=False)
        state_before = n['copy'].deepcopy(optimizer.state_dict())
        params_before = n['flat_parameters'](actor)
        rng_before = torch.get_rng_state().clone()
        ql, qo = n['gradient_pair'](actor, self.base, self.models, c, z, xi)
        ze, xie = n['noise'](c, 16, n['seed_key'](91001, 1840002))
        metrics, paired_paths = n['virtual_adam_pair'](actor, optimizer, self.base, c, ql, qo, ze, xie)
        self.assertTrue(torch.equal(params_before, n['flat_parameters'](actor)))
        self.assertTrue(n['exact_tree_equal'](state_before, optimizer.state_dict()))
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertAlmostEqual(metrics['D_J'], float(paired_paths[0].mean()), places=6)
        self.assertAlmostEqual(metrics['D_J_mc_se'], float(
            np.std(paired_paths[0], ddof=1)/np.sqrt(len(paired_paths[0]))), places=6)
        self.assertEqual(frozen_hashes, [n['model_hash'](m) for m in [self.base, *self.models]])

    def test_signed_effects_and_cross_covariance(self):
        n = self.n
        rows = []
        for seed in (1, 2, 3):
            for label, batch, step, cost, reward in [
                ('AM-learned-B64', 64, 300, 10., 2.),
                ('AM-oracle-B64', 64, 300, 8., 3.),
                ('AM-learned-B128', 128, 300, 7., 4.),
                ('AM-oracle-B128', 128, 300, 6., 4.5),
                ('AM-learned-B128', 128, 150, 8., 3.),
                ('AM-oracle-B128', 128, 150, 7., 3.5)]:
                rows.append(dict(seed=seed, method=label, batch_size=batch, update=step,
                                 cost=cost, reward=reward))
        frame = pd.DataFrame(rows)
        by_seed, _ = n['seed_effects'](frame, 'cost', 'equal_updates')
        values = {row['effect']: row['value'] for row in by_seed if row['seed'] == 1}
        self.assertEqual(values, dict(A64=2., A128=1., S_learned=3., S_oracle=2., I=1.))
        by_seed, _ = n['seed_effects'](frame, 'reward', 'equal_updates')
        values = {row['effect']: row['value'] for row in by_seed if row['seed'] == 1}
        self.assertEqual(values, dict(A64=1., A128=.5, S_learned=2., S_oracle=1.5, I=.5))
        by_seed, _ = n['seed_effects'](frame, 'cost', 'equal_rollouts')
        self.assertEqual({row['effect']: row['value'] for row in by_seed if row['seed'] == 1}['S_learned'], 2.)
        qo = torch.tensor([[1., 0.], [2., 0.], [4., 0.]])
        d = .5*qo
        ql = qo+d
        vl = n['trace_covariance'](ql)
        vo = n['trace_covariance'](qo)
        vd = n['trace_covariance'](d)
        cross = n['trace_covariance'](qo, d)
        self.assertGreater(cross, 0)
        self.assertAlmostEqual(vl, vo+vd+2*cross)
        self.assertNotAlmostEqual(vl, vo+vd)


if __name__ == '__main__':
    unittest.main()
