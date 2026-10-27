"""Audit the visible notebook implementations against the frozen native backend.

Import only definitions via AST: no long experiment runs or output directories
are created by these tests. The actual notebooks are executed separately.
"""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch
from pagr import runtime as rt

ROOT = Path(__file__).resolve().parents[1]


def navigation_definitions():
    source = ROOT/'notebooks/02_navigation_am_and_candidates.py'
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(
        node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    namespace = {'__name__': 'notebook_definitions'}
    exec(compile(tree, str(source), 'exec'), namespace)
    return namespace


@unittest.skipUnless(importlib.util.find_spec('nbformat') and importlib.util.find_spec('pandas'),
                     'Install .[notebook] to test standalone notebook sources')
class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.n = navigation_definitions()
        cls.la, _, _, cls.vg = rt.backend()
        cls.task, cls.cfg, cls.critics, cls.c, _ = cls.la.load_inputs('navigation', 18001)
        cls.nc = cls.n['Config']()

    def test_scripts_are_the_notebook_source(self):
        import nbformat
        spec = importlib.util.spec_from_file_location('builder', ROOT/'scripts/build_research_notebooks.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        for name in builder.NAMES:
            path = ROOT/'notebooks'/f'{name}.py'
            expected = builder.cells_from_script(path)
            actual = nbformat.read(path.with_suffix('.ipynb'), as_version=4)
            self.assertEqual([(c.cell_type, c.source) for c in expected],
                             [(c.cell_type, c.source) for c in actual.cells])
            ast.parse(path.read_text())
            self.assertNotIn('runtime.replay(', '\n'.join(c.source for c in expected if c.cell_type == 'code'))

    def test_regenerated_data_exact(self):
        data = self.n['generate_data'](self.nc, 18001)
        reference = np.load(rt.ASSETS/'original/pilot/navigation/seed_18001/critics/data.npz')
        for key in data:
            self.assertTrue(np.array_equal(data[key].numpy(), reference[key]), key)

    def test_dynamics_evaluation_and_matching_exact(self):
        torch.manual_seed(947)
        actor = self.n['Controller'](64)
        native = self.task.Controller(64)
        native.load_state_dict(actor.state_dict())
        p, u = self.n['rollout'](actor, self.nc, 13, self.n['rng'](84))
        pn, un = self.task.rollout(native, self.cfg, 13, self.task.rng(84))
        self.assertTrue(torch.equal(p, pn))
        self.assertTrue(torch.equal(u, un))
        terminals = torch.stack([self.task.raw_gradient(c, p[:, -1]) for c in self.critics])
        actual = self.n['matching_gradients'](actor, p, terminals, self.nc)
        expected = self.la.matching_gradients(native, p, terminals, self.task, self.cfg)
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(self.n['evaluate'](actor, self.nc, 80, n=64),
                         self.task.evaluate(native, self.cfg, 80, n=64))

    def test_native_adam_and_guard_exact_no_truth(self):
        torch.manual_seed(947)
        actor = self.n['Controller'](64)
        native = self.task.Controller(64)
        native.load_state_dict(actor.state_dict())
        opt = torch.optim.Adam(actor.parameters(), lr=.002)
        native_opt = torch.optim.Adam(native.parameters(), lr=.002)
        before = self.n['vector'](actor)
        for step in range(2):
            paths, _ = self.n['rollout'](actor, self.nc, 13, self.n['rng'](84+step))
            with self.n['no_oracle_access']():
                self.n['proposal'](actor, opt, paths, self.critics, self.nc)
            self.vg.proposal(native, native_opt, paths, self.critics, self.task, self.cfg)
            self.assertTrue(torch.equal(self.n['vector'](actor), self.vg.vector(native)))
        proposed = self.n['vector'](actor)
        for robust in (False, True):
            with self.n['no_oracle_access']():
                actual = self.n['guard'](actor, before, proposed, self.nc, self.critics, 79,
                                         self.n['Options'](guard_paths=16),
                                         self.n['Adversary'](self.c, .3), robust)
            expected = self.vg.guard(native, before, proposed, self.task, self.cfg, self.critics, 79,
                                     self.vg.Options(guard_paths=16), self.la.Adversary(self.c, .3), robust)
            self.assertEqual(actual, expected)
            self.assertTrue(torch.equal(self.n['vector'](actor), self.vg.vector(native)))

    def test_terminal_derivative_and_actor_projection(self):
        # The notebook checks the exact error-propagation/parameter-projection
        # identities using real learned critics, not an assumed error vector.
        n = self.n
        n.update(CFG=self.nc, OPT=n['Options'](), SEEDS=(18001,), CHECKS=[],
                 CRITICS={(18001, 'relu'): self.critics},
                 DATA={18001: n['generate_data'](self.nc, 18001)}, display=lambda *a: None)
        n['audit_numerics']()
        self.assertEqual(len(n['CHECKS']), 7)
        self.assertTrue(all(c['passed'] for c in n['CHECKS']))


if __name__ == '__main__':
    unittest.main()
