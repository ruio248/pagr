"""Numerical checks for the dimension/coupling notebook without training on import."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def definitions():
    path = ROOT/'notebooks/02_flow_am_dimension_coupling.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(
        node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    namespace = {'__name__': 'dimension_notebook_definitions',
                 'ROOT': ROOT, 'SPEC_PATH': ROOT/'configs/flow_am_dimension_coupling.json'}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


@unittest.skipUnless(importlib.util.find_spec('pandas'), 'Install .[notebook]')
class DimensionCouplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(89001)
        cls.n = definitions()
        cls.c = cls.n['configuration']('smoke')

    def fixture(self, dim):
        n, c = self.n, self.c
        base = n['freeze'](n['Velocity'](dim, c.width))
        actor = n['actor_copy'](base)
        model = n['freeze'](n['RewardModel'](dim, c.width))
        z, xi = n['noise'](c, dim, 6, 89010+dim)
        paths, control = n['rollout'](actor, base, c, z, xi)
        return base, actor, [model], paths, control, z, xi

    def test_protocol_and_network_sampler_shapes(self):
        n, c = self.n, self.c
        formal = n['configuration']('formal')
        self.assertEqual([(t.name, t.dim, t.rho) for t in formal.tasks],
                         [('independent_d2', 2, 0.), ('independent_d16', 16, 0.),
                          ('coupled_d16', 16, .5)])
        self.assertEqual(formal.seeds, (38001, 38002, 38003))
        self.assertEqual((formal.steps, formal.actor_batch, formal.actor_updates), (64, 64, 300))
        self.assertNotIn(c.seeds[0], formal.seeds)
        self.assertFalse(formal.optional_equal_error_diagnostic_enabled)
        for dim in (2, 16):
            with self.subTest(dim=dim):
                base, actor, models, paths, control, z, xi = self.fixture(dim)
                self.assertEqual(base(z, .3).shape, (6, dim))
                self.assertEqual(models[0](z).shape, (6,))
                self.assertEqual(paths.shape, (6, c.steps+1, dim))
                self.assertEqual(control.shape, (6, c.steps, dim))
                self.assertEqual(xi.shape, (6, c.steps, dim))
                self.assertEqual(n['ode_sample'](base, z, 4).shape, (6, dim))
                self.assertTrue(torch.equal(n['flat_parameters'](base), n['flat_parameters'](actor)))
                self.assertTrue(all(not p.requires_grad for p in base.parameters()))
                self.assertTrue(all(p.requires_grad for p in actor.parameters()))

    def test_behavior_inverse_and_analytic_reward_gradient(self):
        n = self.n
        for dim in (2, 16):
            z = torch.randn(40, dim)
            self.assertTrue(torch.equal(n['behavior_transform'](z, 0.), z))
            a = n['behavior_transform'](z, .5)
            self.assertLess(float((n['behavior_inverse'](a, .5)-z).abs().max()), 1e-6)
            self.assertTrue(torch.allclose(a[:, 1:], z[:, 1:]+.5*torch.sin(z[:, :-1])))
            auto = n['scalar_gradient'](n['true_reward'], a)
            self.assertTrue(torch.allclose(auto, n['true_gradient'](a), atol=1e-6))

    def test_reward_raw_units_and_input_gradient_scale(self):
        n = self.n
        model = n['RewardModel'](16, self.c.width)
        model.label_mean.fill_(20.)
        model.label_std.fill_(7.)
        x = torch.randn(4, 16).requires_grad_(True)
        grad_normal = torch.autograd.grad(model.normalized(x).sum(), x, retain_graph=True)[0]
        grad_raw = torch.autograd.grad(model(x).sum(), x)[0]
        self.assertTrue(torch.allclose(grad_raw, 7*grad_normal, atol=1e-5))
        self.assertTrue(torch.allclose(model(x), 20+7*model.normalized(x)))

    def test_frozen_base_vjp_and_explicit_lean_recurrence(self):
        n, c = self.n, self.c
        for dim in (2, 16):
            base, actor, models, paths, *_ = self.fixture(dim)
            x = paths[:, 0].detach().requires_grad_(True)
            v = torch.randn_like(x)
            auto = torch.autograd.grad((base(x, .4)*v).sum(), x)[0]
            direct = base.input_vjp(x.detach(), .4, v)
            self.assertTrue(torch.allclose(auto, direct, rtol=2e-5, atol=2e-6))
            terminal = torch.randn(len(paths), dim)
            lean = n['lean_adjoint'](paths, base, c, terminal)
            times, _, h = n['grid'](c)
            p = terminal
            explicit = [p]
            for k in range(c.steps-1, -1, -1):
                xk = paths[:, k].detach().requires_grad_(True)
                drift = 2*base(xk, times[k])-xk/times[k]
                jtv = torch.autograd.grad((drift*p).sum(), xk)[0]
                p = p+h*jtv
                explicit.append(p)
            self.assertTrue(torch.allclose(lean, torch.stack(explicit[::-1], 1),
                                           rtol=2e-5, atol=2e-6))

    def test_terminal_label_parameter_gradient_identities_and_detachment(self):
        n, c = self.n, self.c
        base, actor, models, paths, *_ = self.fixture(16)
        self.assertFalse(paths.requires_grad)
        delta = torch.randn(len(paths), 16)
        zero = torch.zeros_like(delta)
        propagated = n['lean_adjoint'](paths, base, c, delta)
        self.assertTrue(torch.equal(propagated[:, -1], delta))
        self.assertFalse(propagated.requires_grad)
        p0 = n['lean_adjoint'](paths, base, c, zero)
        p1 = n['lean_adjoint'](paths, base, c, delta)
        self.assertTrue(torch.allclose(p1-p0, propagated))
        q0 = n['flat_gradient'](n['matching_loss'](actor, base, paths, p0, c), actor)
        q1 = n['flat_gradient'](n['matching_loss'](actor, base, paths, p1, c), actor)
        times, nu, h = n['grid'](c)
        x = paths[:, :-1]
        t = times[None, :, None].expand(len(x), -1, 1)
        linear = (h/(len(x)*actor.dim)*
                  (2*(actor(x, t)-base(x, t))*propagated[:, 1:]).sum())
        q_delta = n['flat_gradient'](linear, actor)
        self.assertTrue(torch.allclose(q1-q0, q_delta, rtol=2e-5, atol=2e-6))
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))

    def test_matching_whole_loss_divided_by_dimension(self):
        n, c = self.n, self.c
        for dim in (2, 16):
            base, actor, _, paths, *_ = self.fixture(dim)
            labels = torch.randn_like(paths)
            loss = n['matching_loss'](actor, base, paths, labels, c)
            times, nu, h = n['grid'](c)
            t = times[None, :, None].expand(len(paths), -1, 1)
            residual = 2*(actor(paths[:, :-1], t)-base(paths[:, :-1], t))/nu[None, :, None]
            residual = residual+nu[None, :, None]*labels[:, 1:]
            unnormalized = .5*h*residual.square().sum((-1, -2)).mean()
            self.assertTrue(torch.allclose(loss*dim, unnormalized, atol=1e-5))

    def test_truth_guard_and_virtual_adam_purity(self):
        n, c = self.n, self.c
        base, actor, models, paths, control, z, xi = self.fixture(2)
        optimizer = n['adam_for'](actor, c)
        before = n['flat_parameters'](actor)
        before_opt = copy.deepcopy(optimizer.state_dict())
        with n['forbid_truth']():
            n['actor_update'](actor, optimizer, base, models, c, z, xi, 'learned')
            with self.assertRaises(AssertionError):
                n['true_reward'](paths[:, -1])
        q1, q2 = torch.randn_like(before), torch.randn_like(before)
        after = n['flat_parameters'](actor).clone()
        after_opt = copy.deepcopy(optimizer.state_dict())
        n['virtual_adam_steps'](actor, optimizer, q1, q2, c)
        self.assertTrue(torch.equal(after, n['flat_parameters'](actor)))
        self.assertTrue(n['exact_tree_equal'](after_opt, optimizer.state_dict()))
        self.assertFalse(n['exact_tree_equal'](before_opt, optimizer.state_dict()))

    def test_finite_gaussian_chain_kl_and_seed_namespaces(self):
        n, c = self.n, self.c
        for dim in (2, 16):
            base, actor, _, paths, control, z, xi = self.fixture(dim)
            metrics, arrays = n['evaluate'](actor, base, c, z, xi)
            self.assertLess(metrics['kl_identity_max_abs'], 2e-5)
            self.assertTrue(torch.allclose(arrays['energy'],
                                           .5*n['grid'](c)[2]*control.square().sum((-1, -2))))
            self.assertTrue(torch.allclose(arrays['cost'],
                                           arrays['energy']-c.beta*arrays['reward']))
        seeds = set()
        for task_index in range(3):
            for seed in n['configuration']('formal').seeds:
                key = n['seed_key'](task_index, seed, 10)
                self.assertNotIn(key, seeds)
                seeds.add(key)

    def test_historical_flow_archive_unchanged(self):
        path = ROOT/'notebooks/archive/flow_am_four_mode/manifest.json'
        self.assertTrue(path.exists())
        manifest = json.loads(path.read_text())
        for item in manifest['files']:
            for relative in (item['original'], item['archived']):
                self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),
                                 item['sha256'])
        for directory in manifest['historical_result_directories']:
            self.assertTrue((ROOT/directory/'completion.json').exists())


if __name__ == '__main__':
    unittest.main()
