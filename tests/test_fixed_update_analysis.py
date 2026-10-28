"""Audit the fixed-N analysis against the preserved formal training run."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import nbformat
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'runs/02-flow-am-dim-formal-8q5obfng'
ANALYSIS = ROOT/'runs/02-flow-am-update-analysis-formal-q6xpxevn'


def definitions():
    path = ROOT/'notebooks/02_flow_am_dimension_coupling.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(
        node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    namespace = {'__name__': 'fixed_update_definitions', 'ROOT': ROOT,
                 'SPEC_PATH': ROOT/'configs/flow_am_dimension_coupling.json',
                 'FIXED_SPEC_PATH': ROOT/'configs/flow_am_fixed_update_analysis.json',
                 'FIXED_SPEC_SHA': hashlib.sha256(
                     (ROOT/'configs/flow_am_fixed_update_analysis.json').read_bytes()).hexdigest(),
                 'CONFIG_SHA': hashlib.sha256(
                     (ROOT/'configs/flow_am_dimension_coupling.json').read_bytes()).hexdigest()}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


class FixedUpdateAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.n = definitions()
        cls.c = cls.n['configuration']('formal')
        cls.curves = pd.read_csv(SOURCE/'learning-curves.csv')
        cls.final = pd.read_csv(SOURCE/'final.csv')

    def test_N_K_B_and_checkpoint_optimizer_counts(self):
        c = self.c
        protocol = self.n['fixed_analysis_protocol'](c, SOURCE)
        self.assertEqual(protocol['actor_update_counts'], [0, 50, 100, 150, 200, 250, 300])
        self.assertEqual(protocol['core_checkpoints'], [50, 150, 300])
        self.assertFalse(protocol['match_path_KL'])
        self.assertEqual((c.actor_updates, c.steps, c.actor_batch), (300, 64, 64))
        self.assertEqual(c.actor_checkpoints, (0, 150, 300))
        self.assertEqual(c.curve_paths, 512)
        self.assertEqual(c.final_paths, 4096)
        self.assertEqual(set(self.curves['update']), set(range(0, 301, 50)))
        for task in c.tasks:
            for seed in c.seeds:
                location = SOURCE/task.name/f'seed-{seed}'
                for source in ('learned', 'oracle'):
                    budget = json.loads((location/source/'budget.json').read_text())
                    self.assertEqual(budget['updates'], c.actor_updates)
                    self.assertEqual(budget['training_rollouts'], c.actor_updates*c.actor_batch)
                    for step in c.actor_checkpoints:
                        state = torch.load(location/source/f'checkpoint-{step}.pt',
                                           weights_only=True)
                        self.assertEqual(state['update'], step)
                        if step == 0:
                            self.assertFalse(state['optimizer']['state'])
                        else:
                            counts = {int(value['step']) for value in
                                      state['optimizer']['state'].values()}
                            self.assertEqual(counts, {step})

    def test_initialization_and_pairing(self):
        n, c = self.n, self.c
        paired, summary = n['paired_updates'](self.curves, c)
        self.assertEqual(len(paired), 63)
        self.assertEqual(len(summary), 21)
        self.assertTrue((paired[paired.N.eq(0)][
            ['delta_reward_per_dim', 'delta_J_per_dim', 'delta_KL_per_dim']] == 0).all().all())
        self.assertTrue((paired.K == 64).all() and (paired.B == 64).all())
        self.assertTrue((paired.evaluation_paths == 512).all())
        residual = paired.delta_J_per_dim-(c.beta*paired.delta_reward_per_dim-
                                           paired.delta_KL_per_dim)
        self.assertLess(float(residual.abs().max()), 1e-7)
        example = paired[(paired.task == 'independent_d16') &
                         (paired.seed == 38002) & (paired.N == 150)].iloc[0]
        rows = self.curves[(self.curves.task == 'independent_d16') &
                           (self.curves.seed == 38002) &
                           (self.curves['update'] == 150)].set_index('source')
        self.assertAlmostEqual(example.delta_reward_per_dim,
                               rows.loc['oracle'].reward_per_dim-
                               rows.loc['learned'].reward_per_dim)
        self.assertAlmostEqual(example.delta_J_per_dim,
                               rows.loc['learned'].cost_per_dim-
                               rows.loc['oracle'].cost_per_dim)
        self.assertEqual(set(summary.training_seeds), {3})

    def test_curve_and_final_streams_are_separate(self):
        n, c = self.n, self.c
        final = n['paired_final'](self.final, c)
        self.assertEqual(len(final), 9)
        self.assertTrue((final.N == 300).all() and (final.K == 64).all()
                        and (final.B == 64).all())
        self.assertTrue((final.evaluation_paths == 4096).all())
        self.assertEqual(set(final.evaluation_noise_role), {700000})
        curve = self.curves[(self.curves.task == 'independent_d2') &
                            (self.curves.seed == 38001) &
                            (self.curves.source == 'learned') &
                            (self.curves['update'] == 300)].iloc[0]
        high = final[(final.task == 'independent_d2') & (final.seed == 38001)].iloc[0]
        self.assertNotEqual(curve.reward_per_dim, high.learned_reward_per_dim)
        residual = final.delta_J_per_dim-(c.beta*final.delta_reward_per_dim-
                                          final.delta_KL_per_dim)
        self.assertLess(float(residual.abs().max()), 1e-7)

    def test_read_only_reuse_preserves_originals(self):
        n, c = self.n, self.c
        paths = [SOURCE/'learning-curves.csv', SOURCE/'final.csv',
                 SOURCE/'completion.json',
                 SOURCE/'independent_d16/seed-38002/base.pt',
                 SOURCE/'independent_d16/seed-38002/learned/checkpoint-300.pt']
        before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        with tempfile.TemporaryDirectory(prefix='pagr-analysis-test-', dir=ROOT/'runs') as temp:
            destination = Path(temp)
            loaded = n['load_completed_suite'](c, SOURCE, destination)
            self.assertEqual(len(loaded['curves']), 126)
            lineage = json.loads((destination/'source-manifest.json').read_text())
            self.assertEqual(lineage['new_actor_optimizer_steps'], 0)
            self.assertEqual(lineage['new_training_runs'], 0)
            self.assertEqual(lineage['source_checkpoint_replays'], 27)
        self.assertEqual(before, [hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in paths])
        archive = json.loads((ROOT/'notebooks/archive/dimension_coupling_initial/manifest.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT/archive['executed_notebook']).read_bytes()).hexdigest(),
                         archive['executed_notebook_sha256'])

    def test_executed_analysis_provenance_and_tables(self):
        n, c = self.n, self.c
        completion = json.loads((ANALYSIS/'completion.json').read_text())
        self.assertEqual(completion['kind'], 'read_only_fixed_update_reanalysis')
        self.assertEqual(completion['source_run'], str(SOURCE.relative_to(ROOT)))
        self.assertEqual(completion['new_actor_optimizer_steps'], 0)
        self.assertEqual(completion['new_checkpoint_evaluations'], 0)
        self.assertEqual(completion['fixed_update_spec_sha256'],
                         hashlib.sha256((ROOT/'configs/flow_am_fixed_update_analysis.json').read_bytes()).hexdigest())
        self.assertEqual(completion['source_sha256'],
                         hashlib.sha256((ROOT/'notebooks/02_flow_am_dimension_coupling.py').read_bytes()).hexdigest())
        notebook = nbformat.read(ROOT/'notebooks/02_flow_am_dimension_coupling.ipynb', as_version=4)
        code = [cell for cell in notebook.cells if cell.cell_type == 'code']
        self.assertEqual(len(code), 6)
        self.assertEqual(notebook.metadata.pagr.source_sha256, completion['source_sha256'])
        self.assertTrue(all(cell.execution_count is not None for cell in code))
        self.assertFalse(any(output.output_type == 'error' for cell in code
                             for output in cell.outputs))
        manifest = json.loads((ANALYSIS/'source-manifest.json').read_text())
        self.assertEqual(manifest['new_training_runs'], 0)
        for key, digest in manifest['input_sha256'].items():
            self.assertEqual(hashlib.sha256((SOURCE/key).read_bytes()).hexdigest(), digest)
        for key, digest in manifest['preserved_model_data_checkpoint_sha256'].items():
            task, seed, *rest = key.split('/')
            base = SOURCE/task/f'seed-{seed}'
            filename = ('/'.join(rest)+'.pt' if rest[0] in ('base', 'reward-0', 'reward-1',
                        'reward-2', 'reward-3', 'reward-4') else
                        'data.npz' if rest[0] == 'data' else
                        '/'.join(rest)+'.pt')
            self.assertEqual(hashlib.sha256((base/filename).read_bytes()).hexdigest(), digest)
        paired, summary = n['paired_updates'](self.curves, c)
        final = n['paired_final'](self.final, c)
        pd.testing.assert_frame_equal(pd.read_csv(ANALYSIS/'paired-updates-by-seed.csv'),
                                      paired, check_exact=False, rtol=0, atol=1e-14)
        pd.testing.assert_frame_equal(pd.read_csv(ANALYSIS/'paired-updates-summary.csv'),
                                      summary, check_exact=False, rtol=0, atol=1e-14)
        pd.testing.assert_frame_equal(pd.read_csv(ANALYSIS/'final-4096-paired.csv'),
                                      final, check_exact=False, rtol=0, atol=1e-14)
        self.assertEqual(set(completion['figures']),
                         {'fixed-updates-true-reward.png', 'fixed-updates-true-J.png',
                          'fixed-updates-path-KL.png', 'paired-fixed-update-differences.png'})
        self.assertTrue(all((ANALYSIS/name).stat().st_size > 1000
                            for name in completion['figures']))


if __name__ == '__main__':
    unittest.main()
