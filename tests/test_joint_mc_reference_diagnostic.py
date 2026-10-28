"""Monte Carlo score, current-policy continuation, and paired-probe checks."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location(
    'joint_mc_diagnostic',ROOT/'notebooks/02b_joint_mc_reference_diagnostic.py')
D=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=D
SPEC.loader.exec_module(D)


class CurrentPolicyMCReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.c=D.N.Config(dimension=4,horizon=3,sigma_env=.5,
                         flow_steps=5,warmup_steps=2)
        cls.task=D.N.SparseReachTask(cls.c)
        cls.actor=D.N.ConditionalVelocity(cls.c)
        cls.base=D.N.ConditionalVelocity(cls.c)
        cls.y=torch.tensor([1.,.3,.2,-.1])
        cls.a=torch.zeros(4)

    def test_terminal_gaussian_score_matches_exact_policy_independent_gradient(self):
        c=self.c
        result=D.mc_current_policy_reference(self.task,self.actor,self.base,c,
                                             c.horizon-1,self.y,self.a,30000,101,256)
        exact_q,exact_grad=D.N.oracle_torch(
            self.task,torch.tensor([c.horizon-1]),self.y[None],self.a[None])
        self.assertLess(abs(result['q']-float(exact_q)),4*result['q_se']+.002)
        self.assertTrue(np.all(np.abs(result['gradient']-exact_grad[0].numpy())<
                               5*result['gradient_se']+.015))

    def test_multistep_score_uses_current_continuation_and_not_fixed_oracle(self):
        c=self.c
        count=6000
        def reference_rollout(actor,base,c,kk,state,z,flow_noise):
            action=self.task.reference_action(state,z)
            return torch.stack((action,action),1),None
        # Substituting pi_ref only within this test supplies an independent
        # analytic check of the score identity at nonterminal time.
        with patch.object(D.N,'rollout',side_effect=reference_rollout):
            with patch.object(D.N,'oracle_torch',side_effect=AssertionError('oracle leaked')):
                result=D.mc_current_policy_reference(
                    self.task,self.actor,self.base,c,0,self.y,self.a,count,102,256)
        exact_q,exact_grad=D.N.oracle_torch(
            self.task,torch.tensor([0]),self.y[None],self.a[None])
        self.assertLess(abs(result['q']-float(exact_q)),5*result['q_se']+.01)
        self.assertTrue(np.all(np.abs(result['gradient']-exact_grad[0].numpy())<
                               6*result['gradient_se']+.035))
        # The policy used after the fixed first action matters.
        actual=D.mc_current_policy_reference(
            self.task,self.actor,self.base,c,0,self.y,self.a,512,102,128)
        self.assertEqual(actual['gradient'].shape,(c.dimension,))
        self.assertTrue(np.isfinite(actual['gradient']).all())

    def test_paired_evaluation_identical_branches_cancel_exactly(self):
        actors=dict(before=self.actor,learned=self.actor,mc_reference=self.actor)
        result=D.paired_one_step_evaluation(self.task,self.base,self.c,actors,48,11)
        self.assertEqual(result['delta_success_reference_minus_learned'],0.)
        self.assertEqual(result['delta_success_se'],0.)
        self.assertEqual(result['success_before'],result['success_learned'])

    def test_saved_joint_snapshot_probe_does_not_mutate_source(self):
        run=ROOT/'runs/02-qam-td-formal-31c55defef'
        if not run.is_dir():self.skipTest('formal checkpoints unavailable')
        c=D.source_config(run);task=D.N.SparseReachTask(c)
        design=dict(environment_times=[0,7],states_per_time=1,
                    mc_samples_per_split=32,mc_chunk=16,paired_eval_episodes=24)
        checkpoint=run/'seed-49001/joint-B-N0.pt'
        digest=D.sha256(checkpoint)
        with tempfile.TemporaryDirectory() as folder:
            points,snapshot=D.diagnose_snapshot(task,run,49001,0,c,design,Path(folder))
            self.assertEqual(len(points),2)
            self.assertLess(snapshot['recurrence_max_error'],3e-5)
            self.assertEqual(snapshot['eval_episodes'],24)
            self.assertTrue((Path(folder)/snapshot['mechanism_npz']).is_file())
        self.assertEqual(digest,D.sha256(checkpoint))


if __name__=='__main__':unittest.main()
