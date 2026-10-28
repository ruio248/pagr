"""Independent checks for the multi-step TD/QAM notebook; no formal training."""
import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('qam_notebook',ROOT/'notebooks/02_qam_td_terminal_error.py')
N=importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name]=N
SPEC.loader.exec_module(N)


class MultiStepQAMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.c=N.Config(flow_steps=5,warmup_steps=2,actor_batch=6,critic_members=2,
                       critic_next_samples=2,critic_batch=12)
        cls.task=N.SparseReachTask(cls.c)

    def test_environment_markov_time_reward_and_full_rank(self):
        c=self.c;task=self.task
        self.assertEqual(np.linalg.matrix_rank(task.B64),c.dimension)
        self.assertLess(np.linalg.cond(task.B64),4)
        for j in range(c.dimension):
            a=torch.zeros(1,c.dimension);a[0,j]=1
            yn,_,_=task.step(0,torch.zeros_like(a),a,torch.zeros_like(a))
            self.assertGreater(float(yn.norm()),.49)
        data=N.collect_episodes(task,64,100)
        self.assertEqual(len(data['k']),64*c.horizon)
        self.assertTrue(bool((data['r'][data['k']<c.horizon-1]==0).all()))
        self.assertTrue(bool((data['done']==(data['k']==c.horizon-1)).all()))
        self.assertTrue(bool((data['kn']==data['k']+1).all()))
        for k in range(c.horizon):self.assertEqual(int((data['k']==k).sum()),64)

    def test_oracle_mc_bellman_fd_and_boundary(self):
        checks=N.verify_oracle(self.task,9876)
        self.assertIn('finite_difference',set(checks.check))
        self.assertIn('bellman',set(checks.check))
        y=torch.zeros((1,self.c.dimension));a=torch.zeros_like(y)
        q,g=N.oracle_torch(self.task,torch.tensor([7]),y,a)
        self.assertGreater(float(q),0)
        self.assertLess(float(g.norm()),1e-8)
        # Different k, same y/a: finite-horizon value need not agree.
        q0=N.oracle_torch(self.task,torch.tensor([0]),y,a,False)
        self.assertNotAlmostEqual(float(q0),float(q),places=4)

    def test_td_terminal_mask_and_no_oracle_training(self):
        c=self.c;task=self.task
        data=N.collect_episodes(task,8,108)
        critic=N.Critic(c);target=N.Critic(c)
        optim=torch.optim.Adam(critic.parameters(),lr=c.critic_lr)
        terminal=(data['done']).nonzero().flatten()[:5]
        batch={key:value[terminal] for key,value in data.items()}
        result=N.td_target(task,target,batch,N.generator(7),c)
        self.assertTrue(torch.equal(result,batch['r']))
        indices=torch.arange(12)
        with patch.object(N,'oracle_torch',side_effect=AssertionError('oracle in TD')):
            loss,_=N.td_step(task,critic,target,optim,data,indices,N.generator(8),c)
        self.assertTrue(np.isfinite(loss))

    def test_conditioned_flow_vjp_and_learning_boundary(self):
        c=self.c;task=self.task
        base=N.ConditionalVelocity(c)
        k=torch.tensor([0,5]);y=torch.randn(2,c.dimension)
        x=torch.randn_like(y);vec=torch.randn_like(y)
        xleaf=x.detach().requires_grad_(True)
        direct=torch.autograd.grad((base(k,y,xleaf,.4)*vec).sum(),xleaf)[0]
        manual=base.input_vjp(k,y,x,.4,vec)
        self.assertTrue(torch.allclose(direct,manual,atol=1e-6))
        self.assertFalse(torch.allclose(base(k,y,x,.4),base(k,y+1,x,.4)))
        critic=N.Critic(c)
        endpoint=torch.randn(2,c.dimension)
        with patch.object(N,'oracle_torch',side_effect=AssertionError('learned called oracle')):
            label=N.terminal_label(task,critic,c,k,y,endpoint,'learned')
        self.assertEqual(label.shape,endpoint.shape)

    def test_common_path_identity_and_virtual_adam_isolation(self):
        c=self.c;task=self.task
        base=N.ConditionalVelocity(c)
        for p in base.parameters():p.requires_grad_(False)
        actor=N.ConditionalVelocity(c)
        critic=N.Critic(c)
        opt=torch.optim.Adam(actor.parameters(),lr=c.actor_lr)
        k=torch.tensor([0,1,4,7,2,3]);y=torch.randn(6,c.dimension)
        z,xi=N.paired_generation_noise(c,6,500)
        old=N.parameter_vector(actor).clone()
        with tempfile.TemporaryDirectory() as temp:
            row=N.mechanism_diagnostic(task,critic,base,actor,opt,c,k,y,z,xi,4,0,Path(temp))
            arr=np.load(Path(temp)/row['diagnostic_npz'])
            self.assertEqual(arr['matching_label_error'].shape,(6,c.flow_steps,c.dimension))
        self.assertLess(row['recurrence_max_error'],3e-5)
        self.assertLess(row['parameter_projection_max_error'],3e-5)
        self.assertTrue(torch.equal(old,N.parameter_vector(actor)))
        self.assertFalse(bool(opt.state))

    def test_paths_terminal_and_adjoint_are_detached(self):
        c=self.c;task=self.task
        base=N.ConditionalVelocity(c);actor=N.ConditionalVelocity(c);critic=N.Critic(c)
        k=torch.tensor([0,4]);y=torch.randn(2,c.dimension)
        z,xi=N.paired_generation_noise(c,2,399)
        paths,_=N.rollout(actor,base,c,k,y,z,xi)
        terminal=N.terminal_label(task,critic,c,k,y,paths[:,-1],'learned')
        adj=N.lean_adjoint(base,c,k,y,paths,terminal)
        self.assertFalse(paths.requires_grad)
        self.assertFalse(terminal.requires_grad)
        self.assertFalse(adj.requires_grad)
        loss=N.matching_loss(actor,base,c,k,y,paths,adj)
        self.assertTrue(loss.requires_grad)

    def test_joint_td_uses_actor_sampler_not_reference(self):
        c=self.c;task=self.task
        data=N.collect_episodes(task,3,601)
        batch={key:value[:6] for key,value in data.items()}
        target=N.Critic(c);base=N.ConditionalVelocity(c);actor=N.ConditionalVelocity(c)
        with patch.object(task,'reference_action',side_effect=AssertionError('pi_ref in joint')):
            with patch.object(N,'oracle_torch',side_effect=AssertionError('Qref in joint')):
                label=N.td_target(task,target,batch,N.generator(3),c,'actor',actor,base)
        self.assertEqual(label.shape,(6,))
        self.assertTrue(bool(torch.isfinite(label).all()))

    def test_paired_noise_and_fixed_counts(self):
        c=self.c
        a=N.paired_generation_noise(c,8,9)
        b=N.paired_generation_noise(c,8,9)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))
        self.assertEqual(c.flow_steps,5)
        self.assertEqual(c.actor_batch,6)
        full,_=N.load_config()
        self.assertEqual(tuple(full.curve_updates),(0,50,100,150,200,250,300))
        self.assertEqual(full.actor_updates,300)
        self.assertEqual(full.flow_steps,64)
        self.assertEqual(full.actor_batch,64)


if __name__=='__main__':unittest.main()
