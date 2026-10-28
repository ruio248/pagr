"""Small, independent checks; never replays a formal training run."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('point_obstacle_notebook',
    ROOT/'notebooks/02_qam_point_obstacle.py')
M=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=M
SPEC.loader.exec_module(M)


class PointObstacleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        from dataclasses import replace
        cls.c=replace(M.Config(),horizon=3,flow_steps=2,warmup_steps=1,
                      critic_members=2,reference_grid=9,reference_policy_samples=2,
                      reference_transition_samples=8,reference_gradient_samples=32)
        cls.task=M.PointObstacle(cls.c)

    def test_segment_collision_priority_and_no_clamp(self):
        p=np.array([[.30,.50],[.50,.32],[.85,.50],[.98,.60]])
        pn=np.array([[.70,.50],[.50,.32],[.85,.50],[1.03,.60]])
        moved,r,done,status=self.task.outcome(0,p,pn)
        self.assertEqual(list(status),['collision','moving','success','collision'])
        self.assertEqual(list(r),[-1.,0.,1.,-1.])
        self.assertTrue(done[[0,2,3]].all())
        self.assertFalse(done[1])
        self.assertEqual(float(moved[3,0]),1.03)

    def test_horizon_mask_and_transition_consistency(self):
        p=np.array([[.12,.40],[.85,.50]])
        a=np.zeros((2,2));noise=np.zeros((2,2))
        pn,r,done,status=self.task.transition(self.c.horizon-1,p,a,noise)
        self.assertEqual(list(status),['timeout','success'])
        self.assertTrue(done.all())
        self.assertEqual(list(r),[0.,1.])
        same=self.task.outcome(self.c.horizon-1,p,pn)
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip((pn,r,done,status),same)))

    def test_tanh_chain_and_score_against_small_fd(self):
        c=self.c;task=self.task
        p=np.array([[.97,.60]])
        a=np.array([[.7,0.]])
        quad=M.normal_quadrature(16384)
        table=np.zeros((3,3))
        q,g=M.integral(task,c.horizon-1,p,a,table,quad,True)
        step=.03
        plus=M.integral(task,c.horizon-1,p,a+[[step,0]],table,quad)[0]
        minus=M.integral(task,c.horizon-1,p,a-[[step,0]],table,quad)[0]
        self.assertLess(abs(float(g[0,0]-(plus-minus)/(2*step))),.035)
        self.assertTrue(np.isfinite(q).all())

    def test_data_episode_split_and_route_coverage(self):
        c=self.c
        train,summary=M.collect_episodes(self.task,300,100)
        valid,_=M.collect_episodes(self.task,50,200)
        self.assertEqual(summary['episodes'],300)
        self.assertTrue(np.all(train['kn']==train['k']+1))
        self.assertEqual(set(np.unique(train['route'])),{0,1,2})
        self.assertTrue(np.all(train['episode_id']<300))
        self.assertTrue(np.all(valid['episode_id']+300>=300))

    def test_conditional_flow_and_current_actor_td(self):
        c=self.c
        actor=M.core.ConditionalVelocity(c)
        x=torch.zeros((2,2));k=torch.tensor([0,2]);p=torch.tensor([[.1,.4],[.9,.6]])
        values=actor(k,p,x,.5)
        self.assertFalse(torch.equal(values[0],values[1]))
        data,_=M.collect_episodes(self.task,8,201)
        train=M.as_torch(data)
        critic=M.core.Critic(c);target=M.core.Critic(c)
        opt=torch.optim.Adam(critic.parameters(),lr=c.critic_lr)
        with patch.object(M,'sample_actor',wraps=M.sample_actor) as sampled:
            loss,_=M.td_step(critic,target,opt,train,torch.arange(5),
                             M.core.generator(77),c,actor,actor)
            self.assertEqual(sampled.call_count,1)
            self.assertIs(sampled.call_args.args[0],actor)
            self.assertTrue(np.isfinite(loss))

    def test_policy_expectation_not_max_and_hash(self):
        c=self.c;actor=M.core.ConditionalVelocity(c)
        base=M.core.ConditionalVelocity(c)
        table=M.PolicyTable(self.task,actor,base,9,2,8,32)
        self.assertEqual(table.values.shape,(c.horizon+1,9,9))
        self.assertEqual(table.action_calls,c.horizon*9*9*2)
        p=table.nodes[10:11]
        z,flow=M.policy_noise(c,2,216)
        aa=[M.sample_actor(actor,base,c,c.horizon-1,p,z[i:i+1],flow[i:i+1])
            for i in range(2)]
        qs=[M.integral(self.task,c.horizon-1,p,a,table.values[c.horizon],table.quad)[0]
            for a in aa]
        self.assertAlmostEqual(float(table.values[c.horizon-1].reshape(-1)[10]),
                               float(np.mean(qs)),places=6)
        self.assertLessEqual(float(table.values[c.horizon-1].reshape(-1)[10]),max(qs))
        table.assert_current(actor,base)
        with torch.no_grad():next(actor.parameters()).add_(1e-3)
        with self.assertRaises(ValueError):table.assert_current(actor,base)

    def test_identity_transform_is_not_new_method(self):
        x=torch.tensor([1.,2.])
        self.assertIs(M.gradient_transform(x,None,'terminal'),x)
        self.assertIs(M.gradient_transform(None,x,'propagated'),x)
        with self.assertRaises(ValueError):M.gradient_transform(x,None,'unknown')


if __name__=='__main__':unittest.main()
