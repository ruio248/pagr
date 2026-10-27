import copy
import json
import unittest
import numpy as np
import torch
from pagr import runtime as rt


class ResearchTests(unittest.TestCase):
    def test_asset_integrity(self):
        self.assertGreater(rt.verify()["verified_files"],200)

    def test_all_results_and_warnings_retained(self):
        data=rt.summary()
        self.assertEqual(data["training_runs"],42)
        for task in rt.TASKS:
            self.assertEqual(set(data["tasks"][task]),set(rt.METHODS))
        audit=json.loads((rt.ASSETS/"reference/delivery_audit.json").read_text())
        self.assertFalse(audit["passed"])
        self.assertTrue(audit["passed_required_integrity_checks"])
        self.assertEqual(len(audit["solver_warnings"]),6)
        self.assertEqual(audit["certified_selected_empirical_nonincrease"],300)

    def test_lean_no_oracle_and_guard(self):
        la,aa,sn,vg=rt.backend()
        for name in rt.TASKS:
            task,cfg,critics,c,_=la.load_inputs(name,18001)
            torch.manual_seed(927)
            actor=task.Controller(cfg.actor_width)
            paths,_=task.rollout(actor,cfg,8,task.rng(84))
            terminal=torch.stack([task.raw_gradient(m,paths[:,-1]) for m in critics])
            _,targets,_=la.matching_gradients(actor,paths,terminal,task,cfg)
            expected=torch.stack([task.lean_targets(paths,z,cfg) for z in terminal])
            self.assertTrue(torch.allclose(targets,expected,atol=2e-6,rtol=2e-6))
            saved={k:getattr(task,k) for k in ("true_terminal_value","true_terminal_gradient","observed_terminal_cost")}
            pred,full=la.base.prediction,la.base.matching_gradients
            def forbidden(*args,**kwargs):raise AssertionError("Oracle/full adjoint invoked in training")
            def only_lean(*args,**kwargs):
                self.assertIs(kwargs.get("spatial_jacobian"),False)
                return pred(*args,**kwargs)
            for key in saved:setattr(task,key,forbidden)
            la.base.prediction,la.base.matching_gradients=only_lean,forbidden
            try:
                before=vg.vector(actor)
                vg.proposal(actor,torch.optim.Adam(actor.parameters(),lr=.002),paths,critics,task,cfg)
                proposed=vg.vector(actor)
                info=vg.guard(actor,before,proposed,task,cfg,critics,79,vg.Options(guard_paths=16),la.Adversary(c,.3),True)
                self.assertLessEqual(min(info["scores"]),0.)
            finally:
                for key,value in saved.items():setattr(task,key,value)
                la.base.prediction,la.base.matching_gradients=pred,full

    def test_all_estimators_load(self):
        la,aa,sn,_=rt.backend()
        for task in rt.TASKS:
            for seed in rt.SEEDS:
                module,cfg,_,_,_=la.load_inputs(task,seed)
                x=module.tensor(cfg.start)[None]
                for method in ("smooth","residual"):
                    critic=aa.load(task,seed,method)[2]
                    self.assertTrue(torch.isfinite(module.raw_gradient(critic,x)).all())
                for critic in sn.load(task,seed)[2]:
                    self.assertTrue(torch.isfinite(module.raw_gradient(critic,x)).all())

    def test_silu_initial_parameter_match(self):
        la,_,sn,_=rt.backend()
        for name in rt.TASKS:
            task,cfg,_,_,_=la.load_inputs(name,18001)
            torch.manual_seed(11);a=task.TerminalCritic(cfg.critic_width)
            torch.manual_seed(11);b=sn.model(task,cfg)
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())))


if __name__=="__main__":unittest.main()
