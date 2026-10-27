"""Data-anchored terminal-gradient repair; unchanged lean AM + Adam actor.

This is a critic/label-estimation experiment, NOT a new optimal-control PDE.
The RBF smoothness prior is explicit. Hyperparameters use only the original
512 noisy validation values, never true values, true derivatives or returns.
Two fixed candidates: direct smooth scalar fit and smooth NN residual repair.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'am_terminal_consistent_lean'))
import lean_am as la


class MeanCritic(torch.nn.Module):
    def __init__(self, critics):
        super().__init__()
        self.members=torch.nn.ModuleList(critics)
    def forward(self,x):
        return torch.stack([c(x) for c in self.members]).mean(0)


class SmoothCritic(torch.nn.Module):
    def __init__(self,model,base=None):
        super().__init__()
        for key in ('center','scale','alpha','offset'):
            self.register_buffer(key,torch.as_tensor(model[key],dtype=torch.float64))
        self.register_buffer('train_x',torch.as_tensor(model['train'],dtype=torch.float64))
        self.ell=float(model['ell'])
        self.base=base
    def forward(self,x):
        v=(x.double()-self.center)/self.scale
        d2=(v.square().sum(-1,keepdim=True)+self.train_x.square().sum(-1)[None]-2*v@self.train_x.T).clamp_min(0)
        out=self.offset+torch.exp(-d2/(2*self.ell**2))@self.alpha
        if self.base is not None:out=out+self.base(x).double()
        return out.to(x.dtype)


def kernel(x,z,ell):
    d2=(x.square().sum(-1,keepdim=True)+z.square().sum(-1)[None]-2*x@z.T).clamp_min(0)
    return torch.exp(-d2/(2*ell**2))


@torch.no_grad()
def fit(name,seed,method):
    assert method in ('smooth','residual')
    task,cfg,critics,_,manifest=la.load_inputs(name,seed)
    data=np.load(la.PREVIOUS/'pilot'/name/f'seed_{seed}'/'critics/data.npz')
    x,y,xv,yv=[torch.tensor(data[k]).double() for k in ('x','y','xvalid','yvalid')]
    base=MeanCritic(critics)
    bx,bv=base(x.float()).double(),base(xv.float()).double()
    center=x.mean(0);scale=x.std(0).clamp_min(.1)
    xz,xvz=(x-center)/scale,(xv-center)/scale
    target=y-bx if method=='residual' else y
    offset=target.mean()
    rows=[];best=None;start=time.monotonic()
    for ell in (.5,1.,2.,4.):
        k=kernel(xz,xz,ell);kv=kernel(xvz,xz,ell)
        for ridge in (.001,.01,.1,1.,10.):
            chol=torch.linalg.cholesky(k+ridge*torch.eye(len(x),dtype=x.dtype))
            alpha=torch.cholesky_solve((target-offset)[:,None],chol).squeeze(-1)
            pv=offset+kv@alpha+(bv if method=='residual' else 0)
            mse=(pv-yv).square().mean().item()
            rows.append(dict(ell=ell,ridge=ridge,validation_mse=mse))
            if best is None or mse<best['validation_mse']:
                best=dict(ell=ell,ridge=ridge,validation_mse=mse,
                          center=center,scale=scale,train=xz,alpha=alpha,offset=offset)
    folder=ROOT/'models'/name/f'seed_{seed}'/method
    folder.mkdir(parents=True,exist_ok=True)
    torch.save(best,folder/'model.pt')
    info=dict(task=name,seed=seed,method=method,grid=rows,
              chosen={k:best[k] for k in ('ell','ridge','validation_mse')},
              original_validation_mse=(bv-yv).square().mean().item(),
              training_points=len(x),validation_points=len(xv),
              input_manifest=manifest,seconds=time.monotonic()-start,
              real_new_terminal_queries=0,gradient_labels_used=0)
    (folder/'fit.json').write_text(json.dumps(info,indent=2))
    print('FIT',name,seed,method,info['chosen'],'original',info['original_validation_mse'],flush=True)
    return info


def load(name,seed,method):
    task,cfg,critics,c,manifest=la.load_inputs(name,seed)
    model=torch.load(ROOT/'models'/name/f'seed_{seed}'/method/'model.pt',weights_only=True)
    critic=SmoothCritic(model,MeanCritic(critics) if method=='residual' else None)
    critic.eval()
    for p in critic.parameters():p.requires_grad_(False)
    return task,cfg,critic,c,manifest


def train(name,seed,method,folder=None):
    task,cfg,critic,c,_=load(name,seed,method)
    folder=Path(folder) if folder else ROOT/'results'/name/f'seed_{seed}'/method
    # The original lean training loop and Adam path are reused unmodified.
    result=la.train(name,cfg,seed,[critic],c,la.Options(method='am_adam'),folder)
    result.update(terminal_estimator=method,critic_model_sha256=la.rh.rel.sha(ROOT/'models'/name/f'seed_{seed}'/method/'model.pt'))
    (folder/'result.json').write_text(json.dumps(result,indent=2))
    return result


def freeze():
    p=dict(tasks=['navigation','arm'],seeds=[18001,18002,18003],methods=['smooth','residual'],
           length_scales=[.5,1.,2.,4.],ridges=[.001,.01,.1,1.,10.],
           selection='Only original noisy scalar validation MSE. No return/true-cost/gradient tuning.',
           primary='smooth vs original lean AM+Adam. Residual repair is a second prespecified candidate, not a hidden sweep.',
           gate='At least 1% mean cost improvement and 3/3 wins vs original AM on either task; independent new-seed confirmation required before broad claims.',
           actor=asdict(la.Options(method='am_adam')),
           interpretation='Same framework/optimizer and data, different terminal gradient estimator. Gains cannot be attributed to HJBI or to uncertainty suppression without further ablations.',
           prior='Stationary Gaussian RBF smoothness in data-standardized coordinates; NOT learned or calibrated gradient-error coverage.',
           provenance={str(Path(__file__).resolve()):la.rh.rel.sha(__file__),str(Path(la.__file__)):la.rh.rel.sha(la.__file__)})
    path=ROOT/'protocol.json'
    if path.exists():
        prior=json.loads(path.read_text())
        if prior!=p:
            old=dict(prior);old.pop('provenance');new=dict(p);new.pop('provenance')
            assert old==new and not list((ROOT/'results').glob('*/*/*/result.json'))
            # Record the pre-training PyTorch buffer-name collision fix.
            (ROOT/'pretraining_protocol_revision.json').write_text(json.dumps(dict(previous=prior,
                reason='Rename train buffer to train_x; train is a reserved nn.Module method. No actor training or parameter selection changed.'),indent=2))
            path.write_text(json.dumps(p,indent=2))
    else:path.write_text(json.dumps(p,indent=2))
    return p


def audit():
    rows={}
    for name in ('navigation','arm'):
        for method in ('smooth','residual'):
            task,cfg,critic,c,_=load(name,18001,method)
            torch.manual_seed(671)
            actor=task.Controller(cfg.actor_width)
            paths,_=task.rollout(actor,cfg,8,task.rng(642))
            x=paths[:,-1]
            z=task.raw_gradient(critic,x)
            # Finite differences check only implementation, not truth alignment.
            v=torch.randn_like(x);v=v/v.norm(dim=-1,keepdim=True)
            h=.003 if method=='residual' else .001
            numeric=(critic(x+h*v)-critic(x-h*v))/(2*h)
            analytic=(z*v).sum(-1)
            error=(numeric-analytic).norm()/analytic.norm().clamp_min(1e-10)
            # NN residual includes ReLU kinks: separate exact smooth check below.
            if method=='smooth':assert error<.005
            saved={k:getattr(task,k) for k in ('true_terminal_value','true_terminal_gradient','observed_terminal_cost')}
            pred,full=la.base.prediction,la.base.matching_gradients
            def forbidden(*a,**kw):raise AssertionError('Forbidden true gradient/value or full adjoint')
            def lean_pred(*a,**kw):
                assert kw.get('spatial_jacobian') is False
                return pred(*a,**kw)
            for k in saved:setattr(task,k,forbidden)
            la.base.prediction,la.base.matching_gradients=lean_pred,forbidden
            try:
                la.update(actor,torch.optim.Adam(actor.parameters(),lr=.002),paths,[critic],task,cfg,
                          la.Options(method='am_adam'),{},None,0)
            finally:
                for k,vv in saved.items():setattr(task,k,vv)
                la.base.prediction,la.base.matching_gradients=pred,full
            rows[name+'_'+method]=dict(no_truth_no_full_adjoint=True,directional_derivative_relative_error=float(error))
    out=dict(passed=True,checks=rows)
    (ROOT/'audit.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2),flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['navigation','arm'])
    p.add_argument('--audit',action='store_true');a=p.parse_args()
    protocol=freeze()
    if a.task:
        for seed in protocol['seeds']:
            for method in protocol['methods']:
                folder=ROOT/'models'/a.task/f'seed_{seed}'/method
                if not (folder/'fit.json').exists():fit(a.task,seed,method)
                out=ROOT/'results'/a.task/f'seed_{seed}'/method
                if not (out/'result.json').exists():train(a.task,seed,method,out)
    if a.audit:audit()
