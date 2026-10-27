"""Propagation-aware robust AM updates. No manufactured critic-gradient errors.

Initial task deliberately reuses the previous navigation implementation unchanged.
All critic members see bootstrap samples of one finite scalar-observation dataset.
"""
from pathlib import Path
import sys, json, copy, hashlib, time, math
from dataclasses import asdict, dataclass
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'am_gradient_reliability_toys'))
import toy_core as task
torch.set_num_threads(1)

@dataclass
class Settings:
    members:int=5
    rho:float=.5
    updates:int=500
    lr:float=.002
    beta2:float=.99
    momentum:float=0.
    epsilon:float=.01
    max_update_norm:float=.3
    batch:int=64
    final_paths:int=4096
    eval_every:int=100

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fit_ensemble(cfg,seed,settings,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    spec=dict(config=asdict(cfg),settings=asdict(settings),seed=seed)
    data_rng=task.rng(seed+200000)
    x=task.offline_positions(cfg.critic_train,cfg,data_rng)
    y=task.observed_terminal_cost(x,cfg,data_rng)
    xv=task.offline_positions(cfg.critic_valid,cfg,data_rng)
    yv=task.observed_terminal_cost(xv,cfg,data_rng)
    np.savez_compressed(folder/'data.npz',x=x.numpy(),y=y.numpy(),xvalid=xv.numpy(),yvalid=yv.numpy())
    members=[];records=[]
    for m in range(settings.members):
        path=folder/f'critic_{m}.pt'
        torch.manual_seed(seed+100000+m*1009)
        critic=task.TerminalCritic(cfg.critic_width)
        if path.exists():
            save=torch.load(path,weights_only=True)
            assert save['config']==asdict(cfg) and save['seed']==seed
            critic.load_state_dict(save['state_dict']);info=save['fit']
        else:
            rg=task.rng(seed+900000+m*1013)
            # Member zero uses the complete dataset; all members use identical
            # scalar observations, never extra queries or true gradient labels.
            boot=torch.arange(len(x)) if m==0 else torch.randint(len(x),(len(x),),generator=rg)
            opt=torch.optim.Adam(critic.parameters(),lr=cfg.critic_lr)
            best=float('inf');history=[]
            for k in range(1,cfg.critic_updates+1):
                ix=boot[torch.randint(len(boot),(128,),generator=rg)]
                loss=((critic(x[ix])-y[ix])/20.).square().mean()
                opt.zero_grad();loss.backward();opt.step()
                if k%50==0 or k==1:
                    with torch.no_grad():v=(critic(xv)-yv).square().mean().item()
                    history.append(dict(step=k,noisy_validation_mse=v))
                    if v<best:best=v;state=copy.deepcopy(critic.state_dict());best_step=k
            critic.load_state_dict(state)
            info=dict(best_step=best_step,noisy_validation_mse=best,history=history)
            torch.save(dict(state_dict=state,config=asdict(cfg),seed=seed,member=m,fit=info),path)
        critic.eval()
        for p in critic.parameters():p.requires_grad_(False)
        members.append(critic);records.append(info)
    (folder/'fit.json').write_text(json.dumps(records,indent=2))
    return members

def flat_grad(loss,actor,retain_graph=False):
    return torch.cat([g.flatten() for g in torch.autograd.grad(loss,actor.parameters(),retain_graph=retain_graph)])

@torch.no_grad()
def apply_step(actor,delta):
    offset=0
    for p in actor.parameters():
        p.add_(delta[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()

def member_gradients(actor,paths,zs,cfg):
    batch=paths.shape[0]
    tt=(torch.arange(cfg.steps)*cfg.dt)[None,:,None].expand(batch,-1,-1)
    pred=actor(paths[:,:-1].reshape(-1,paths.shape[-1]),tt.reshape(-1,1)).reshape(batch,cfg.steps,-1)
    out=[]
    for m,z in enumerate(zs):
        target=task.lean_targets(paths,z,cfg)
        loss=.5*(pred-target).square().sum(-1).mean()
        out.append(flat_grad(loss,actor,retain_graph=m+1<len(zs)))
    return torch.stack(out)

def robust_residual(mean,deviations,rho):
    """prox_{rho sqrt(v^T C v)}(mean), C = deviations.T @ deviations.

    Moreau identity = mean minus Euclidean projection onto the covariance
    ellipsoid. The projection is solved in the <= M-1-dimensional eigenspace.
    No P x P covariance matrix is constructed. rho=0 is exactly identity.
    """
    if rho==0:return mean.clone(),dict(rank=0,removed=0.)
    # Double precision prevents numerical directions of an M-row rank M-1
    # matrix from being mistaken for real uncertainty directions.
    a=deviations.double();mu=mean.double()
    eig,vec=torch.linalg.eigh(a@a.T)
    use=eig>max(eig.max().item()*1e-9,1e-14)
    eig=eig[use];vec=vec[:,use]
    if len(eig)==0:return mean.clone(),dict(rank=0,removed=0.)
    basis=a.T@vec/eig.sqrt()
    coeff=basis.T@mu
    if ((coeff.square()/eig).sum()<=rho*rho).item():
        projection=coeff
    else:
        lo=0.;hi=max(float((eig*coeff.square()).sum().sqrt()/rho),1e-12)
        for _ in range(45):
            tau=(lo+hi)/2
            norm=(eig*coeff.square()/(eig+tau).square()).sum().item()
            if norm>rho*rho:lo=tau
            else:hi=tau
        projection=eig*coeff/(eig+hi)
    removed=basis@projection
    return (mu-removed).to(mean.dtype),dict(rank=len(eig),removed=removed.norm().item())

def proposed_step(qs,second,settings,method,first=None,iteration=1):
    raw_mean=qs.mean(0)
    if settings.momentum>0:
        # One shared clipping factor preserves cross-member differences, and
        # matches native Adam applied to the ensemble-mean gradient.
        qs=qs*min(1.,5./(raw_mean.norm().item()+1e-6))
    raw_mean=qs.mean(0)
    second.mul_(settings.beta2).addcmul_(raw_mean,raw_mean,value=1-settings.beta2)
    if settings.momentum>0:
        assert first is not None
        first.mul_(settings.momentum).add_(qs,alpha=1-settings.momentum)
        qs=first/(1-settings.momentum**iteration)
        h=(second/(1-settings.beta2**iteration)).sqrt()+settings.epsilon
    else:h=second.sqrt()+settings.epsilon
    # The robust proximal problem is solved BEFORE applying the update.
    # With momentum it protects the ensemble's accumulated first moment, not
    # necessarily the current true AM gradient. No current-loss descent claim.
    mean=qs.mean(0)
    scale=h.rsqrt()
    mu=mean*scale
    deviations=(qs-mean)*scale/math.sqrt(max(len(qs)-1,1))
    residual,info=robust_residual(mu,deviations,settings.rho)
    plain=-settings.lr*scale*mu
    robust=-settings.lr*scale*residual
    if method=='robust':delta=robust
    elif method=='norm_matched':delta=plain*(robust.norm()/plain.norm().clamp_min(1e-12))
    else:delta=plain
    delta=delta*min(1.,settings.max_update_norm/max(delta.norm().item(),1e-12))
    info.update(step_norm=delta.norm().item(),plain_norm=plain.norm().item(),
                cosine=torch.nn.functional.cosine_similarity(delta,plain,dim=0).item(),
                relative_removed=(robust-plain).norm().item()/max(plain.norm().item(),1e-12))
    return delta,info

def train(cfg,seed,settings,members,method,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed+300000)
    actor=task.Controller(cfg.actor_width)
    opt=torch.optim.Adam(actor.parameters(),lr=settings.lr)
    second=torch.zeros(sum(p.numel() for p in actor.parameters()))
    first=None
    rg=task.rng(seed+400000)
    records=[];updates=[];started=time.monotonic()
    for k in range(settings.updates+1):
        if k%settings.eval_every==0 or k==settings.updates:
            result=task.evaluate(actor,cfg,seed+500000,n=1024)
            records.append(dict(step=k,**result))
            if k in [0,settings.updates//2,settings.updates]:torch.save(actor.state_dict(),folder/f'actor_{k}.pt')
        if k==settings.updates:break
        paths,_=task.rollout(actor,cfg,settings.batch,rg)
        if method in ['oracle','oracle_adam']:zs=[task.true_terminal_gradient(paths[:,-1],cfg)]
        else:zs=[task.raw_gradient(c,paths[:,-1]) for c in members]
        if method in ['single_adam','single_rms']:zs=zs[:1]
        if method=='terminal_weight':
            z=torch.stack(zs);mean=z.mean(0)
            variance=z.var(0,unbiased=True).sum(-1,keepdim=True)
            shrink=mean.square().sum(-1,keepdim=True)/(mean.square().sum(-1,keepdim=True)+settings.rho*variance+1e-6)
            zs=[mean*shrink]
        qs=member_gradients(actor,paths,zs,cfg)
        if method in ['single_adam','ensemble_adam','oracle_adam']:
            q=qs.mean(0);offset=0;opt.zero_grad()
            for p in actor.parameters():p.grad=q[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
            torch.nn.utils.clip_grad_norm_(actor.parameters(),5.)
            opt.step();info={}
        else:
            if first is None:first=torch.zeros_like(qs)
            delta,info=proposed_step(qs,second,settings,method,first,k+1)
            apply_step(actor,delta)
        if k%25==0:updates.append(dict(step=k,**info))
    elapsed=time.monotonic()-started
    final,arrays=task.evaluate(actor,cfg,seed+600000,n=settings.final_paths,keep_paths=True)
    fine=task.evaluate(actor,cfg,seed+700000,n=2048,steps=2*cfg.steps)
    torch.save(actor.state_dict(),folder/'actor.pt')
    torch.save(opt.state_dict(),folder/'optimizer.pt')
    np.savez_compressed(folder/'evaluation.npz',**arrays)
    result=dict(method=method,seed=seed,config=asdict(cfg),settings=asdict(settings),
                history=records,update_diagnostics=updates,final=final,fine_grid=fine,seconds=elapsed)
    (folder/'result.json').write_text(json.dumps(result,indent=2))
    print(f'{cfg.name} seed={seed} {method} rho={settings.rho}: {final["cost"]:.4f}, success={final["success"]:.3f}, {elapsed:.1f}s',flush=True)
    return result

def suite(cfg,seeds,settings,methods,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    spec=dict(config=asdict(cfg),settings=asdict(settings),methods=methods,seeds=seeds,
              source_sha256=sha(__file__),task_sha256=sha(task.__file__),environment=task.environment_manifest())
    if (folder/'manifest.json').exists():
        previous=json.loads((folder/'manifest.json').read_text())
        for key in ['config','settings','methods','seeds','source_sha256','task_sha256']:
            assert json.dumps(previous[key],sort_keys=True)==json.dumps(spec[key],sort_keys=True),f'Existing run differs: {key}; use a new output folder.'
    (folder/'manifest.json').write_text(json.dumps(spec,indent=2))
    # Save the source used for each development iteration; later algorithm edits
    # cannot silently redefine old experiment implementations.
    snapshot=folder/'source_snapshot.txt'
    if not snapshot.exists():snapshot.write_text(Path(__file__).read_text())
    task_snapshot=folder/'task_source_snapshot.txt'
    if not task_snapshot.exists():task_snapshot.write_text(Path(task.__file__).read_text())
    rows=[]
    for seed in seeds:
        base=folder/f'seed_{seed}'
        members=fit_ensemble(cfg,seed,settings,base/'critics')
        for method in methods:
            result=base/method/'result.json'
            if result.exists():row=json.loads(result.read_text())
            else:row=train(cfg,seed,settings,members,method,base/method)
            rows.append(row)
    (folder/'results.json').write_text(json.dumps(rows,indent=2))
    return rows

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--rho',type=float,default=.5)
    p.add_argument('--output',default='dev_baseline');p.add_argument('--seeds',type=int,nargs='+',default=[411,412])
    p.add_argument('--methods',nargs='+',default=['single_adam','ensemble_adam','ensemble_rms','robust','norm_matched','terminal_weight','oracle'])
    p.add_argument('--updates',type=int,default=500)
    a=p.parse_args()
    cfg=task.Config(name='single_obstacle',critic_train=256,critic_mc=1,obstacle_weight=55.)
    suite(cfg,a.seeds,Settings(rho=a.rho,updates=a.updates),a.methods,ROOT/a.output)
