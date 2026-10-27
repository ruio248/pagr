"""One fixed smooth-activation ablation, not a new uncertainty algorithm.

Same scalar observations, parameter count, initialization, bootstrap/minibatch
stream, optimizer, critic update budget and noisy validation selection as ReLU.
Only the two ReLU activations are changed to SiLU. No tuning activation/width.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import copy
import json
import time
import numpy as np
import torch
import anchored_am as aa
la=aa.la
ROOT=aa.ROOT


def model(task,cfg):
    critic=task.TerminalCritic(cfg.critic_width)
    count=0
    for i,layer in enumerate(critic.net):
        if isinstance(layer,torch.nn.ReLU):
            critic.net[i]=torch.nn.SiLU();count+=1
    assert count==2
    return critic


def fit(name,seed):
    task,cfg,old,_,manifest=la.load_inputs(name,seed)
    data=np.load(la.PREVIOUS/'pilot'/name/f'seed_{seed}'/'critics/data.npz')
    x,y,xv,yv=[torch.tensor(data[k]) for k in ('x','y','xvalid','yvalid')]
    folder=ROOT/'models'/name/f'seed_{seed}'/'silu'
    folder.mkdir(parents=True,exist_ok=True)
    records=[];start=time.monotonic()
    for m in range(5):
        path=folder/f'critic_{m}.pt'
        if path.exists():
            records.append(torch.load(path,weights_only=True)['fit']);continue
        torch.manual_seed(seed+100000+m*1009)
        critic=model(task,cfg)
        rg=task.rng(seed+900000+m*1013)
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
        info=dict(best_step=best_step,noisy_validation_mse=best,history=history)
        torch.save(dict(state_dict=state,config=asdict(cfg),seed=seed,member=m,fit=info),path)
        records.append(info)
    members=load(name,seed)[2]
    with torch.no_grad():
        old_mse=(aa.MeanCritic(old)(xv)-yv).square().mean().item()
        new_mse=(aa.MeanCritic(members)(xv)-yv).square().mean().item()
    result=dict(members=records,input_manifest=manifest,seconds=time.monotonic()-start,
                original_validation_mse=old_mse,validation_mse=new_mse,
                data_queries=0,gradient_labels=0,critic_updates_per_member=cfg.critic_updates)
    (folder/'fit.json').write_text(json.dumps(result,indent=2))
    print('FIT SiLU',name,seed,old_mse,'->',new_mse,flush=True)
    return result


def load(name,seed):
    task,cfg,_,c,manifest=la.load_inputs(name,seed)
    members=[]
    for m in range(5):
        critic=model(task,cfg)
        critic.load_state_dict(torch.load(ROOT/'models'/name/f'seed_{seed}'/'silu'/f'critic_{m}.pt',weights_only=True)['state_dict'])
        critic.eval()
        for p in critic.parameters():p.requires_grad_(False)
        members.append(critic)
    return task,cfg,members,c,manifest


def freeze():
    p=dict(tasks=['navigation','arm'],seeds=[18001,18002,18003],activation='SiLU; fixed, no activation sweep',
           matched='Original data, network widths/parameter counts, initialization, bootstrap and minibatch streams, optimizer/lr, update budget, validation-selected critic checkpoints.',
           selection='No true value, true derivative or control return used to select critic checkpoint.',
           actor='Unchanged lean AM+Adam, 1000 updates, batch64, fixed final checkpoint.',
           motivation='Diagnose whether differentiable scalar-estimation prior repairs labels before any uncertainty penalty.',
           gate='Primary task arm: >=1% improvement AND 3/3 wins vs original AM+Adam. Secondary navigation; no claim of a novel HJBI algorithm.',
           stage='Development after negative finite-guard and kernel diagnostics; reused seeds, not independent confirmation.',
           source_sha256=la.rh.rel.sha(__file__))
    path=ROOT/'silu_protocol.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:path.write_text(json.dumps(p,indent=2))
    return p


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=['navigation','arm']);a=p.parse_args()
    protocol=freeze()
    for seed in protocol['seeds']:
        folder=ROOT/'models'/a.task/f'seed_{seed}'/'silu'
        if not (folder/'fit.json').exists():fit(a.task,seed)
        task,cfg,members,c,_=load(a.task,seed)
        out=ROOT/'results'/a.task/f'seed_{seed}'/'silu'
        if not (out/'result.json').exists():
            la.train(a.task,cfg,seed,members,c,la.Options(method='am_adam'),out)
