"""Six-joint planar arm pose-generation SOC with noisy motor execution.

The terminal cost is a genuine forward-kinematics task, not a designed gradient
corruption. Exact expected cost/gradient are used for evaluation/oracle only.
The SDE time is pose-generation time, not physical robot dynamics.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
import reliability as rel
base=rel.task
tensor=base.tensor;rng=base.rng;raw_gradient=base.raw_gradient;environment_manifest=base.environment_manifest

@dataclass
class Config(base.Config):
    name:str='six_joint_reach'
    steps:int=32
    sigma:float=.7
    start:tuple=(0.,.2,-.1,.1,-.1,0.)
    goal:tuple=(1.4,1.3)
    links:tuple=(.8,.65,.55,.4,.35,.25)
    orientation:float=math.pi/2
    stiffness:float=1.
    terminal_weight:float=12.
    orientation_weight:float=2.
    posture_weight:float=.1
    outcome_std:float=.15
    success_radius:float=.35
    critic_train:int=1024
    critic_valid:int=512
    critic_mc:int=4
    critic_updates:int=2500
    critic_width:int=128
    obstacles:tuple=()

class TerminalCritic(nn.Module):
    def __init__(self,width=128):
        super().__init__();self.net=nn.Sequential(nn.Linear(6,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))
    def forward(self,x):return 20*self.net(x/2).squeeze(-1)

class Controller(nn.Module):
    def __init__(self,width=64):
        super().__init__();self.net=nn.Sequential(nn.Linear(7,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,6))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,t):
        if not torch.is_tensor(t):t=torch.full_like(x[...,:1],float(t))
        return self.net(torch.cat([x/2,2*t-1],dim=-1))

def forward_kinematics(q,cfg):
    angles=q.cumsum(-1);lengths=tensor(cfg.links)
    return torch.stack([(angles.cos()*lengths).sum(-1),(angles.sin()*lengths).sum(-1)],dim=-1)

def physical_cost(q,cfg):
    ee=forward_kinematics(q,cfg)
    return cfg.terminal_weight*(ee-tensor(cfg.goal)).square().sum(-1)+cfg.orientation_weight*(1-torch.cos(q.sum(-1)-cfg.orientation))+cfg.posture_weight*(q-tensor(cfg.start)).square().sum(-1)

def observed_terminal_cost(x,cfg,generator):
    actual=x[:,None,:]+cfg.outcome_std*torch.randn(len(x),cfg.critic_mc,6,generator=generator)
    return physical_cost(actual,cfg).mean(1)

def true_terminal_value(q,cfg):
    angles=q.cumsum(-1);lengths=tensor(cfg.links);var=cfg.outcome_std**2
    decay=torch.exp(-.5*var*torch.arange(1,7,dtype=q.dtype))
    expected_x=(angles.cos()*lengths*decay).sum(-1)
    expected_y=(angles.sin()*lengths*decay).sum(-1)
    sq=torch.full(q.shape[:-1],float(lengths.square().sum()),dtype=q.dtype,device=q.device)
    for i in range(6):
        for j in range(i+1,6):sq+=2*lengths[i]*lengths[j]*torch.cos(angles[...,i]-angles[...,j])*math.exp(-.5*(j-i)*var)
    g=tensor(cfg.goal)
    result=cfg.terminal_weight*(sq-2*g[0]*expected_x-2*g[1]*expected_y+g.square().sum())
    result+=cfg.orientation_weight*(1-torch.cos(angles[...,-1]-cfg.orientation)*math.exp(-3*var))
    result+=cfg.posture_weight*((q-tensor(cfg.start)).square().sum(-1)+6*var)
    return result

def true_terminal_gradient(q,cfg):
    with torch.enable_grad():
        x=q.detach().requires_grad_(True)
        return torch.autograd.grad(true_terminal_value(x,cfg).sum(),x)[0].detach()

@torch.no_grad()
def rollout(actor,cfg,n,generator,steps=None):
    steps=cfg.steps if steps is None else steps;dt=1/steps
    xs=[tensor(cfg.start).expand(n,-1).clone()];us=[]
    for k in range(steps):
        u=actor(xs[-1],k*dt)
        xs.append(xs[-1]+dt*(-cfg.stiffness*(xs[-1]-tensor(cfg.start))+cfg.sigma*u)+cfg.sigma*math.sqrt(dt)*torch.randn(n,6,generator=generator))
        us.append(u)
    return torch.stack(xs,1),torch.stack(us,1)

def offline_positions(n,cfg,generator):
    coefficients=1.5*torch.randn(n,2,6,generator=generator)
    def behavior(x,t):return coefficients[:,0]+math.sin(math.pi*t)*coefficients[:,1]
    return rollout(behavior,cfg,n,generator)[0][:,-1]

@torch.no_grad()
def lean_targets(paths,terminal,cfg):
    a=terminal.clone();targets=[];F=1-cfg.dt*cfg.stiffness
    for k in range(cfg.steps-1,-1,-1):targets.append(-cfg.sigma*a);a=a*F
    return torch.stack(targets[::-1],1)

@torch.no_grad()
def evaluate(actor,cfg,seed,n=1024,steps=None,keep_paths=False):
    paths,controls=rollout(actor,cfg,n,rng(seed),steps);dt=1/controls.shape[1]
    energy=.5*controls.square().sum(-1).sum(-1)*dt;terminal=true_terminal_value(paths[:,-1],cfg);cost=energy+terminal
    ee=forward_kinematics(paths[:,-1],cfg);distance=(ee-tensor(cfg.goal)).norm(dim=-1)
    orientation=torch.atan2(torch.sin(paths[:,-1].sum(-1)-cfg.orientation),torch.cos(paths[:,-1].sum(-1)-cfg.orientation)).abs()
    success=(distance<cfg.success_radius)&(orientation<.5)
    result=dict(cost=cost.mean().item(),cost_mc_se=cost.std().item()/math.sqrt(n),terminal=terminal.mean().item(),
                energy=energy.mean().item(),running=0.,success=success.float().mean().item(),collision=0.,distance=distance.mean().item(),orientation_error=orientation.mean().item())
    if keep_paths:return result,dict(paths=paths.numpy(),controls=controls.numpy(),per_path_cost=cost.numpy(),end_effector=ee.numpy(),success=success.numpy())
    return result
