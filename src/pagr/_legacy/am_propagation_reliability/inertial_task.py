"""Inertial navigation with acceleration controls, two obstacles and docking.

Four-dimensional state (position, velocity), two-dimensional controls/noise.
Natural stochastic endpoint observations; nonlinear quartic docking objective.
Offline data are endpoints of randomized behavior controls in the same dynamics.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
import reliability as rel
base=rel.task
tensor=base.tensor
rng=base.rng
environment_manifest=base.environment_manifest
raw_gradient=base.raw_gradient

@dataclass
class Config(base.Config):
    name:str='inertial_docking'
    steps:int=40
    horizon:float=2.0
    sigma:float=.7
    start:tuple=(-2.,0.,0.,0.)
    goal:tuple=(2.,0.,0.,0.)
    obstacles:tuple=((-0.65,.2,.4),(.65,-.2,.4))
    obstacle_weight:float=55.
    terminal_weight:float=8.
    velocity_weight:float=2.
    quartic_weight:float=.8
    outcome_std:float=.2
    velocity_outcome_std:float=.3
    success_radius:float=.65
    success_velocity:float=.8
    critic_train:int=512
    critic_valid:int=512
    critic_mc:int=4
    critic_updates:int=2200
    critic_width:int=128
    damping:float=.7
    stiffness:float=.8
    capture_reward:float=0.
    capture_position_scale:float=.4
    capture_velocity_scale:float=.5
    @property
    def dt(self):return self.horizon/self.steps

class TerminalCritic(nn.Module):
    def __init__(self,width=128):
        super().__init__();self.net=nn.Sequential(nn.Linear(4,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))
    def forward(self,x):return 20*self.net(x/tensor([3.,3.,2.,2.])).squeeze(-1)

class Controller(nn.Module):
    def __init__(self,width=64):
        super().__init__();self.net=nn.Sequential(nn.Linear(5,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,2))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,t):
        if not torch.is_tensor(t):t=torch.full_like(x[...,:1],float(t))
        return self.net(torch.cat([x/tensor([3.,3.,2.,2.]),t-1.],dim=-1))

def matrices(cfg,dt=None):
    dt=cfg.dt if dt is None else dt
    A=tensor([[0,0,1,0],[0,0,0,1],[-cfg.stiffness,0,-cfg.damping,0],[0,-cfg.stiffness,0,-cfg.damping]])
    S=cfg.sigma*tensor([[0,0],[0,0],[1,0],[0,1]])
    return A,torch.eye(4)+dt*A,S

def reference(t,cfg):
    s=t/cfg.horizon;disp=tensor(cfg.goal[:2])-tensor(cfg.start[:2])
    p=tensor(cfg.start[:2])+disp*(3*s*s-2*s*s*s)
    v=disp*(6*s-6*s*s)/cfg.horizon
    acc=disp*(6-12*s)/cfg.horizon**2
    # Optional pre-existing curved behavior prior, not a method-specific route.
    p=p+tensor([0.,cfg.base_arch*math.sin(math.pi*s)**2])
    v=v+tensor([0.,cfg.base_arch*math.pi/cfg.horizon*math.sin(2*math.pi*s)])
    acc=acc+tensor([0.,2*cfg.base_arch*(math.pi/cfg.horizon)**2*math.cos(2*math.pi*s)])
    return torch.cat([p,v]),torch.cat([v,acc])

def drift(x,t,cfg):
    A,_,_=matrices(cfg);m,mdot=reference(t,cfg)
    return mdot+(x-m)@A.T

@torch.no_grad()
def rollout(actor,cfg,n,generator,steps=None):
    steps=cfg.steps if steps is None else steps;dt=cfg.horizon/steps
    _,_,S=matrices(cfg)
    xs=[tensor(cfg.start).expand(n,-1).clone()];us=[]
    for k in range(steps):
        u=actor(xs[-1],k*dt);noise=torch.randn(n,2,generator=generator)
        xs.append(xs[-1]+dt*(drift(xs[-1],k*dt,cfg)+u@S.T)+math.sqrt(dt)*noise@S.T)
        us.append(u)
    return torch.stack(xs,1),torch.stack(us,1)

def offline_positions(n,cfg,generator):
    coefficients=torch.randn(n,3,2,generator=generator)*tensor([1.6,1.6])
    def behavior(x,t):
        phase=math.pi*t/cfg.horizon
        return coefficients[:,0]+coefficients[:,1]*math.sin(phase)+coefficients[:,2]*math.cos(phase)
    return rollout(behavior,cfg,n,generator)[0][:,-1]

def outcome_scales(cfg):return tensor([cfg.outcome_std,cfg.outcome_std,cfg.velocity_outcome_std,cfg.velocity_outcome_std])

def observed_terminal_cost(x,cfg,generator):
    y=x[:,None,:]+torch.randn(len(x),cfg.critic_mc,4,generator=generator)*outcome_scales(cfg)
    e=y-tensor(cfg.goal)
    cost=cfg.terminal_weight*e[...,:2].square().sum(-1)+cfg.velocity_weight*e[...,2:].square().sum(-1)
    cost+=cfg.quartic_weight*e[...,:2].pow(4).sum(-1)
    if cfg.capture_reward:
        scale=tensor([cfg.capture_position_scale]*2+[cfg.capture_velocity_scale]*2)
        cost+=cfg.capture_reward*(1-torch.exp(-.5*(e/scale).square().sum(-1)))
    return cost.mean(1)

def true_terminal_value(x,cfg):
    e=x-tensor(cfg.goal);var=outcome_scales(cfg).square()
    weights=tensor([cfg.terminal_weight]*2+[cfg.velocity_weight]*2)
    cost=((e.square()+var)*weights).sum(-1)
    cost+=cfg.quartic_weight*(e[...,:2].pow(4)+6*var[:2]*e[...,:2].square()+3*var[:2].square()).sum(-1)
    if cfg.capture_reward:
        s2=tensor([cfg.capture_position_scale**2]*2+[cfg.capture_velocity_scale**2]*2)
        capture=(s2/(s2+var)).sqrt().prod()*torch.exp(-.5*(e.square()/(s2+var)).sum(-1))
        cost+=cfg.capture_reward*(1-capture)
    return cost

def true_terminal_gradient(x,cfg):
    e=x-tensor(cfg.goal);var=outcome_scales(cfg).square()
    grad=2*e*tensor([cfg.terminal_weight]*2+[cfg.velocity_weight]*2)
    grad[...,:2]+=cfg.quartic_weight*(4*e[...,:2].pow(3)+12*var[:2]*e[...,:2])
    if cfg.capture_reward:
        s2=tensor([cfg.capture_position_scale**2]*2+[cfg.capture_velocity_scale**2]*2)
        capture=(s2/(s2+var)).sqrt().prod()*torch.exp(-.5*(e.square()/(s2+var)).sum(-1))
        grad+=cfg.capture_reward*capture[...,None]*e/(s2+var)
    return grad

def running_cost_and_gradient(x,cfg):
    cost=torch.zeros(x.shape[:-1]);grad=torch.zeros_like(x)
    for cx,cy,r in cfg.obstacles:
        d=x[...,:2]-tensor([cx,cy]);s=.85*r
        bump=cfg.obstacle_weight*torch.exp(-d.square().sum(-1)/(2*s*s))
        cost+=bump;grad[...,:2]-=bump[...,None]*d/(s*s)
    return cost,grad

@torch.no_grad()
def lean_targets(paths,terminal,cfg):
    _,F,S=matrices(cfg);a=terminal.clone();targets=[]
    running=running_cost_and_gradient(paths[:,:-1],cfg)[1]
    for k in range(cfg.steps-1,-1,-1):
        targets.append(-a@S);a=a@F+cfg.dt*running[:,k]
    return torch.stack(targets[::-1],1)

@torch.no_grad()
def evaluate(actor,cfg,seed,n=1024,steps=None,keep_paths=False):
    paths,controls=rollout(actor,cfg,n,rng(seed),steps);dt=cfg.horizon/controls.shape[1]
    energy=.5*controls.square().sum(-1).sum(-1)*dt
    running=running_cost_and_gradient(paths[:,:-1],cfg)[0].sum(-1)*dt
    terminal=true_terminal_value(paths[:,-1],cfg);cost=energy+running+terminal
    hit=torch.zeros(n,dtype=torch.bool)
    for cx,cy,r in cfg.obstacles:
        a=paths[:,:-1,:2]-tensor([cx,cy]);b=paths[:,1:,:2]-tensor([cx,cy]);v=b-a
        s=(-(a*v).sum(-1)/v.square().sum(-1).clamp_min(1e-12)).clamp(0,1)
        hit|=((a+s[...,None]*v).square().sum(-1)<r*r).any(-1)
    distance=(paths[:,-1,:2]-tensor(cfg.goal[:2])).norm(dim=-1)
    speed=(paths[:,-1,2:]-tensor(cfg.goal[2:])).norm(dim=-1)
    success=(~hit)&(distance<cfg.success_radius)&(speed<cfg.success_velocity)
    result=dict(cost=cost.mean().item(),cost_mc_se=cost.std().item()/math.sqrt(n),energy=energy.mean().item(),
                running=running.mean().item(),terminal=terminal.mean().item(),success=success.float().mean().item(),
                collision=hit.float().mean().item(),distance=distance.mean().item(),speed=speed.mean().item())
    if keep_paths:return result,dict(paths=paths.numpy(),controls=controls.numpy(),per_path_cost=cost.numpy(),success=success.numpy(),collision=hit.numpy())
    return result
