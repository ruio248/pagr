# %% [markdown]
# # Notebook 2 — TD critic action-gradient error in conditional Flow AM
#
# This is a new 16-dimensional, eight-step sparse-reward environment. Environment
# time `k`, internal Flow time `tau`, and Actor optimizer count `N` are distinct.
# Experiment A fits Q^{pi_ref} by TD and compares a frozen learned critic with
# its analytic Q^{pi_ref} oracle in otherwise paired Lean-AM policy extraction.
# Experiment B alternates TD updates using the *current* Actor's next action
# with AM updates. Q^{pi_ref} is deliberately **not** called its oracle.
# The old one-step reward-regression notebooks and all their runs remain intact.
#
# Original QAM reference: https://arxiv.org/html/2601.14234v3 and official
# https://github.com/ColinQiyangLi/qam/blob/2726d767c9a0a7a46d49693f0391f73dc2cf58ac/agents/qam.py
# We retain critic TD backup, terminal -beta grad_a Q, a conditional Flow, Lean
# VJPs and matching. Our endpoint is a finite-epsilon, all-Gaussian SDE chain;
# official code uses a shifted schedule, final ODE step, ODE action deployment,
# clipping, possible best-of-n and further engineering. This is not a line-by-line
# or benchmark reproduction. No robustification algorithm is introduced.

# %%
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, ncx2
import torch
from torch import nn
from torch.nn import functional as Fnn

ROOT = Path.cwd()
CONFIG_PATH = ROOT/'configs/qam_td_terminal_error.json'
torch.set_num_threads(min(4, os.cpu_count() or 1))


@dataclass(frozen=True)
class Config:
    design_seed: int = 60729
    dimension: int = 16
    horizon: int = 8
    kappa: float = .1
    sigma_pi: float = .35
    sigma_env: float = .15
    gamma: float = .98
    reference_success_quantile: float = .2
    train_episodes: int = 512
    valid_episodes: int = 128
    eval_episodes_curve: int = 128
    eval_episodes_final: int = 512
    extract_states_curve: int = 128
    extract_states_final: int = 512
    base_updates: int = 2500
    base_batch: int = 256
    critic_updates: int = 10000
    critic_batch: int = 256
    critic_next_samples: int = 4
    critic_members: int = 5
    critic_lr: float = .001
    critic_ema: float = .005
    actor_updates: int = 300
    actor_batch: int = 64
    actor_lr: float = .0003
    actor_clip: float = 5.
    beta: float = 10.
    flow_steps: int = 64
    warmup_steps: int = 8
    flow_epsilon: float = .05
    curve_updates: tuple = (0, 50, 100, 150, 200, 250, 300)
    diagnostic_updates: tuple = (0, 150, 300)


def load_config():
    raw = json.loads(CONFIG_PATH.read_text())
    fields = Config.__dataclass_fields__
    vals = {k: tuple(v) if k.endswith('_updates') and isinstance(v, list) else v
            for k, v in raw.items() if k in fields}
    return Config(**vals), raw


def seed_for(seed, role):
    return seed*10007+role*1009


def generator(seed):
    return torch.Generator(device='cpu').manual_seed(int(seed))


def check_finite(name, *values):
    if not all(bool(torch.isfinite(v).all()) for v in values):
        raise FloatingPointError(name)


def parameter_vector(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def gradient_vector(model):
    return torch.cat([p.grad.detach().flatten() for p in model.parameters()])


def set_gradient(model, values):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = values[offset:offset+n].reshape_as(p).clone()
        offset += n
    assert offset == values.numel()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SparseReachTask:
    """A single fixed task; design is identical across all train seeds."""
    def __init__(self, c):
        self.c = c
        rng = np.random.default_rng(c.design_seed)
        raw = rng.normal(size=(c.dimension, c.dimension))
        orthogonal, r = np.linalg.qr(raw)
        orthogonal = orthogonal*np.sign(np.diag(r))[None, :]
        singular = np.linspace(.5, 1.5, c.dimension)
        self.B64 = (orthogonal*singular[None, :])@orthogonal.T
        self.B = torch.from_numpy(self.B64).to(torch.float32)
        decay = 1-c.kappa
        variance = decay**(2*c.horizon) + (c.sigma_pi**2+c.sigma_env**2)*sum(
            decay**(2*j) for j in range(c.horizon))
        self.radius = math.sqrt(variance*chi2.ppf(c.reference_success_quantile, c.dimension))
        assert np.linalg.matrix_rank(self.B64) == c.dimension
        assert np.linalg.cond(self.B64) < 4

    def reference_action(self, y, noise):
        assert y.shape == noise.shape
        target = -self.c.kappa*y+self.c.sigma_pi*noise
        return torch.linalg.solve(self.B.to(y), target.T).T

    def step(self, k, y, action, env_noise):
        yn = y+action@self.B.to(y).T+self.c.sigma_env*env_noise
        reward = ((yn.square().sum(-1).sqrt() <= self.radius).to(y.dtype)
                  if k == self.c.horizon-1 else y.new_zeros(len(y)))
        return yn, reward, k == self.c.horizon-1


def collect_episodes(task, episodes, seed):
    """Separate RNG roles for y0, behavior action, and environment noise."""
    c = task.c
    y = torch.randn(episodes, c.dimension, generator=generator(seed_for(seed, 1)))
    action_rng = generator(seed_for(seed, 2))
    env_rng = generator(seed_for(seed, 3))
    columns = {name: [] for name in ('k', 'y', 'a', 'r', 'kn', 'yn', 'done')}
    for k in range(c.horizon):
        action = task.reference_action(y, torch.randn(y.shape, generator=action_rng))
        yn, reward, done = task.step(k, y, action, torch.randn(y.shape, generator=env_rng))
        for name, value in (('k', torch.full((episodes,), k, dtype=torch.long)),
                            ('y', y), ('a', action), ('r', reward),
                            ('kn', torch.full((episodes,), k+1, dtype=torch.long)),
                            ('yn', yn), ('done', torch.full((episodes,), done))):
            columns[name].append(value)
        y = yn
    return {name: torch.cat(vals).detach() for name, vals in columns.items()}


def data_summary(data, c):
    return dict(episodes=len(data['k'])//c.horizon, transitions=len(data['k']),
                successes=int(data['r'].sum()), nonzero_reward_fraction=float((data['r']>0).float().mean()),
                transitions_by_k={str(k): int((data['k']==k).sum()) for k in range(c.horizon)})


def oracle_numpy(task, k, y, action, with_gradient=True):
    """Q^{pi_ref}; float64 SciPy CDF, NOT a training target."""
    c = task.c
    k = np.asarray(k, dtype=np.int64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1, c.dimension)
    action = np.asarray(action, dtype=np.float64).reshape(-1, c.dimension)
    if len(k) != len(y) or len(y) != len(action) or (k<0).any() or (k>=c.horizon).any():
        raise ValueError('oracle input shapes/time')
    n = c.horizon-1-k
    decay = (1-c.kappa)**n
    v = decay**2*c.sigma_env**2 + (c.sigma_pi**2+c.sigma_env**2)*np.array([
        sum((1-c.kappa)**(2*j) for j in range(int(ni))) for ni in n])
    mu = decay[:, None]*(y+action@task.B64.T)
    lam = np.einsum('ij,ij->i', mu, mu)/v
    radius2_over_v = task.radius**2/v
    cdf = ncx2.cdf(radius2_over_v, c.dimension, lam)
    value = c.gamma**n*cdf
    if not with_gradient:
        return value
    cdf_plus = ncx2.cdf(radius2_over_v, c.dimension+2, lam)
    # d/dlambda CDF_d = (CDF_{d+2}-CDF_d)/2; chain rule cancels 2.
    grad = ((c.gamma**n*decay/v)*(cdf_plus-cdf))[:, None]*(mu@task.B64)
    if not (np.isfinite(value).all() and np.isfinite(grad).all()):
        raise FloatingPointError('oracle CDF/gradient')
    return value, grad


def oracle_torch(task, k, y, action, with_gradient=True):
    result = oracle_numpy(task, k.detach().numpy(), y.detach().double().numpy(),
                          action.detach().double().numpy(), with_gradient)
    if with_gradient:
        value, grad = result
        return (torch.as_tensor(value, dtype=action.dtype),
                torch.as_tensor(grad, dtype=action.dtype))
    return torch.as_tensor(result, dtype=action.dtype)


def verify_oracle(task, seed=9001):
    """Independent MC, Bellman and multiple-step finite-difference checks."""
    c = task.c
    rng = np.random.default_rng(seed)
    rows = []
    for k in (0, 3, c.horizon-1):
        y = rng.normal(size=(3, c.dimension))*.7
        a = np.linalg.solve(task.B64, (-c.kappa*y).T).T
        kvals = np.full(len(y), k)
        value, grad = oracle_numpy(task, kvals, y, a)
        for h in (1e-3, 3e-4, 1e-4):
            direction = rng.normal(size=a.shape)
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            fd = (oracle_numpy(task, kvals, y, a+h*direction, False)-
                  oracle_numpy(task, kvals, y, a-h*direction, False))/(2*h)
            err = float(np.max(np.abs(fd-(grad*direction).sum(-1))))
            rows.append(dict(check='finite_difference', k=k, h=h, max_error=err))
            assert err < 3e-6, rows[-1]
        assert (value >= 0).all() and (value <= c.gamma**(c.horizon-1-k)+1e-12).all()
        # Fresh independent Gaussian draws; direct terminal Bernoulli MC.
        sample_count = 30000
        for i in range(len(y)):
            n = c.horizon-1-k
            noise = rng.normal(size=(sample_count, c.dimension))*c.sigma_env
            samples = y[i]+task.B64@a[i]+noise
            for j in range(n):
                eps = rng.normal(size=(sample_count, c.dimension))*c.sigma_pi
                env = rng.normal(size=(sample_count, c.dimension))*c.sigma_env
                samples = (1-c.kappa)*samples+eps+env
            outcome = (np.linalg.norm(samples, axis=1)<=task.radius).mean()*c.gamma**n
            tolerance = 5*math.sqrt(max(value[i]*(1-value[i]), .01)/sample_count)+.003
            assert abs(value[i]-outcome)<tolerance, (k, value[i], outcome)
            rows.append(dict(check='mc_return', k=k, analytic=float(value[i]), mc=float(outcome),
                             tolerance=tolerance))
        # Independent one-step Gaussian Bellman check. At terminal there is no backup.
        if k < c.horizon-1:
            count = 12000
            next_y = y[0]+task.B64@a[0]+c.sigma_env*rng.normal(size=(count, c.dimension))
            next_a = np.linalg.solve(task.B64, (-c.kappa*next_y+
                       c.sigma_pi*rng.normal(size=(count,c.dimension))).T).T
            bellman = c.gamma*oracle_numpy(task, np.full(count,k+1), next_y, next_a, False).mean()
            assert abs(value[0]-bellman)<.012, (k, value[0], bellman)
            rows.append(dict(check='bellman', k=k, analytic=float(value[0]), mc=float(bellman)))
    zero_y = np.zeros((1,c.dimension))
    zero_a = np.zeros_like(zero_y)
    qzero, gzero = oracle_numpy(task, [c.horizon-1], zero_y, zero_a)
    assert np.isfinite(qzero).all() and np.isfinite(gzero).all() and np.linalg.norm(gzero)<1e-12
    far = np.full_like(zero_y, 1e3)
    qfar, gfar = oracle_numpy(task, [c.horizon-1], zero_y, far)
    assert np.isfinite(qfar).all() and np.isfinite(gfar).all()
    rows.append(dict(check='boundary', q_zero=float(qzero[0]), q_far=float(qfar[0])))
    return pd.DataFrame(rows)


# %% [markdown]
# ## Offline transitions and actual TD fitting
#
# Only real `(k,y,a,r,k+1,y_next,done)` transitions enter TD. For nonterminal
# transitions the target is `r + gamma * mean_{4 ref actions,5 target members} Q_target`;
# terminal transitions use `r`. The analytic oracle is never consulted by TD.
# Both the critic and conditional behavior Flow see remaining horizon via k/H.

# %%
class Critic(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.members = nn.ModuleList([nn.Sequential(
            nn.Linear(2*c.dimension+1,128), nn.SiLU(), nn.Linear(128,128),
            nn.SiLU(), nn.Linear(128,1)) for _ in range(c.critic_members)])

    def forward(self, k, y, a):
        feature = torch.cat((k.reshape(-1,1).to(y.dtype)/self.c.horizon,y,a), -1)
        return torch.cat([member(feature) for member in self.members], -1)


class ConditionalVelocity(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.net = nn.Sequential(nn.Linear(2*c.dimension+2,128), nn.SiLU(),
                                 nn.Linear(128,128), nn.SiLU(), nn.Linear(128,c.dimension))

    def features(self, k, y, x, tau):
        shape = x.shape[:-1]
        kval = k.to(x.dtype).reshape(shape+(1,))/self.c.horizon
        tauval = torch.as_tensor(tau, dtype=x.dtype).expand(shape+(1,))
        return torch.cat((kval,y,x,tauval), -1)

    def forward(self, k, y, x, tau):
        return self.net(self.features(k,y,x,tau))

    def input_vjp_cache(self, k, y, x, tau):
        feature = self.features(k,y,x,tau)
        pre1 = self.net[0](feature)
        pre2 = self.net[2](Fnn.silu(pre1))
        return pre1, pre2

    def input_vjp_from_cache(self, pre1, pre2, vec):
        def silu_prime(z):
            sig = torch.sigmoid(z)
            return sig*(1+z*(1-sig))
        z = (vec@self.net[4].weight)*silu_prime(pre2)
        z = (z@self.net[2].weight)*silu_prime(pre1)
        return (z@self.net[0].weight)[..., self.c.dimension+1:2*self.c.dimension+1]

    def input_vjp(self, k, y, x, tau, vec):
        return self.input_vjp_from_cache(*self.input_vjp_cache(k,y,x,tau),vec)


def td_target(task, target, batch, rng, c, mode='reference', actor=None, base=None):
    with torch.no_grad():
        if mode == 'reference':
            yn = batch['yn'].repeat_interleave(c.critic_next_samples,0)
            kn = batch['kn'].repeat_interleave(c.critic_next_samples,0)
            eps = torch.randn(yn.shape, generator=rng)
            an = task.reference_action(yn,eps)
            next_q = target(kn,yn,an).mean(-1).reshape(-1,c.critic_next_samples).mean(-1)
        elif mode == 'actor':
            assert actor is not None and base is not None
            yn, kn = batch['yn'], batch['kn']
            z = torch.randn(yn.shape,generator=rng)
            xi = torch.randn((len(yn),c.flow_steps,c.dimension),generator=rng)
            an = rollout(actor,base,c,kn,yn,z,xi)[0][:,-1]
            next_q = target(kn,yn,an).mean(-1)
        else:
            raise ValueError(mode)
        return batch['r']+c.gamma*(~batch['done']).to(batch['r'])*next_q


def td_step(task, critic, target, optimizer, data, indices, rng, c, mode='reference', actor=None, base=None):
    batch = {key:value[indices] for key,value in data.items()}
    label = td_target(task,target,batch,rng,c,mode,actor,base)
    prediction = critic(batch['k'],batch['y'],batch['a'])
    loss = (prediction-label[:,None]).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        for p,tp in zip(critic.parameters(),target.parameters()):
            tp.lerp_(p,c.critic_ema)
    check_finite('TD critic',loss)
    return float(loss.detach()), float(prediction.mean().detach())


def critic_diagnostics(task, critic, target, valid, c, seed):
    with torch.no_grad():
        pred = critic(valid['k'],valid['y'],valid['a']).mean(-1)
        oracle = oracle_torch(task,valid['k'],valid['y'],valid['a'],False)
        rng = generator(seed_for(seed,91))
        residual = pred-td_target(task,target,valid,rng,c)
    n = c.horizon-1-valid['k']
    ceiling = torch.pow(torch.as_tensor(c.gamma),n)
    return dict(oracle_q_rmse=float((pred-oracle).square().mean().sqrt()),
                validation_bellman_rmse=float(residual.square().mean().sqrt()),
                value_out_of_range_fraction=float(((pred<0)|(pred>ceiling)).float().mean()),
                q_mean=float(pred.mean()), q_std=float(pred.std(unbiased=False)),
                oracle_q_mean=float(oracle.mean()))


def train_td_reference(task, train, valid, c, seed, output):
    torch.manual_seed(seed_for(seed,100))
    critic, target = Critic(c), Critic(c)
    target.load_state_dict(critic.state_dict())
    optimizer = torch.optim.Adam(critic.parameters(),lr=c.critic_lr)
    rng = generator(seed_for(seed,101))
    logs = []
    for step in range(1,c.critic_updates+1):
        indices = torch.randint(len(train['k']),(c.critic_batch,),generator=rng)
        loss,qmean = td_step(task,critic,target,optimizer,train,indices,rng,c)
        if step % max(1,c.critic_updates//20)==0 or step==1:
            logs.append(dict(update=step,td_loss=loss,training_batch_q_mean=qmean,
                             **critic_diagnostics(task,critic,target,valid,c,seed)))
            print('TD',seed,step,'loss',round(loss,5),'Q',round(qmean,4),flush=True)
    torch.save(dict(critic=critic.state_dict(),target=target.state_dict(),
                    optimizer=optimizer.state_dict()),output/'critic-A.pt')
    pd.DataFrame(logs).to_csv(output/'critic-A-log.csv',index=False)
    return critic,target,logs


def train_base(train,c,seed,output):
    torch.manual_seed(seed_for(seed,200))
    model = ConditionalVelocity(c)
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
    rng = generator(seed_for(seed,201))
    logs=[]
    for step in range(1,c.base_updates+1):
        index = torch.randint(len(train['k']),(c.base_batch,),generator=rng)
        k,y,a = (train[key][index] for key in ('k','y','a'))
        z = torch.randn(a.shape,generator=rng)
        t = torch.rand((len(index),1),generator=rng)
        x = (1-t)*z+t*a
        loss=(model(k,y,x,t)-(a-z)).square().sum(-1).mean()/c.dimension
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step % max(1,c.base_updates//10)==0 or step==1:
            logs.append(dict(update=step,flow_mse_per_dim=float(loss.detach())))
            print('FM',seed,step,round(float(loss.detach()),4),flush=True)
    torch.save(model.state_dict(),output/'base.pt')
    pd.DataFrame(logs).to_csv(output/'base-log.csv',index=False)
    for p in model.parameters():p.requires_grad_(False)
    return model


# %% [markdown]
# ## Conditional finite-chain sampler and Lean Adjoint
#
# The frozen Base is warmed up by eight RK2 ODE steps from 0 to epsilon=.05.
# Thereafter every one of K=64 left-endpoint steps is Gaussian EM, including
# the last one. `X_1` denotes completion of *this action generation*, never the
# end of the environment episode. The endpoint is not clipped or denoised.
# The fixed `(k,y)` condition is excluded from D_X b_base. This discretization
# differs from official QAM's shifted noise schedule/final ODE step and its
# default ODE deployment; both TD next actions and deployment below use this
# same finite-chain SDE sampler. No inner energy enters environment TD reward.

# %%
def flow_grid(c):
    h=(1-c.flow_epsilon)/c.flow_steps
    t = c.flow_epsilon+h*torch.arange(c.flow_steps,dtype=torch.float32)
    nu = torch.sqrt(2*(1-t)/t)
    return t,nu,h


@torch.no_grad()
def warmup(base,c,k,y,z):
    x = z.clone()
    h=c.flow_epsilon/c.warmup_steps
    for j in range(c.warmup_steps):
        t=j*h
        midpoint=x+.5*h*base(k,y,x,t)
        x=x+h*base(k,y,midpoint,t+.5*h)
    return x


@torch.no_grad()
def rollout(actor,base,c,k,y,z,xi):
    assert z.shape == y.shape and xi.shape == (len(y),c.flow_steps,c.dimension)
    t,nu,h=flow_grid(c)
    x=warmup(base,c,k,y,z)
    paths=[x]
    controls=[]
    for j in range(c.flow_steps):
        va=actor(k,y,x,t[j]); vb=base(k,y,x,t[j])
        controls.append(2*(va-vb)/nu[j])
        x=x+h*(2*va-x/t[j])+math.sqrt(h)*nu[j]*xi[:,j]
        paths.append(x)
    paths=torch.stack(paths,1).detach()
    controls=torch.stack(controls,1).detach()
    check_finite('rollout',paths,controls)
    return paths,controls


@torch.no_grad()
def lean_adjoint(base,c,k,y,paths,terminal):
    t,_,h=flow_grid(c)
    p=terminal.detach()
    sequence=[p]
    for j in range(c.flow_steps-1,-1,-1):
        vjp=base.input_vjp(k,y,paths[:,j],t[j],p)
        p=p+h*(2*vjp-p/t[j])
        sequence.append(p)
    sequence=torch.stack(sequence[::-1],1).detach()
    check_finite('lean adjoint',sequence)
    return sequence


def terminal_label(task,critic,c,k,y,endpoint,source):
    if source=='oracle':
        gradient=oracle_torch(task,k,y,endpoint)[1]
    elif source=='learned':
        with torch.enable_grad():
            a=endpoint.detach().requires_grad_(True)
            gradient=torch.autograd.grad(critic(k,y,a).mean(-1).sum(),a)[0].detach()
    else:raise ValueError(source)
    result=-c.beta*gradient
    check_finite('terminal label',result)
    return result


def matching_loss(actor,base,c,k,y,paths,adjoints):
    t,nu,h=flow_grid(c)
    n=len(y);d=c.dimension;K=c.flow_steps
    kk=k[:,None].expand(n,K).reshape(-1)
    yy=y[:,None,:].expand(n,K,d).reshape(-1,d)
    xx=paths[:,:-1].detach().reshape(-1,d)
    tt=t[None,:].expand(n,K).reshape(-1,1)
    with torch.no_grad():vb=base(kk,yy,xx,tt)
    va=actor(kk,yy,xx,tt)
    controls=2*(va-vb).reshape(n,K,d)/nu[None,:,None]
    residual=controls+nu[None,:,None]*adjoints[:,1:].detach()
    return .5*h*residual.square().sum((1,2)).mean()/d


def actor_gradient(actor,base,c,k,y,paths,adjoints):
    actor.zero_grad(set_to_none=True)
    loss=matching_loss(actor,base,c,k,y,paths,adjoints)
    loss.backward()
    q=gradient_vector(actor).clone()
    actor.zero_grad(set_to_none=True)
    check_finite('matching parameter gradient',q)
    return q,float(loss.detach())


def actor_step(actor,opt,base,critic,task,c,k,y,z,xi,source):
    paths,_=rollout(actor,base,c,k,y,z,xi)
    terminal=terminal_label(task,critic,c,k,y,paths[:,-1],source)
    adj=lean_adjoint(base,c,k,y,paths,terminal)
    q,loss=actor_gradient(actor,base,c,k,y,paths,adj)
    set_gradient(actor,q)
    clipped=torch.nn.utils.clip_grad_norm_(actor.parameters(),c.actor_clip,error_if_nonfinite=True)
    opt.step();actor.zero_grad(set_to_none=True)
    return dict(matching_loss=loss,matching_gradient_norm=float(q.norm()),
                pre_clip_norm=float(clipped))


def virtual_adam(actor,opt,q,c):
    branch=copy.deepcopy(actor)
    branch_opt=torch.optim.Adam(branch.parameters(),lr=c.actor_lr)
    branch_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    before=parameter_vector(branch)
    set_gradient(branch,q)
    torch.nn.utils.clip_grad_norm_(branch.parameters(),c.actor_clip,error_if_nonfinite=True)
    branch_opt.step()
    return parameter_vector(branch)-before


def mechanism_diagnostic(task,critic,base,actor,opt,c,k,y,z,xi,seed,N,output):
    """Same actor, same paths: value, terminal, propagated, matching, Adam."""
    paths,_=rollout(actor,base,c,k,y,z,xi)
    endpoint=paths[:,-1]
    with torch.no_grad():
        qhat=critic(k,y,endpoint).mean(-1)
        qref=oracle_torch(task,k,y,endpoint,False)
    oracle_grad=oracle_torch(task,k,y,endpoint)[1]
    learned_grad=-terminal_label(task,critic,c,k,y,endpoint,'learned')/c.beta
    delta_terminal=-c.beta*(learned_grad-oracle_grad)
    pL=lean_adjoint(base,c,k,y,paths,-c.beta*learned_grad)
    pO=lean_adjoint(base,c,k,y,paths,-c.beta*oracle_grad)
    dp=pL-pO
    dp_expected=lean_adjoint(base,c,k,y,paths,delta_terminal)
    recurrence_error=float((dp-dp_expected).abs().max())
    assert recurrence_error < 3e-5,(seed,N,recurrence_error)
    t,nu,h=flow_grid(c)
    label_diff=nu[None,:,None]*dp[:,1:]
    qL,lossL=actor_gradient(actor,base,c,k,y,paths,pL)
    qO,lossO=actor_gradient(actor,base,c,k,y,paths,pO)
    # Jacobian projection identity, without materializing an enormous J_theta.
    n=len(y);d=c.dimension;K=c.flow_steps
    kk=k[:,None].expand(n,K).reshape(-1)
    yy=y[:,None,:].expand(n,K,d).reshape(-1,d)
    xx=paths[:,:-1].reshape(-1,d)
    tt=t[None,:].expand(n,K).reshape(-1,1)
    actor.zero_grad(set_to_none=True)
    projection=(2*h/(n*d))*(actor(kk,yy,xx,tt).reshape(n,K,d)*dp[:,1:]).sum()
    projection.backward()
    projected=gradient_vector(actor).clone()
    actor.zero_grad(set_to_none=True)
    parameter_error=float(((qL-qO)-projected).abs().max())
    assert parameter_error < 3e-5,(seed,N,parameter_error)
    adamL=virtual_adam(actor,opt,qL,c)
    adamO=virtual_adam(actor,opt,qO,c)
    grad_error=learned_grad-oracle_grad
    numerator=(learned_grad*oracle_grad).sum(-1)
    den=learned_grad.norm(dim=-1)*oracle_grad.norm(dim=-1)
    valid=den>1e-10
    cosine=float((numerator[valid]/den[valid]).mean()) if bool(valid.any()) else float('nan')
    by_time=(dp.square().mean((0,2)).sqrt()).numpy()
    label_by_time=label_diff.square().mean((0,2)).sqrt().numpy()
    file=output/f'mechanism-seed{seed}-N{N}.npz'
    np.savez_compressed(file,terminal=endpoint.numpy(),k=k.numpy(),q_value_error=(qhat-qref).numpy(),
                        terminal_gradient_error=grad_error.numpy(),terminal_gradient_ref=oracle_grad.numpy(),
                        adjoint_error=dp.numpy(),matching_label_error=label_diff.numpy(),
                        matching_parameter_gradient_error=(qL-qO).numpy(),
                        adam_step_error=(adamL-adamO).numpy())
    return dict(seed=seed,N=N,q_value_rmse=float((qhat-qref).square().mean().sqrt()),
                gradient_abs_rms=float(grad_error.square().mean().sqrt()),
                gradient_rel_rms=float(grad_error.square().mean().sqrt()/oracle_grad.square().mean().sqrt().clamp_min(1e-12)),
                gradient_cosine=cosine,gradient_nonzero_fraction=float((oracle_grad.norm(dim=-1)>1e-6).float().mean()),
                adjoint_rms_by_flow_time=json.dumps(by_time.tolist()),
                label_rms_by_flow_time=json.dumps(label_by_time.tolist()),
                matching_gradient_difference_norm=float((qL-qO).norm()),
                adam_step_difference_norm=float((adamL-adamO).norm()),
                recurrence_max_error=recurrence_error,parameter_projection_max_error=parameter_error,
                learned_matching_loss=lossL,oracle_matching_loss=lossO,
                diagnostic_npz=file.name)


# %% [markdown]
# ## Fixed-N extraction and true closed-loop environment evaluation
#
# Extraction uses a *fixed independent* environment-state distribution and
# analytic Q^{pi_ref}: E[path KL - beta Q_ref(k,y,X1)]. It is not the actual
# closed-loop return. For deployment, a new action is generated at each
# environment step and only the final success indicator pays reward.
# Both methods share initial states, flow noise and environment-noise draws;
# evaluation never updates parameters. The same SDE sampler is used in A/B.

# %%
def paired_generation_noise(c,n,seed):
    rng=generator(seed)
    return (torch.randn((n,c.dimension),generator=rng),
            torch.randn((n,c.flow_steps,c.dimension),generator=rng))


@torch.no_grad()
def evaluate_extraction(task,actor,base,c,states,seed):
    k,y=states['k'],states['y']
    z,xi=paired_generation_noise(c,len(k),seed)
    paths,u=rollout(actor,base,c,k,y,z,xi)
    qref=oracle_torch(task,k,y,paths[:,-1],False)
    _,_,h=flow_grid(c)
    energy=.5*h*u.square().sum((1,2))
    j=energy-c.beta*qref
    return dict(extract_J=float(j.mean()),extract_J_se=float(j.std(unbiased=True)/math.sqrt(len(j))),
                extract_KL=float(energy.mean()),extract_Q_ref=float(qref.mean()))


@torch.no_grad()
def evaluate_environment(task,actor,base,c,episodes,seed):
    rng0=generator(seed_for(seed,11));rngact=generator(seed_for(seed,12));rngenv=generator(seed_for(seed,13))
    y=torch.randn((episodes,c.dimension),generator=rng0)
    terminal_reward=None
    for k0 in range(c.horizon):
        k=torch.full((episodes,),k0,dtype=torch.long)
        z=torch.randn((episodes,c.dimension),generator=rngact)
        xi=torch.randn((episodes,c.flow_steps,c.dimension),generator=rngact)
        action=rollout(actor,base,c,k,y,z,xi)[0][:,-1]
        y,terminal_reward,_=task.step(k0,y,action,torch.randn(y.shape,generator=rngenv))
    success=float(terminal_reward.mean())
    return dict(success_rate=success,discounted_return=success*c.gamma**(c.horizon-1),
                eval_episodes=episodes)


def evaluate_checkpoint(task,actor,base,c,eval_states,seed,episodes):
    return dict(**evaluate_extraction(task,actor,base,c,eval_states,seed),
                **evaluate_environment(task,actor,base,c,episodes,seed))


def train_extraction(task,critic,base,train,eval_states,c,seed,output):
    learned=copy.deepcopy(base);oracle=copy.deepcopy(base)
    for model in (learned,oracle):
        for p in model.parameters():p.requires_grad_(True)
    optL=torch.optim.Adam(learned.parameters(),lr=c.actor_lr)
    optO=torch.optim.Adam(oracle.parameters(),lr=c.actor_lr)
    rng=generator(seed_for(seed,301))
    curves=[];logs=[];mechanisms=[]
    for N in range(c.actor_updates+1):
        if N in c.curve_updates:
            for name,model in (('learned',learned),('oracle',oracle)):
                row=dict(seed=seed,N=N,method=name,
                         **evaluate_checkpoint(task,model,base,c,eval_states,
                                               seed_for(seed,500+N),c.eval_episodes_curve))
                curves.append(row)
                torch.save(dict(model=model.state_dict(),optimizer=(optL if name=='learned' else optO).state_dict()),
                           output/f'actor-A-{name}-N{N}.pt')
                print('A eval',seed,N,name,row['extract_J'],row['success_rate'],flush=True)
        if N in c.diagnostic_updates:
            indices=torch.randint(len(train['k']),(c.actor_batch,),generator=generator(seed_for(seed,400+N)))
            k,y=train['k'][indices],train['y'][indices]
            z,xi=paired_generation_noise(c,len(k),seed_for(seed,600+N))
            mechanisms.append(mechanism_diagnostic(task,critic,base,learned,optL,c,k,y,z,xi,seed,N,output))
        if N==c.actor_updates:break
        index=torch.randint(len(train['k']),(c.actor_batch,),generator=rng)
        k,y=train['k'][index],train['y'][index]
        z=torch.randn((len(index),c.dimension),generator=rng)
        xi=torch.randn((len(index),c.flow_steps,c.dimension),generator=rng)
        # Learned update is explicitly prohibited from consulting oracle.
        resultL=actor_step(learned,optL,base,critic,task,c,k,y,z,xi,'learned')
        resultO=actor_step(oracle,optO,base,critic,task,c,k,y,z,xi,'oracle')
        if N%25==0:logs.extend([dict(seed=seed,N=N+1,method='learned',**resultL),
                               dict(seed=seed,N=N+1,method='oracle',**resultO)])
    pd.DataFrame(curves).to_csv(output/'experiment-A-curves.csv',index=False)
    pd.DataFrame(mechanisms).to_csv(output/'experiment-A-mechanism.csv',index=False)
    pd.DataFrame(logs).to_csv(output/'experiment-A-training.csv',index=False)
    return learned,oracle,curves,mechanisms


def train_joint(task,pretrained_critic,base,train,c,seed,output):
    """One current-Actor TD step followed by one Lean-AM step, N times."""
    critic=copy.deepcopy(pretrained_critic)
    target=copy.deepcopy(pretrained_critic)
    actor=copy.deepcopy(base)
    for p in actor.parameters():p.requires_grad_(True)
    copt=torch.optim.Adam(critic.parameters(),lr=c.critic_lr)
    aopt=torch.optim.Adam(actor.parameters(),lr=c.actor_lr)
    rng=generator(seed_for(seed,701))
    curves=[];logs=[]
    for N in range(c.actor_updates+1):
        if N in c.curve_updates:
            row=dict(seed=seed,N=N,method='joint_learned',
                     **evaluate_environment(task,actor,base,c,c.eval_episodes_curve,seed_for(seed,500+N)))
            curves.append(row)
            torch.save(dict(actor=actor.state_dict(),critic=critic.state_dict(),target=target.state_dict(),
                            actor_optimizer=aopt.state_dict(),critic_optimizer=copt.state_dict()),
                       output/f'joint-B-N{N}.pt')
            print('B eval',seed,N,row['success_rate'],flush=True)
        if N==c.actor_updates:break
        index=torch.randint(len(train['k']),(c.critic_batch,),generator=rng)
        loss,qmean=td_step(task,critic,target,copt,train,index,rng,c,'actor',actor,base)
        index=torch.randint(len(train['k']),(c.actor_batch,),generator=rng)
        k,y=train['k'][index],train['y'][index]
        z=torch.randn((len(index),c.dimension),generator=rng)
        xi=torch.randn((len(index),c.flow_steps,c.dimension),generator=rng)
        actor_log=actor_step(actor,aopt,base,critic,task,c,k,y,z,xi,'learned')
        if N%25==0:logs.append(dict(seed=seed,N=N+1,critic_updates=N+1,td_loss=loss,q_mean=qmean,**actor_log))
    pd.DataFrame(curves).to_csv(output/'experiment-B-curves.csv',index=False)
    pd.DataFrame(logs).to_csv(output/'experiment-B-training.csv',index=False)
    return actor,curves


# %% [markdown]
# ## Pre-registered exploratory protocol, provenance, plots, and interpretation
#
# The radius and B are fixed once from an independent design seed, *before*
# training seeds. The three formal seeds are fixed in JSON and never selected by
# A's Oracle advantage. Each seed has its own disjoint train/validation/eval
# episodes. N=0,50,...,300 are reported without best-checkpoint selection.
# A is a controlled fixed-pi_ref policy-extraction experiment; B is a minimal
# joint TD+AM version with Actor next-action backup. Their statistics are never
# combined, and Q^{pi_ref} is never used as an Oracle for B.

# %%
def action_value_future_check(task, data, output):
    c=task.c
    early=(data['k']==0).nonzero().flatten()[:256]
    k,y,a=(data[name][early] for name in ('k','y','a'))
    assert bool((data['r'][early]==0).all())
    q,grad=oracle_torch(task,k,y,a)
    # Vary an action direction at exactly the same early environment state.
    perturb=torch.zeros_like(a);perturb[:,0]=.25
    qshift=oracle_torch(task,k,y,a+perturb,False)
    result=dict(k=0,immediate_reward=float(data['r'][early].mean()),
                Q_mean=float(q.mean()),Q_std=float(q.std(unbiased=False)),
                Q_action_perturb_mean_abs_change=float((qshift-q).abs().mean()),
                Q_action_gradient_nonzero_fraction=float((grad.norm(dim=-1)>1e-6).float().mean()))
    assert result['Q_action_perturb_mean_abs_change']>1e-6
    assert result['Q_action_gradient_nonzero_fraction']>0
    (output/'future-value-check.json').write_text(json.dumps(result,indent=2))
    return result


def plot_results(output):
    curves=pd.read_csv(output/'experiment-A-curves.csv')
    diag=pd.read_csv(output/'experiment-A-mechanism.csv')
    b=pd.read_csv(output/'experiment-B-curves.csv')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for method, group in curves.groupby('method'):
        agg=group.groupby('N')[['extract_J','success_rate']].mean()
        axes[0].plot(agg.index,agg.extract_J,marker='o',label=method)
        axes[1].plot(agg.index,agg.success_rate,marker='o',label=method)
    agg=b.groupby('N').success_rate.mean()
    axes[1].plot(agg.index,agg,marker='s',ls=':',label='joint learned (separate B)')
    for (seed,N),group in diag.groupby(['seed','N']):
        values=np.array(json.loads(group.iloc[0].adjoint_rms_by_flow_time))
        axes[2].plot(np.linspace(0,1,len(values)),values,alpha=.35,label=f'{seed} N{N}')
    axes[0].set(title='A: fixed-state extraction objective',xlabel='Actor optimizer steps N',ylabel='J_extract (lower better)')
    axes[1].set(title='A/B: real sparse environment',xlabel='Actor optimizer steps N',ylabel='success rate (higher better)')
    axes[2].set(title='A: common-path adjoint error',xlabel='internal Flow time tau',ylabel='RMS delta p')
    axes[0].legend();axes[1].legend();axes[2].legend(fontsize=7,ncol=2)
    for axis in axes:axis.grid(alpha=.25)
    fig.tight_layout();fig.savefig(output/'fixed-N-and-propagation.png',dpi=160);plt.close(fig)


def report(output,c,formal):
    A=pd.read_csv(output/'experiment-A-curves.csv')
    B=pd.read_csv(output/'experiment-B-curves.csv')
    M=pd.read_csv(output/'experiment-A-mechanism.csv')
    D=pd.read_csv(output/'critic-A-diagnostics.csv')
    S=pd.read_csv(output/'data-summary.csv')
    end=A[A.N==c.actor_updates].pivot(index='seed',columns='method',values=['extract_J','success_rate','discounted_return'])
    deltaJ=end[('extract_J','learned')]-end[('extract_J','oracle')]
    deltaSuccess=end[('success_rate','oracle')]-end[('success_rate','learned')]
    bend=B[B.N==c.actor_updates].set_index('seed').success_rate
    def table(df):
        def cell(value):
            if isinstance(value,(float,np.floating)):return f'{value:.5g}'
            return str(value)
        names=list(df.columns)
        lines=['| '+' | '.join(map(str,names))+' |',
               '| '+' | '.join(['---']*len(names))+' |']
        lines += ['| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
        return '\n'.join(lines)
    contents=[
        '# TD-Critic terminal-gradient experiment', '',
        f'Profile: **{"formal 3-seed exploratory" if formal else "development smoke; not formal evidence"}**. '
        f'Fixed task d={c.dimension}, H={c.horizon}, sigma_env={c.sigma_env}, gamma={c.gamma}, '
        f'beta={c.beta}, K={c.flow_steps}, Actor B={c.actor_batch}.', '',
        '## Scope and references', '',
        'A: fixed pi_ref, TD learning Q^{pi_ref}, frozen critic, paired Learned/Oracle Lean AM. '
        'B: current Actor next-action TD backup plus alternating AM. B has no exact current-policy Oracle. '
        'Neither is a full official benchmark reproduction, and no new robustification method is claimed.', '',
        'Official qam.py `main` resolved to commit `2726d767c9a0a7a46d49693f0391f73dc2cf58ac`; '
        'critic_loss uses TD target and ensemble, adj_matching initializes with negative critic action gradient, '
        'actor_loss matches the propagated lean label, sample_actions deploys ODE actions with clipping and '
        'possible best-of-n. This run instead uses a shared finite-epsilon all-Gaussian SDE sampler, no clipping.', '',
        '## Offline data and reference value', '',
        table(S), '',
        f'Goal radius {json.loads((output/"task.json").read_text())["radius"]:.6g} was set once from the '
        'fixed reference terminal-state 20% quantile before formal seeds. Analytic Q and gradient are verified '
        'against Monte Carlo, multiple finite-difference h and the Bellman relation; see oracle-checks.csv. '
        'At k<7, immediate r=0 while Q_ref varies with action; see future-value-check.json.', '',
        '## A: critic diagnostics and fixed-N mechanism', '',
        'The critic was trained only from observed transitions with TD target and pi_ref next actions; '
        'Q_ref was not a target or checkpoint-selection criterion. Final checkpoint is fixed.', '',
        table(D), '',
        'On common paths at N=0/150/300, `delta p_1=-beta(grad Qhat-grad Qref)` was independently '
        'propagated and compared with the difference of two full lean recursions. The actual matching-label '
        'difference is `nu_j delta p_{j+1}`; the parameter difference was checked against the Actor-Jacobian '
        'projection and identical-history virtual Adam steps were cloned. See per-seed NPZ and table.', '',
        table(M[['seed','N','q_value_rmse','gradient_abs_rms','gradient_rel_rms',
                 'gradient_cosine','matching_gradient_difference_norm','adam_step_difference_norm',
                 'recurrence_max_error','parameter_projection_max_error']]), '',
        '## A: paired fixed-N outcomes', '',
        'J_extract = mean[path KL - beta Q_ref] on shared independent states; it is not environment return. '
        'A closed-loop success is an actual eight-step episode with fresh actions at each k.', '',
        table(A[['seed','N','method','extract_J','extract_KL','extract_Q_ref',
                 'success_rate','discounted_return','eval_episodes']]), '',
        'Final paired positive deltaJ = Learned J_extract - Oracle J_extract; '
        'positive deltaSuccess = Oracle success - Learned success:', '',
        table(pd.DataFrame({'seed':deltaJ.index,'deltaJ':deltaJ.values,
                            'deltaSuccess':deltaSuccess.values})), '',
        f'Final exploratory mean deltaJ={deltaJ.mean():.6g} ({int((deltaJ>0).sum())}/{len(deltaJ)} positive); '
        f'mean deltaSuccess={deltaSuccess.mean():.6g} ({int((deltaSuccess>0).sum())}/{len(deltaSuccess)} positive). '
        'Evaluation episodes are not independent training replicates; seed is the comparison unit. '
        'No best-N choice or conclusion of general superiority is made.', '',
        '## B: minimal joint TD+AM', '',
        'A-pretrained critic initializes B; every B step uses the **current Actor** to generate its TD next '
        'action with the same SDE sampler, updates critic once, then computes a fresh learned-Q action gradient '
        'and one Lean-AM Actor update. The following is real closed-loop evaluation only. '
        'Q^{pi_ref} is **not** an Oracle for this evolving policy.', '',
        table(B[['seed','N','success_rate','discounted_return','eval_episodes']]), '',
        f'Joint final mean success rate={bend.mean():.6g}; these B results must not be mixed with A paired '
        'Oracle contrasts. B demonstrates the TD→Q-gradient→Lean-adjoint call chain, not a Q-gradient-error '
        'measurement against an exact current-policy value.', '',
        '## Limits', '',
        'Three exploratory seeds, finite data, fixed critic updates and finite-epsilon discretization do not '
        'prove a universal bottleneck. Oracle-Qref in A is a fixed-policy evaluator, not Q*. Oracle matching '
        'gradient is not generally the full true policy gradient. Critic fit, optimization, rollout distribution '
        'and finite Monte Carlo may all limit outcome differences. No robust guard or new algorithm was tested.'
    ]
    (output/'report.md').write_text('\n'.join(contents)+'\n')


def run_experiment(profile='smoke'):
    c,raw=load_config()
    if profile=='smoke':
        from dataclasses import replace
        c=replace(c,train_episodes=48,valid_episodes=16,eval_episodes_curve=24,
                  eval_episodes_final=48,extract_states_curve=32,extract_states_final=64,
                  base_updates=8,critic_updates=12,critic_members=2,
                  actor_updates=4,actor_batch=8,critic_batch=32,flow_steps=8,
                  curve_updates=(0,2,4),diagnostic_updates=(0,2,4))
        seeds=[raw['smoke_seed']]
    elif profile=='formal':seeds=raw['formal_seeds']
    else:raise ValueError(profile)
    name=f'02-qam-td-{profile}-{secrets.token_hex(5)}'
    output=ROOT/'runs'/name
    output.mkdir(parents=True,exist_ok=False)
    task=SparseReachTask(c)
    (output/'task.json').write_text(json.dumps(dict(B=task.B64.tolist(),radius=task.radius,
        design_seed=c.design_seed,reference_success_quantile=c.reference_success_quantile,
        condition_number=float(np.linalg.cond(task.B64))),indent=2))
    (output/'config-effective.json').write_text(json.dumps(c.__dict__,indent=2))
    (output/'provenance.json').write_text(json.dumps(dict(profile=profile,seeds=seeds,
        source_sha256=digest(ROOT/'notebooks/02_qam_td_terminal_error.py'),
        config_sha256=digest(CONFIG_PATH),official_qam_commit=raw['official_qam_commit'],
        old_runs_untouched=True,notebook1_untouched=True),indent=2))
    checks=verify_oracle(task)
    checks.to_csv(output/'oracle-checks.csv',index=False)
    all_A=[];all_B=[];all_M=[];all_D=[];summaries=[];final=[]
    for seed in seeds:
        sub=output/f'seed-{seed}'
        sub.mkdir()
        train=collect_episodes(task,c.train_episodes,seed)
        valid=collect_episodes(task,c.valid_episodes,seed_for(seed,20))
        eval_data=collect_episodes(task,max(c.eval_episodes_final,c.extract_states_final//c.horizon+1),seed_for(seed,30))
        for name,data in (('train',train),('valid',valid),('eval',eval_data)):
            np.savez_compressed(sub/f'{name}-transitions.npz',**{k:v.numpy() for k,v in data.items()})
        summary=data_summary(train,c)
        summaries.append(dict(seed=seed,**{k:v for k,v in summary.items() if k!='transitions_by_k'},
                              **{f'k{j}_count':n for j,n in summary['transitions_by_k'].items()}))
        (sub/'data-summary.json').write_text(json.dumps(summary,indent=2))
        if summary['successes'] in (0,c.train_episodes):
            print('WARNING: degenerate offline successes',seed,summary['successes'],flush=True)
        future=action_value_future_check(task,train,sub)
        base=train_base(train,c,seed,sub)
        critic,target,_=train_td_reference(task,train,valid,c,seed,sub)
        all_D.append(dict(seed=seed,**critic_diagnostics(task,critic,target,valid,c,seed)))
        choice=torch.randperm(len(eval_data['k']),generator=generator(seed_for(seed,40)))
        small=choice[:c.extract_states_curve];large=choice[:c.extract_states_final]
        small_states={x:eval_data[x][small] for x in ('k','y')}
        large_states={x:eval_data[x][large] for x in ('k','y')}
        learned,oracle,curve,mechanism=train_extraction(task,critic,base,train,small_states,c,seed,sub)
        all_A.extend(curve);all_M.extend(mechanism)
        joint,curveB=train_joint(task,critic,base,train,c,seed,sub)
        all_B.extend(curveB)
        for name,actor in (('A-learned',learned),('A-oracle',oracle)):
            final.append(dict(seed=seed,method=name,
                              **evaluate_checkpoint(task,actor,base,c,large_states,
                                                    seed_for(seed,920),c.eval_episodes_final)))
        final.append(dict(seed=seed,method='B-joint',
                          **evaluate_environment(task,joint,base,c,c.eval_episodes_final,
                                                 seed_for(seed,920))))
        pd.DataFrame(final).to_csv(output/'final-independent-evaluation.csv',index=False)
        print('seed completed',seed,'output',sub,flush=True)
    pd.DataFrame(all_A).to_csv(output/'experiment-A-curves.csv',index=False)
    pd.DataFrame(all_B).to_csv(output/'experiment-B-curves.csv',index=False)
    pd.DataFrame(all_M).to_csv(output/'experiment-A-mechanism.csv',index=False)
    pd.DataFrame(all_D).to_csv(output/'critic-A-diagnostics.csv',index=False)
    pd.DataFrame(summaries).to_csv(output/'data-summary.csv',index=False)
    plot_results(output)
    report(output,c,profile=='formal')
    print('OUTPUT',output,flush=True)
    return output


# %% [markdown]
# Execute this notebook with `PAGR_QAM_PROFILE=formal` for the pre-specified
# three-seed study. The default smoke profile checks the entire call chain but
# is never represented as formal evidence. Formal execution writes a unique run
# and never overwrites the archived reward-regression experiments.

# %%
if __name__ == '__main__':
    RUN_OUTPUT=run_experiment(os.environ.get('PAGR_QAM_PROFILE','smoke'))
    print((RUN_OUTPUT/'report.md').read_text()[:1800])
