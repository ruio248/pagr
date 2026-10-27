"""Terminal-consistent minimax Basic-AM. No oracle calls in update().

Exact Euler discrete full adjoint; stop-gradient matching labels. The local
metric is a positive diagonal lagged RMS preconditioner, NOT post-hoc Adam.
One global critic weight vector, constrained by KL and noisy validation MSE,
is shared across all trajectories and times in each update.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'am_hjbi_terminal_robustness'))
import robust_hamiltonian as rh


@dataclass
class Options:
    method: str = 'robust'
    rho: float = .3
    validation_slack: float = .1
    updates: int = 1000
    batch: int = 64
    lr: float = .002
    beta2: float = .99
    epsilon: float = .01
    max_update_norm: float = .3
    eval_every: int = 100
    final_paths: int = 4096


def coefficients(task, cfg):
    if task is rh.arm:
        return (1-cfg.dt*cfg.stiffness)*torch.eye(6), cfg.sigma*torch.eye(6)
    _, f, sigma = task.matrices(cfg)
    return f, sigma


def prediction(actor, paths, cfg, spatial_jacobian=False):
    batch, _, dim = paths.shape
    x = paths[:, :-1].detach().reshape(-1, dim).requires_grad_(spatial_jacobian)
    t = (torch.arange(cfg.steps)*cfg.dt)[None, :, None].expand(batch, -1, -1)
    u = actor(x, t.reshape(-1, 1))
    jac = None
    if spatial_jacobian:
        jac = torch.stack([torch.autograd.grad(u[:, j].sum(), x, retain_graph=True)[0]
                           for j in range(u.shape[-1])], 1).detach()
        jac = jac.reshape(batch, cfg.steps, u.shape[-1], dim)
    return u.reshape(batch, cfg.steps, -1), jac


def matching_gradients(actor, paths, terminals, task, cfg, full=True):
    """Gradient of integral matching loss; equals dJ/dtheta for full=True.

    a_k = F.T a_{k+1} + dt*l_x + dt*Du.T*(u_k + S.T*a_{k+1}).
    Labels use a_{k+1}, not a_k: this is the discrete Euler chain rule.
    """
    u, jac = prediction(actor, paths, cfg, spatial_jacobian=full)
    f, sigma = coefficients(task, cfg)
    running = (torch.zeros_like(paths[:, :-1]) if task is rh.arm else
               task.running_cost_and_gradient(paths[:, :-1], cfg)[1])
    with torch.no_grad():
        a = terminals.clone()
        targets = []
        for k in range(cfg.steps-1, -1, -1):
            residual = u[:, k].detach()[None] + a @ sigma
            targets.append(-a @ sigma)
            a = a @ f + cfg.dt*running[:, k][None]
            if full:
                a += cfg.dt*torch.einsum('mbc,bcs->mbs', residual, jac[:, k])
        targets = torch.stack(targets[::-1], 2)
    losses = .5*(u[None]-targets).square().sum(-1).sum(-1).mean(-1)*cfg.dt
    qs = torch.stack([rh.rel.flat_grad(loss, actor, retain_graph=i+1<len(losses))
                      for i, loss in enumerate(losses)])
    return qs.detach(), targets, losses.detach()


def validation_matrix(critics, data_path):
    data = np.load(data_path)
    x, y = torch.tensor(data['xvalid']), torch.tensor(data['yvalid'])
    with torch.no_grad():
        errors = torch.stack([c(x)-y for c in critics], 1).double().numpy()
    return errors.T @ errors / len(errors)


class Adversary:
    def __init__(self, value_gram, rho, slack=.1):
        self.m = len(value_gram)
        self.pi = np.ones(self.m)/self.m
        self.rho = rho
        self.value_gram = value_gram
        self.limit = float(self.pi @ value_gram @ self.pi)*(1+slack)+1e-10
        self.c = value_gram/max(self.limit, 1e-12)
        self.last = self.pi.copy()

    def kl(self, w):
        return float(np.sum(w*np.log(np.maximum(w, 1e-15)/self.pi)))

    def constraints(self):
        return [dict(type='eq', fun=lambda w:w.sum()-1, jac=lambda w:np.ones(self.m)),
                dict(type='ineq', fun=lambda w:self.rho-self.kl(w),
                     jac=lambda w:-(np.log(np.maximum(w, 1e-15)/self.pi)+1)),
                dict(type='ineq', fun=lambda w:1-w@self.c@w, jac=lambda w:-2*self.c@w)]

    def violation(self, w):
        return max(abs(w.sum()-1), max(0., -float(w.min())),
                   max(0., self.kl(w)-self.rho), max(0., float(w@self.c@w)-1))

    def solve(self, gram, certify=False):
        if self.rho == 0:
            return self.pi.copy(), dict(kl=0., violation=0., iterations=0, gap=0.)
        scale = max(float(np.diag(gram).max()), 1e-12)
        k = (gram+gram.T)*(.5/scale)
        fun = lambda w:.5*float(w@k@w)
        jac = lambda w:k@w
        result = minimize(fun, self.last, jac=jac, method='SLSQP',
                          bounds=[(1e-12, 1.)]*self.m, constraints=self.constraints(),
                          options=dict(ftol=1e-10, maxiter=100))
        if not result.success or self.violation(result.x)>1e-7:
            result = minimize(fun, self.pi, jac=jac, method='SLSQP',
                              bounds=[(1e-12, 1.)]*self.m, constraints=self.constraints(),
                              options=dict(ftol=1e-10, maxiter=300))
        if not result.success or self.violation(result.x)>1e-7:
            raise RuntimeError(f'Adversary solver failed: {result.message}; violation {self.violation(result.x)}')
        w = result.x
        assert fun(w) <= fun(self.pi)+1e-7
        self.last = w.copy()
        info = dict(kl=self.kl(w), violation=self.violation(w), iterations=int(result.nit),
                    validation_ratio=float(w@self.value_gram@w)/(self.pi@self.value_gram@self.pi),
                    weights=w.tolist())
        if certify:
            linear = k@w
            cert = minimize(lambda v:float(linear@v), self.pi, jac=lambda v:linear,
                            bounds=[(1e-12, 1.)]*self.m, constraints=self.constraints(),
                            method='SLSQP', options=dict(ftol=1e-11, maxiter=200))
            assert cert.success and self.violation(cert.x)<1e-7
            info['gap'] = max(0., float(linear@(w-cert.x)))
        return w, info


def update(actor, optimizer, paths, critics, task, cfg, options, state, adversary, step):
    terminals = torch.stack([task.raw_gradient(c, paths[:, -1]) for c in critics])
    # Full/lean native Adam controls need only the shared mean terminal label.
    if options.method in ('am', 'full_adam'):
        terminals = terminals.mean(0, keepdim=True)
    qs, _, losses = matching_gradients(actor, paths, terminals, task, cfg, full=options.method!='am')
    mean = qs.mean(0)
    info = dict(matching_loss=losses.mean().item(), gradient_norm=mean.norm().item())
    if options.method in ('am', 'full_adam'):
        optimizer.zero_grad()
        offset = 0
        for p in actor.parameters():
            p.grad = mean[offset:offset+p.numel()].reshape_as(p).clone()
            offset += p.numel()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.)
        optimizer.step()
        return info
    # R is fixed before observing this batch's q. No first-moment momentum.
    h = (state['v']/(1-options.beta2**step)).sqrt()+options.epsilon if step else torch.ones_like(mean)
    weights = np.ones(len(qs))/len(qs)
    if options.method in ('robust', 'norm_matched'):
        gram = ((qs.double()/h.double())@qs.double().T).numpy()
        weights, diagnostic = adversary.solve(gram, certify=step%50==0)
        info.update(diagnostic)
    qstar = torch.tensor(weights, dtype=qs.dtype)@qs
    plain = -options.lr*mean/h
    delta = -options.lr*qstar/h
    if options.method == 'norm_matched':
        delta = plain*(delta.norm()/plain.norm().clamp_min(1e-12))
    alpha = min(1., options.max_update_norm/max(delta.norm().item(), 1e-12))
    delta *= alpha
    rh.rel.apply_step(actor, delta)
    state['v'].mul_(options.beta2).addcmul_(mean, mean, value=1-options.beta2)
    info.update(step_norm=delta.norm().item(), step_cap_factor=alpha,
                direction_cosine=torch.nn.functional.cosine_similarity(delta, plain, dim=0).item(),
                norm_ratio=delta.norm().item()/max(plain.norm().item(), 1e-12))
    return info


def train(name, cfg, seed, critics, value_gram, options, folder):
    task = rh.get_task(name)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed+300000)
    actor = task.Controller(cfg.actor_width)
    optimizer = torch.optim.Adam(actor.parameters(), lr=options.lr)
    state = dict(v=torch.zeros(sum(p.numel() for p in actor.parameters())))
    adversary = Adversary(value_gram, options.rho, options.validation_slack)
    generator = task.rng(seed+400000)
    history, diagnostics = [], []
    start = time.monotonic()
    for step in range(options.updates+1):
        if step%options.eval_every == 0 or step == options.updates:
            history.append(dict(step=step, **task.evaluate(actor, cfg, seed+500000, n=1024)))
        if step in (0, options.updates//2, options.updates):
            torch.save(dict(actor=actor.state_dict(), state=state, step=step), folder/f'checkpoint_{step}.pt')
        if step == options.updates:
            break
        paths, _ = task.rollout(actor, cfg, options.batch, generator)
        info = update(actor, optimizer, paths, critics, task, cfg, options, state, adversary, step)
        if step%25 == 0:
            diagnostics.append(dict(step=step+1, **info))
    final, arrays = task.evaluate(actor, cfg, seed+600000, n=options.final_paths, keep_paths=True)
    fine = task.evaluate(actor, cfg, seed+700000, n=2048, steps=2*cfg.steps)
    torch.save(actor.state_dict(), folder/'actor.pt')
    np.savez_compressed(folder/'evaluation.npz', **arrays)
    result = dict(task=name, seed=seed, config=asdict(cfg), options=asdict(options), final=final,
                  fine_grid=fine, history=history, diagnostics=diagnostics, seconds=time.monotonic()-start,
                  actor_training_rollouts=options.updates*options.batch,
                  additional_scalar_cost_queries=0, training_true_gradient_calls=0)
    (folder/'result.json').write_text(json.dumps(result, indent=2))
    print(name, seed, options.method, options.rho, 'J', round(final['cost'], 6),
          'seconds', round(result['seconds'], 1), flush=True)
    return result
