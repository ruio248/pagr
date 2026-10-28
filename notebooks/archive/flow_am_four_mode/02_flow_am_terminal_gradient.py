# %% [markdown]
# # Notebook 2 — pretrained Flow AM / terminal-gradient reliability
#
# **noise→action generation，不是物理导航。** 固定单一条件的二维终止 bandit，
# $Q_{true}(s,a)=R(a)$。先真实训练行为 Flow，再冻结 base 和五成员 scalar reward
# ensemble。研究终端梯度误差→伴随→matching 标签→参数梯度→Adam 步→有限预算性能。
# 本平台不是完整 Q-learning、TRQAM、OGBench，也不预设 oracle 必须胜出。
# 旧 [导航 SOC notebook](archive/02_navigation_am_and_candidates.ipynb) 原样归档；
# Notebook 1 与历史资产/结论保持不变。此源码包含完整实验，不调用历史 replay。
#
# 依据：[A, §4/5.2, Algorithm 1, G.1](https://arxiv.org/html/2409.08861v5)，
# [B, §3–4](https://arxiv.org/html/2601.14234v3)。公式以下按本实验独立推导。
# 官方实现核对对象：[ColinQiyangLi/qam, agents/qam.py](https://github.com/ColinQiyangLi/qam/blob/2726d767c9a0a7a46d49693f0391f73dc2cf58ac/agents/qam.py)，
# commit `2726d767c9a0a7a46d49693f0391f73dc2cf58ac`，`adj_matching` / `actor_loss`。
# 这是审计参考，不是逐行复现；具体差异见下面的端点表。

# %%
import copy
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Markdown, display
from scipy.stats import t as student_t
from torch import nn


@dataclass(frozen=True)
class FlowConfig:
    profile: str = 'formal'
    seeds: tuple = (28001, 28002, 28003)
    seed_scheme: str = 'root_seed * 10000000 + role_offset'
    means: tuple = ((-1.5, -1.5), (-1.5, 1.5), (1.5, -1.5), (1.5, 1.5))
    weights: tuple = (.25, .25, .25, .25)
    behavior_std: float = .4
    reward_heights: tuple = (.2, .8, 1.4, 2.)
    reward_width: float = .85
    reward_quadratic: float = .025
    observation_std: float = .2
    behavior_train: int = 4096
    behavior_valid: int = 2048
    reward_train: int = 512
    reward_valid: int = 256
    members: int = 5
    width: int = 64
    fm_updates: int = 2500
    fm_batch: int = 256
    fm_lr: float = .001
    fm_validate_every: int = 100
    reward_updates: int = 1800
    reward_batch: int = 128
    reward_lr: float = .001
    reward_validate_every: int = 100
    beta: float = 1.5
    epsilon: float = .05
    warmup_steps: int = 8
    steps: int = 32
    ode_steps: int = 128
    actor_updates: int = 300
    actor_batch: int = 64
    actor_lr: float = .0003
    adam_betas: tuple = (.9, .999)
    adam_eps: float = 1e-8
    clip: float = 5.
    curve_every: int = 50
    diagnostic_steps: tuple = (0, 150, 300)
    curve_paths: int = 768
    final_paths: int = 4096
    diagnostic_paths: int = 256
    virtual_paths: int = 2048
    conditional_futures: int = 256
    conditional_states: int = 3
    conditional_fractions: tuple = (.25, .5, .75)
    fine_retrain_first_seed: bool = True


def configuration(profile='formal'):
    c = FlowConfig()
    if profile == 'smoke':
        c = replace(c, profile=profile, seeds=(91001,), fm_updates=100,
                    reward_updates=100, actor_updates=4, steps=8,
                    diagnostic_steps=(0, 2, 4), curve_every=2, fm_validate_every=50,
                    reward_validate_every=50, curve_paths=64, final_paths=128,
                    diagnostic_paths=24, virtual_paths=64, conditional_futures=24,
                    conditional_states=2)
    elif profile != 'formal':
        raise ValueError(profile)
    return c


def generator(seed):
    return torch.Generator(device='cpu').manual_seed(int(seed))


def seed_key(root_seed, role_offset):
    # Disjoint integer namespaces across independent training seeds. A bare
    # root_seed+offset would overlap adjacent seeds' data/member/update streams.
    if not 0 <= role_offset < 10000000:
        raise ValueError('RNG role offset exceeds its reserved namespace')
    return int(root_seed)*10000000+int(role_offset)


def ensure_finite(name, *xs):
    for x in xs:
        if not torch.isfinite(x).all():
            raise FloatingPointError(f'{name}: non-finite values; no paths discarded')


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_hash(model):
    return hashlib.sha256(''.join(tensor_hash(p) for p in model.state_dict().values()).encode()).hexdigest()


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    return model


def actor_copy(base):
    actor = copy.deepcopy(base)
    for p in actor.parameters():
        p.requires_grad_(True)
    return actor


def flat_parameters(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()


def flat_gradient(loss, model, retain_graph=False):
    return torch.cat([x.reshape(-1) for x in torch.autograd.grad(
        loss, tuple(model.parameters()), retain_graph=retain_graph)]).detach()


def cosine(a, b):
    # Undefined zero-norm cosine is None, never silently called perfect alignment.
    a, b = a.flatten().double(), b.flatten().double()
    d = float(a.norm()*b.norm())
    return float(a@b/d) if d > 1e-20 else None


def mean_se(x):
    x = x.detach().double()
    return float(x.mean()), float(x.std(unbiased=True)/math.sqrt(x.numel()))


torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
CFG = configuration(os.environ.get('PAGR_FLOW_PROFILE', 'formal'))
ROOT = Path.cwd() if (Path.cwd()/'pyproject.toml').exists() else Path.cwd().parent
assert (ROOT/'pyproject.toml').exists(), 'Run from the repository or notebooks directory'
(ROOT/'runs').mkdir(exist_ok=True)
OUT = Path(tempfile.mkdtemp(prefix=f'02-flow-am-{CFG.profile}-', dir=ROOT/'runs'))
CONFIG_SHA = hashlib.sha256(json.dumps(asdict(CFG), sort_keys=True).encode()).hexdigest()
save_json(OUT/'config.json', asdict(CFG))
save_json(OUT/'environment.json', dict(python=sys.version, platform=platform.platform(),
          torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
          threads=torch.get_num_threads(), device='cpu', config_sha256=CONFIG_SHA,
          source_sha256=hashlib.sha256((ROOT/'notebooks/02_flow_am_terminal_gradient.py').read_bytes()).hexdigest()))
print('New run:', OUT.relative_to(ROOT), '\nPre-execution config SHA256:', CONFIG_SHA)
display(pd.DataFrame.from_dict(asdict(CFG), orient='index', columns=['Fixed protocol']))

# %% [markdown]
# ## 1. 数学对象、端点与离散目标
#
# | 对象 | 理想连续定义（列向量） | 实际代码 |
# |---|---|---|
# | 生成方向 / 起点 / 终点 | $t:0\to1$, $Z\sim N(0,I_2)$, $a=X_1$ | `noise`, `rollout` |
# | base / fine velocity | $v_b$, $v_\theta$ | FM-trained `Velocity`, exact full copy |
# | diffusion | $\nu^2=2(1-t)/t$ | `grid`: only $\epsilon\le t_k<1$ |
# | base drift | $b_b=2v_b-x/t$ | `2*base-x/t` |
# | control | $u=2(v_\theta-v_b)/\nu$ | `controls` / `matching_loss` |
# | controlled drift | $b_b+\nu u=2v_\theta-x/t$ | `rollout` |
# | terminal cost | $g=-\beta R$ or $-\beta\hat R$ | `terminal_label` |
# | lean adjoint | $-\dot p=(D_xb_b)^Tp$, $p_1=\nabla g$ | `lean_adjoint` |
# | matching | $\frac12E\int\|u+\nu p\|^2dt$ | local `matching_loss`, detached paths/labels |
# | actual grid | warmup $[0,\epsilon]$, then $t_k=\epsilon+kh$ | $h=(1-\epsilon)/K$, $k=0..K-1$ |
#
# **Endpoint audit.** [A] G.1 regularizes the diffusion on its uniform grid;
# the pinned official QAM implementation uses $\sigma_h^2=2(1-t+h)/(t+h)$,
# drift $2v-x/(t+h)$ and a last **base ODE** step. Its backward time arguments
# also differ from its forward ones. We record these conventions, not label them bugs.
# Here the sampler is deliberately different: shared **frozen-base RK2 warmup**
# to $\epsilon=.05$, followed by K Euler–Maruyama Gaussian transitions, including
# the final transition. Terminal gradient is evaluated at the actual noisy $X_K$.
# There is no denoising replacement, clipping of actions, or deterministic last step.
# No division at 0 or 1; $\nu(t_{K-1})>0$. No sigma-only modification is made.
# Omission of controlled evolution before epsilon, finite EM/RK2 steps and learned
# FM error prevent any claim of exact memoryless / exponential-tilt recovery.
# B's HTML Eq.5 has a plus sign inconsistent with Eq.7; our convention is Eq.7:
# the linear-bridge score is $(t v-x)/(1-t)$, giving $b=v+\nu^2\mathrm{score}/2=2v-x/t$.
#
# **Discrete lean pairing (not full BPTT).** Freeze a sampled path and define
# $F_k=I+h[2D_xv_b(X_k,t_k)-I/t_k]$;
# $p_k=F_k^Tp_{k+1}$, with terminal $p_K=-\beta\nabla R(X_K)$.
# The local update at $X_k$ pairs with **$p_{k+1}$**, i.e. the next-state sensitivity
# of a discrete transition. Paper same-time pairing is another continuous-limit
# discretization; we do not claim these finite-step algorithms identical.
#
# $L=\frac h{2B}\sum_{i,k}\|2(v_\theta-v_b)/\nu_k+\nu_k p_{k+1}\|^2$.
# If using $v_{target}=v_b-\nu_k^2p_{k+1}/2$, its exact weight is $2h/\nu_k^2$.
# There is **no extra energy loss**. Lean matching's parameter derivative is not
# generally $\nabla_\theta J_{true}$; no actor state Jacobian is propagated.
#
# **Finite-chain KL (proved for this implemented process, not ODE action KL):**
# transition mean $m_\theta=x+h(2v_\theta-x/t)$, covariance $h\nu_k^2I$.
# The common warmup distribution cancels in the chain rule; each conditional KL is
# $\|m_\theta-m_b\|^2/(2h\nu_k^2)=\frac h2\|u\|^2$.
# Thus $D_{KL}(P_\theta^{grid}\|P_b^{grid})=E_{P_\theta}\sum_k\frac h2\|u_k\|^2$.
# The identity is exact for this chain; reported expectations are Monte Carlo estimates.
# $J_{true}^{grid}=E[\sum_k\frac h2\|u_k\|^2-\beta R(X_K)]$.
# It is not the full ideal time-0 SOC, nor a measured ODE action-distribution KL.

# %%
class Velocity(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, width), nn.SiLU(), nn.Linear(width, width),
                                 nn.SiLU(), nn.Linear(width, 2))

    def forward(self, x, t):
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device).expand(*x.shape[:-1], 1)
        return self.net(torch.cat((x, t, torch.sin(math.pi*t), torch.cos(math.pi*t)), -1))


class RewardModel(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width),
                                 nn.SiLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def behavior_sample(c, n, seed):
    gen = generator(seed)
    indices = torch.multinomial(torch.tensor(c.weights), n, replacement=True, generator=gen)
    return torch.tensor(c.means)[indices]+c.behavior_std*torch.randn(n, 2, generator=gen)


def true_reward(x, c):
    means = x.new_tensor(c.means)
    d2 = (x[..., None, :]-means).square().sum(-1)
    return (torch.exp(-d2/(2*c.reward_width**2))*x.new_tensor(c.reward_heights)).sum(-1) \
        - c.reward_quadratic*x.square().sum(-1)


def scalar_gradient(fn, x):
    with torch.enable_grad():
        z = x.detach().requires_grad_(True)
        answer = torch.autograd.grad(fn(z).sum(), z)[0]
    ensure_finite('scalar_gradient', answer)
    return answer.detach()


def true_gradient(x, c):
    return scalar_gradient(lambda z: true_reward(z, c), x)


@contextmanager
def forbid_truth():
    saved = {k: globals()[k] for k in ('true_reward', 'true_gradient')}
    def forbidden(*args, **kwargs):
        raise AssertionError('learned training cannot query true reward or gradient')
    globals().update({k: forbidden for k in saved})
    try:
        yield
    finally:
        globals().update(saved)


def make_data(c, seed):
    data = dict(behavior=behavior_sample(c, c.behavior_train, seed_key(seed, 10)),
                heldout=behavior_sample(c, c.behavior_valid, seed_key(seed, 11)),
                reward_x=behavior_sample(c, c.reward_train, seed_key(seed, 12)),
                valid_x=behavior_sample(c, c.reward_valid, seed_key(seed, 13)))
    for x, y, offset in [('reward_x', 'reward_y', 14), ('valid_x', 'valid_y', 15)]:
        data[y] = true_reward(data[x], c)+c.observation_std*torch.randn(len(data[x]), generator=generator(seed_key(seed, offset)))
    return data


def fm_examples(actions, n, gen):
    a = actions[torch.randint(len(actions), (n,), generator=gen)]
    z, t = torch.randn(n, 2, generator=gen), torch.rand(n, 1, generator=gen)
    return (1-t)*z+t*a, t, a-z


def fit_base(c, data, seed):
    torch.manual_seed(seed_key(seed, 20))
    model = Velocity(c.width)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.fm_lr)
    gen = generator(seed_key(seed, 21))
    xv, tv, yv = fm_examples(data['heldout'], c.behavior_valid, generator(seed_key(seed, 22)))
    history, best, selected, checkpoint = [], float('inf'), -1, None
    for step in range(c.fm_updates+1):
        if step % c.fm_validate_every == 0 or step == c.fm_updates:
            with torch.no_grad():
                valid = float((model(xv, tv)-yv).square().sum(-1).mean())
            if not math.isfinite(valid):
                raise FloatingPointError('FM validation non-finite')
            history.append(dict(stage='fm', member=-1, step=step, valid_mse=valid))
            if valid < best:  # fixed rule: minimum held-out FM loss, ties earliest
                best, selected, checkpoint = valid, step, copy.deepcopy(model.state_dict())
        if step == c.fm_updates:
            break
        x, t, target = fm_examples(data['behavior'], c.fm_batch, gen)
        loss = (model(x, t)-target).square().sum(-1).mean()
        ensure_finite('FM loss', loss)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.load_state_dict(checkpoint)
    return freeze(model), history, dict(selected_step=selected, validation_loss=best)


def fit_rewards(c, data, seed):
    models, histories, selections = [], [], []
    for member in range(c.members):
        torch.manual_seed(seed_key(seed, 100+member))
        model = RewardModel(c.width)
        optimizer = torch.optim.Adam(model.parameters(), lr=c.reward_lr)
        gen = generator(seed_key(seed, 200+member))
        # Bootstrap fixed observed training pairs; no new truth queries.
        idx = torch.randint(c.reward_train, (c.reward_train,), generator=gen)
        x, y = data['reward_x'][idx], data['reward_y'][idx]
        best, selected, checkpoint = float('inf'), -1, None
        for step in range(c.reward_updates+1):
            if step % c.reward_validate_every == 0 or step == c.reward_updates:
                with torch.no_grad():
                    valid = float((model(data['valid_x'])-data['valid_y']).square().mean())
                if not math.isfinite(valid):
                    raise FloatingPointError('reward validation non-finite')
                histories.append(dict(stage='reward', member=member, step=step, valid_mse=valid))
                if valid < best:
                    best, selected, checkpoint = valid, step, copy.deepcopy(model.state_dict())
            if step == c.reward_updates:
                break
            batch = torch.randint(len(x), (c.reward_batch,), generator=gen)
            loss = (model(x[batch])-y[batch]).square().mean()
            ensure_finite('reward loss', loss)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.load_state_dict(checkpoint)
        models.append(freeze(model))
        selections.append(dict(member=member, selected_step=selected, validation_loss=best))
    return models, histories, selections


def reward_mean(models, x):
    return torch.stack([m(x) for m in models]).mean(0)


def terminal_label(models, x, c, oracle=False):
    gradient = true_gradient(x, c) if oracle else scalar_gradient(lambda z: reward_mean(models, z), x)
    return -c.beta*gradient


def grid(c, dtype=torch.float32):
    h = (1-c.epsilon)/c.steps
    ts = c.epsilon+h*torch.arange(c.steps, dtype=dtype)
    return ts, torch.sqrt(2*(1-ts)/ts), h


def noise(c, n, seed, steps=None):
    # Separate initial/future streams, sample-major draws: 2x batch shares first B.
    # Fine increments pair to coarse increments for K/2K sensitivity.
    steps = steps or c.steps
    common_steps = 2*c.steps
    if steps not in (c.steps, common_steps):
        raise ValueError('Only the pre-specified K or 2K grids')
    # Fixed-size blocks also prevent CPU randn's shape-dependent vectorization
    # from breaking prefixes for small/non-aligned diagnostic batches.
    gz, gf = generator(seed), generator(seed+1)
    blocks = math.ceil(n/64)
    z = torch.cat([torch.randn(64, 2, generator=gz) for _ in range(blocks)])[:n]
    fine = torch.cat([torch.randn(64, common_steps, 2, generator=gf) for _ in range(blocks)])[:n]
    xi = fine if steps == common_steps else fine.reshape(n, c.steps, 2, 2).sum(2)/math.sqrt(2)
    return z, xi


@torch.no_grad()
def ode_sample(model, z, steps, start=0., end=1.):
    x, h = z.detach().clone(), (end-start)/steps
    for k in range(steps):
        t = start+k*h
        mid = x+.5*h*model(x, t)
        x = x+h*model(mid, t+.5*h)
    ensure_finite('ODE', x)
    return x


@torch.no_grad()
def rollout(actor, base, c, z, xi, start_index=0, initial_state=None):
    ts, nu, h = grid(c, z.dtype)
    if initial_state is None:
        assert start_index == 0
        x = ode_sample(base, z, c.warmup_steps, end=c.epsilon)
    else:
        x = initial_state.detach().clone()
    states, controls = [x], []
    for k in range(start_index, c.steps):
        v, vb = actor(x, ts[k]), base(x, ts[k])
        controls.append(2*(v-vb)/nu[k])
        x = x+h*(2*v-x/ts[k])+math.sqrt(h)*nu[k]*xi[:, k]
        states.append(x)
    paths, controls = torch.stack(states, 1).detach(), torch.stack(controls, 1).detach()
    ensure_finite('SDE rollout', paths, controls)
    return paths, controls


def base_jacobians(base, paths, c, start_index=0):
    # Freeze weights, not input differentiation. In 2D two VJPs recover all rows.
    ts, _, h = grid(c, paths.dtype)
    x = paths[:, :-1].detach().requires_grad_(True)
    t = ts[start_index:][None, :, None].expand(len(x), -1, 1)
    with torch.enable_grad():
        v = base(x, t)
        rows = [torch.autograd.grad(v[..., j].sum(), x, retain_graph=j == 0)[0] for j in range(2)]
    jac = torch.stack(rows, -2).detach()  # [..., output, input]
    eye = torch.eye(2, dtype=x.dtype)
    f = eye+h*(2*jac-eye/ts[start_index:][None, :, None, None])
    ensure_finite('base Jacobians', f)
    return f.detach()


@torch.no_grad()
def lean_adjoint(f, terminal):
    p, values = terminal.detach(), [terminal.detach()]
    for k in range(f.shape[1]-1, -1, -1):
        p = torch.einsum('bij,bj->bi', f[:, k].transpose(-1, -2), p)
        values.append(p)
    result = torch.stack(values[::-1], 1).detach()
    ensure_finite('lean adjoint', result)
    return result


def matching_loss(actor, base, paths, p, c):
    x = paths[:, :-1].detach()
    ts, nu, h = grid(c, x.dtype)
    t = ts[None, :, None].expand(len(x), -1, 1)
    with torch.no_grad():
        vb = base(x, t)
    control = 2*(actor(x, t)-vb)/nu[None, :, None]
    residual = control+nu[None, :, None]*p[:, 1:].detach()
    return .5*h*residual.square().sum((-1, -2)).mean()


def adam_for(actor, c):
    return torch.optim.Adam(actor.parameters(), lr=c.actor_lr, betas=c.adam_betas, eps=c.adam_eps)


def apply_gradient(actor, optimizer, q, c):
    ensure_finite('parameter gradient', q)
    optimizer.zero_grad(set_to_none=True)
    offset = 0
    for p in actor.parameters():
        p.grad = q[offset:offset+p.numel()].reshape_as(p).clone()
        offset += p.numel()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), c.clip, error_if_nonfinite=True)
    clipped = torch.cat([p.grad.flatten() for p in actor.parameters()]).detach().clone()
    optimizer.step()
    ensure_finite('updated actor', flat_parameters(actor))
    return clipped


def actor_update(actor, optimizer, base, models, c, z, xi, oracle=False):
    paths, _ = rollout(actor, base, c, z, xi)
    f = base_jacobians(base, paths, c)
    p = lean_adjoint(f, terminal_label(models, paths[:, -1], c, oracle))
    loss = matching_loss(actor, base, paths, p, c)
    q = flat_gradient(loss, actor)
    clipped = apply_gradient(actor, optimizer, q, c)
    return dict(loss=float(loss.detach()), gradient_norm=float(q.norm()), clipped_norm=float(clipped.norm()))

# %% [markdown]
# ## 2. 数据、预训练、固定预算与对照
#
# 四个已有模式，等权 Gaussian，std=0.4。真实平滑奖励为宽度0.85的四个
# Gaussian bump（高度0.2/0.8/1.4/2）减 $0.025\|a\|^2$，没有高频项。
# 每 seed 512个训练+256个验证 scalar observations，独立 $N(0,0.2^2)$ 噪声。
# 五个 SiLU 模型使用相同观测的 bootstrap，分别按 noisy validation MSE 选最早最小值。
# FM 用4096行为样本、2048 held-out bridge samples，选最早最小 held-out FM loss。
# **任何模型选择均不使用 oracle evaluation reward 或方法胜负。**
#
# 正式 seeds=28001/28002/28003；独立 smoke=91001。全部固定300次 Adam更新：
# frozen（0次）、AM-learned（batch64）、AM-oracle（batch64）、AM-learned-2x（batch128）。
# Oracle仅替换终端梯度；不是可部署方法或上界。2x为双倍轨迹预算，不是同预算新算法。
# 每 seed 共享数据、base、初始参数、可耦合噪声；策略分化后状态路径不相同。
# 子流采用 `root_seed*10000000+role_offset`，避免相邻训练 seed 的加法偏移碰撞。
# 数据10–15；FM20–22；reward成员100+ / 200+；训练100000+；曲线300000；
# 诊断400000+；虚拟步评估500000+；conditional600000+；最终评估700000。
# 首次正式运行后检查发现旧加法 seed 方案有交叉重合，该运行原样保留但不用于最终统计；
# 此修正只改变子流隔离，不改变场景、正式 seed 编号、模型或优化超参数，详见 development JSON。
# Actor 最终 checkpoint 为主结果，不按真实回报挑最好步。
# 共同诊断状态是 AM-learned 的 step0/150/300；在该 actor 上只替换标签做对照。
# K→2K：所有最终模型仅细化评估；另在**第一个预定 seed**重新训练 learned/oracle。
# 双倍网格重训只是一 seed 数值敏感性，不当成额外三 seed 结果。
#
# 各种 evaluation / diagnosis 的真实奖励查询都在 guard 外；learned 更新在
# `forbid_truth()` 中实际运行，模型和 base 参数哈希必须不变。

# %%
@torch.no_grad()
def mode_metrics(x, c):
    distance = torch.cdist(x, x.new_tensor(c.means))
    near, index = distance.min(1)
    result = {f'mode_{j}': float((index == j).float().mean()) for j in range(4)}
    result.update(within_radius_08=float((near <= .8).float().mean()), nearest_mode_distance=float(near.mean()))
    return result


@torch.no_grad()
def sliced_distance(x, y, seed=61923):
    # Fixed 32 directions, empirical equal-size sliced W2; not a KL estimator.
    n = min(len(x), len(y))
    directions = torch.randn(32, 2, generator=generator(seed))
    directions = directions/directions.norm(dim=1, keepdim=True)
    a, b = (x[:n]@directions.T).sort(0).values, (y[:n]@directions.T).sort(0).values
    return float((a-b).square().mean().sqrt())


@torch.no_grad()
def evaluate(actor, base, c, z, xi):
    paths, controls = rollout(actor, base, c, z, xi)
    _, nu, h = grid(c)
    energy = .5*h*controls.square().sum((-1, -2))
    # Independently compute Gaussian KL from mean/covariance, not controls.
    ts, _, _ = grid(c)
    x = paths[:, :-1]
    t = ts[None, :, None].expand(len(x), -1, 1)
    mean_actor = x+h*(2*actor(x, t)-x/t)
    mean_base = x+h*(2*base(x, t)-x/t)
    conditional_kl = ((mean_actor-mean_base).square().sum(-1)/(2*h*nu.square())).sum(-1)
    kl_residual = float((energy-conditional_kl).abs().max())
    if not torch.allclose(energy, conditional_kl, atol=2e-5, rtol=2e-5):
        raise AssertionError('Gaussian-transition KL / energy mismatch')
    reward = true_reward(paths[:, -1], c)
    cost = energy-c.beta*reward
    result = {}
    for name, values in [('reward', reward), ('cost', cost), ('energy', energy)]:
        result[name], result[name+'_mc_se'] = mean_se(values)
    result.update(mode_metrics(paths[:, -1], c), kl_identity_max_abs=kl_residual)
    return result, dict(paths=paths, reward=reward, cost=cost, energy=energy)


def projection_audit(actor, base, paths, p_learned, p_oracle, c):
    ql = flat_gradient(matching_loss(actor, base, paths, p_learned, c), actor)
    qo = flat_gradient(matching_loss(actor, base, paths, p_oracle, c), actor)
    ts, _, h = grid(c, paths.dtype)
    x = paths[:, :-1].detach()
    v = actor(x, ts[None, :, None].expand(len(x), -1, 1))
    # nu cancels: delta q = (2h/B) sum D_theta v^T delta p_next.
    projected = flat_gradient(2*h*(v*(p_learned[:, 1:]-p_oracle[:, 1:]).detach()).sum((-1, -2)).mean(), actor)
    error = float((ql-qo-projected).norm())
    if not torch.allclose(ql-qo, projected, atol=2e-5, rtol=2e-4):
        raise AssertionError('label-error / actor-Jacobian projection identity')
    return ql, qo, error


@torch.no_grad()
def propagation_products(f):
    b, k = f.shape[:2]
    product = torch.eye(2, dtype=f.dtype).expand(b, 2, 2).clone()
    products = [product]
    for i in range(k-1, -1, -1):
        product = f[:, i].transpose(-1, -2)@product
        products.append(product)
    return torch.stack(products[::-1], 1)


def conditional_diagnostics(actor, base, models, c, paths, seed, step):
    rows = []
    for fraction in c.conditional_fractions:
        k = int(c.steps*fraction)
        for state_index in range(c.conditional_states):
            z, xi = noise(c, c.conditional_futures, seed_key(seed, 600000+step*100+1000*k+2*state_index))
            initial = paths[state_index, k].expand(c.conditional_futures, -1)
            future, _ = rollout(actor, base, c, z, xi, start_index=k, initial_state=initial)
            f = base_jacobians(base, future, c, start_index=k)
            d = terminal_label(models, future[:, -1], c)-terminal_label(models, future[:, -1], c, True)
            dp = lean_adjoint(f, d)
            _, nu, _ = grid(c)
            # Matching residual-label error at X_k: nu_k delta p_{k+1}.
            error = nu[k]*dp[:, 1]
            mean = error.mean(0)
            variance = error.var(0, unbiased=True)
            rows.append(dict(seed=seed, checkpoint=step, time=float(grid(c)[0][k]),
                             state_index=state_index, state_x=float(initial[0, 0]), state_y=float(initial[0, 1]),
                             futures=c.conditional_futures, conditional_mean_x=float(mean[0]),
                             conditional_mean_y=float(mean[1]), conditional_mean_norm=float(mean.norm()),
                             variance_trace=float(variance.sum()), mean_mc_se_x=float((variance[0]/len(error)).sqrt()),
                             mean_mc_se_y=float((variance[1]/len(error)).sqrt()),
                             pathwise_rms=float(error.square().sum(-1).mean().sqrt())))
    return rows


def diagnostics(actor, optimizer, base, models, c, seed, step, directory):
    before = model_hash(actor)
    z, xi = noise(c, c.diagnostic_paths, seed_key(seed, 400000+2*step))
    paths, _ = rollout(actor, base, c, z, xi)
    endpoint = paths[:, -1]
    f = base_jacobians(base, paths, c)
    zl, zo = terminal_label(models, endpoint, c), terminal_label(models, endpoint, c, True)
    pl, po = lean_adjoint(f, zl), lean_adjoint(f, zo)
    products = propagation_products(f)
    expected = torch.einsum('bkij,bj->bki', products, zl-zo)
    propagation_error = float((pl-po-expected).abs().max())
    if not torch.allclose(pl-po, expected, atol=2e-5, rtol=2e-4):
        raise AssertionError('discrete propagation product mismatch')
    ql, qo, projection_error = projection_audit(actor, base, paths, pl, po, c)
    with torch.no_grad():
        truth = true_reward(endpoint, c)
        residual = reward_mean(models, endpoint)-truth
        member_g = torch.stack([scalar_gradient(m, endpoint) for m in models])
    grad_error = (zl-zo)/c.beta
    relative = float(grad_error.square().sum(-1).mean().sqrt()/(zo/c.beta).square().sum(-1).mean().sqrt())
    rmse = float(residual.square().mean().sqrt())
    row = dict(seed=seed, checkpoint=step, value_rmse=rmse,
               value_std=float(truth.std()), value_rmse_over_true_std=rmse/float(truth.std()),
               gradient_rmse=float(grad_error.square().sum(-1).mean().sqrt()), gradient_relative=relative,
               gradient_cosine=cosine(zl, zo), gradient_cosine_mean=float(torch.nn.functional.cosine_similarity(zl, zo).mean()),
               ensemble_gradient_variance=float(member_g.var(0, unbiased=True).sum(-1).mean()),
               q_error=float((ql-qo).norm()), q_relative=float((ql-qo).norm()/qo.norm()),
               q_cosine=cosine(ql, qo), q_norm_ratio=float(ql.norm()/qo.norm()),
               q_learned_norm=float(ql.norm()), q_oracle_norm=float(qo.norm()),
               projection_residual=projection_error, propagation_residual=propagation_error)
    ts, nu, _ = grid(c)
    time_rows = []
    sv = torch.linalg.svdvals(products)
    for k in range(c.steps+1):
        d = pl[:, k]-po[:, k]
        time_row = dict(seed=seed, checkpoint=step, index=k, time=float(ts[k]) if k < c.steps else 1.,
                        adjoint_error_rms=float(d.square().sum(-1).mean().sqrt()),
                        adjoint_relative=float(d.norm()/po[:, k].norm()),
                        gain_max_mean=float(sv[:, k, 0].mean()), gain_min_mean=float(sv[:, k, 1].mean()),
                        gain_max_p95=float(torch.quantile(sv[:, k, 0], .95)))
        if k < c.steps:
            label = nu[k]*(pl[:, k+1]-po[:, k+1])
            time_row.update(label_error_rms=float(label.square().sum(-1).mean().sqrt()),
                            label_relative=float(label.norm()/(nu[k]*po[:, k+1]).norm()))
        time_rows.append(time_row)
    # Same Adam history, cloned twice. No state or RNG of the real optimizer is mutated.
    clones, updates, clipped = [], [], []
    for q in (ql, qo):
        clone = copy.deepcopy(actor)
        opt = adam_for(clone, c)
        opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        clipped.append(apply_gradient(clone, opt, q, c))
        updates.append(flat_parameters(clone)-flat_parameters(actor))
        clones.append(clone)
    row.update(q_clipped_cosine=cosine(*clipped), q_clipped_learned_norm=float(clipped[0].norm()),
               q_clipped_oracle_norm=float(clipped[1].norm()), step_cosine=cosine(*updates),
               step_norm_ratio=float(updates[0].norm()/updates[1].norm()),
               step_learned_norm=float(updates[0].norm()), step_oracle_norm=float(updates[1].norm()))
    vz, vxi = noise(c, c.virtual_paths, seed_key(seed, 500000+2*step))
    _, original = evaluate(actor, base, c, vz, vxi)
    virtual_arrays = {}
    for name, clone in zip(('learned', 'oracle'), clones):
        _, evaluated = evaluate(clone, base, c, vz, vxi)
        for metric in ('reward', 'cost'):
            d = evaluated[metric]-original[metric]
            row[f'virtual_{name}_{metric}_delta'], row[f'virtual_{name}_{metric}_delta_mc_se'] = mean_se(d)
            virtual_arrays[f'{name}_{metric}_delta'] = d.numpy()
    assert model_hash(actor) == before, 'virtual step changed formal actor'
    assert not paths.requires_grad and not pl.requires_grad and not po.requires_grad
    np.savez_compressed(directory/f'diagnostic-{step}.npz', paths=paths.numpy(), learned=pl.numpy(),
                        oracle=po.numpy(), f=f.numpy(), gains=sv.numpy(), q_learned=ql.numpy(), q_oracle=qo.numpy(),
                        clipped_learned=clipped[0].numpy(), clipped_oracle=clipped[1].numpy(),
                        step_learned=updates[0].numpy(), step_oracle=updates[1].numpy(), **virtual_arrays)
    return row, time_rows, conditional_diagnostics(actor, base, models, c, paths, seed, step)

# %% [markdown]
# ## 3. 可执行实现审计
#
# 以下检查不以方法排名为标准。double precision 的方向有限差分只针对**冻结路径和标签**
# 的 matching loss。不会把 lean AM 强行与穿过全部 rollout 的 BPTT 等同。
# 每个 seed 用真实预训练 base / reward models 再运行一次；任何失败直接中止并保存 traceback。

# %%
def implementation_audit(base, models, c):
    checks = []
    def check(name, condition, value=None):
        checks.append(dict(name=name, passed=bool(condition), value=value))
        if not condition:
            raise AssertionError(name)
    actor = actor_copy(base)
    check('exact full pretrained initialization', torch.equal(flat_parameters(actor), flat_parameters(base)))
    z, xi = noise(c, 12, 71003)
    paths, controls = rollout(actor, base, c, z, xi)
    check('rollout detached and initial control zero', not paths.requires_grad and torch.count_nonzero(controls) == 0)
    f = base_jacobians(base, paths, c)
    zero_p = lean_adjoint(f, terminal_label(models, paths[:, -1], replace(c, beta=0)))
    zero_q = flat_gradient(matching_loss(actor, base, paths, zero_p, c), actor)
    check('no reward guide implies zero AM gradient', torch.count_nonzero(zero_q) == 0)
    replay = rollout(actor, base, c, z, xi)
    check('exact deterministic replay with fixed noise', torch.equal(paths, replay[0]) and torch.equal(controls, replay[1]))
    # Explicit autograd.functional Jacobian vs input VJP and batched transition Jacobian.
    t = float(grid(c)[0][2])
    x = paths[0, 2].clone().requires_grad_(True)
    jac = torch.autograd.functional.jacobian(lambda a: 2*base(a, t)-a/t, x)
    v = x.new_tensor([.4, -.7])
    vjp = torch.autograd.grad((2*base(x, t)-x/t)@v, x)[0]
    check('base VJP equals explicit Jacobian transpose', torch.allclose(vjp, jac.T@v, atol=2e-6))
    check('batched base Jacobian includes minus I/t', torch.allclose(f[0, 2], torch.eye(2)+grid(c)[2]*jac, atol=2e-6))
    endpoint = paths[:, -1]
    zl, zo = terminal_label(models, endpoint, c), terminal_label(models, endpoint, c, True)
    separate = torch.stack([scalar_gradient(m, endpoint) for m in models]).mean(0)
    check('terminal label is scalar ensemble autograd', torch.allclose(zl, -c.beta*separate, atol=2e-6))
    pl, po = lean_adjoint(f, zl), lean_adjoint(f, zo)
    expected = torch.einsum('bkij,bj->bki', propagation_products(f), zl-zo)
    check('discrete terminal error product identity', torch.allclose(pl-po, expected, atol=2e-5, rtol=2e-4))
    check('labels and Jacobians detached', not pl.requires_grad and not f.requires_grad)
    ql, qo, residual = projection_audit(actor, base, paths, pl, po, c)
    check('actor Jacobian projection identity', residual < 1e-4, residual)
    # Weighted velocity form is algebraically identical to matching residual form.
    ts, nu, h = grid(c)
    tt = ts[None, :, None].expand(len(paths), -1, 1)
    vb = base(paths[:, :-1], tt).detach()
    target = vb-.5*nu[None, :, None].square()*pl[:, 1:]
    equivalent = (2*h*(actor(paths[:, :-1], tt)-target).square().sum(-1)/nu.square()).sum(-1).mean()
    check('weighted velocity-target loss equivalence', torch.allclose(equivalent, matching_loss(actor, base, paths, pl, c), atol=2e-5))
    # Double precision central finite difference through local actor only.
    ad, bd = actor_copy(base).double(), copy.deepcopy(base).double()
    pd, xd = pl.double(), paths.double()
    loss = matching_loss(ad, bd, xd, pd, c)
    q = flat_gradient(loss, ad)
    d = torch.randn(q.shape, generator=generator(5017), dtype=torch.float64); d /= d.norm()
    initial = flat_parameters(ad)
    eps = 1e-5
    torch.nn.utils.vector_to_parameters(initial+eps*d, ad.parameters())
    plus = float(matching_loss(ad, bd, xd, pd, c).detach())
    torch.nn.utils.vector_to_parameters(initial-eps*d, ad.parameters())
    minus = float(matching_loss(ad, bd, xd, pd, c).detach())
    finite = (plus-minus)/(2*eps)
    check('frozen-label directional finite difference', abs(finite-float(q@d)) < 1e-7, abs(finite-float(q@d)))
    # Saved paths and p carry no actor graph: actor perturbation doesn't mutate them.
    check('no rollout/adjoint backprop graph', paths.grad_fn is None and pl.grad_fn is None)
    original_hashes = [model_hash(m) for m in [base, *models]]
    with forbid_truth():
        blocked = 0
        for fn, args in [(true_reward, (endpoint, c)), (true_gradient, (endpoint, c))]:
            try:
                fn(*args)
            except AssertionError:
                blocked += 1
        actor_update(actor, adam_for(actor, c), base, models, c, z, xi, oracle=False)
    check('learned update succeeds with both truth APIs blocked', blocked == 2)
    check('base and reward parameters unchanged', original_hashes == [model_hash(m) for m in [base, *models]])
    check('base and reward grads never populated', all(p.grad is None for m in [base, *models] for p in m.parameters()))
    metrics, _ = evaluate(actor, base, c, z, xi)
    check('Gaussian-transition KL equals control energy', metrics['kl_identity_max_abs'] < 2e-5, metrics['kl_identity_max_abs'])
    big_z, big_xi = noise(c, 24, 71003)
    check('2x noise streams have identical prefix', torch.equal(z, big_z[:12]) and torch.equal(xi, big_xi[:12]))
    fine_z, fine_xi = noise(c, 12, 71003, steps=2*c.steps)
    check('K and 2K increments correctly coupled', torch.equal(z, fine_z) and torch.equal(xi, fine_xi.reshape(12, c.steps, 2, 2).sum(2)/math.sqrt(2)))
    return checks


def record_event(out, event, **fields):
    value = dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
    with (out/'events.jsonl').open('a') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def train_actor(base, models, c, protocol, seed, method, directory, out, diagnostic=False,
                gradient_source=None, batch_size=None):
    # The display name is never used to decide which terminal-gradient API runs.
    legacy = {'AM-learned': ('learned', c.actor_batch),
              'AM-oracle': ('oracle', c.actor_batch),
              'AM-learned-2x': ('learned', 2*c.actor_batch)}
    if gradient_source is None or batch_size is None:
        if method not in legacy or gradient_source is not None or batch_size is not None:
            raise ValueError('Specify both gradient_source and batch_size')
        gradient_source, batch_size = legacy[method]
    if gradient_source not in ('learned', 'oracle') or batch_size not in (c.actor_batch, 2*c.actor_batch):
        raise ValueError((gradient_source, batch_size))
    actor = actor_copy(base)
    optimizer = adam_for(actor, c)
    assert torch.equal(flat_parameters(actor), flat_parameters(base))
    hashes = [model_hash(m) for m in [base, *models]]
    curve, update_rows, diag, propagation, conditional = [], [], [], [], []
    start = time.monotonic()
    diagnostic_seconds = 0.
    for step in range(c.actor_updates+1):
        if step % c.curve_every == 0 or step == c.actor_updates:
            z, xi = noise(protocol, c.curve_paths, seed_key(seed, 300000), steps=c.steps)
            row, _ = evaluate(actor, base, c, z, xi)
            row.update(seed=seed, method=method, update=step, train_K=c.steps)
            curve.append(row)
            pd.DataFrame(curve).to_csv(directory/'curve.csv', index=False)
            record_event(out, 'progress', seed=seed, method=method, train_K=c.steps, update=step)
        if diagnostic and step in c.diagnostic_steps:
            diagnostic_start = time.monotonic()
            d, p, cond = diagnostics(actor, optimizer, base, models, c, seed, step, directory)
            diagnostic_seconds += time.monotonic()-diagnostic_start
            diag.append(d); propagation.extend(p); conditional.extend(cond)
        if step in c.diagnostic_steps or step == c.actor_updates:
            torch.save(dict(actor=actor.state_dict(), optimizer=optimizer.state_dict(), update=step,
                            method=method, seed=seed, config=asdict(c), base_hash=hashes[0],
                            elapsed_seconds=time.monotonic()-start,
                            extra_diagnostic_seconds=diagnostic_seconds), directory/f'checkpoint-{step}.pt')
        if step == c.actor_updates:
            break
        n = batch_size
        z, xi = noise(protocol, n, seed_key(seed, 100000+2*step), steps=c.steps)
        if gradient_source == 'oracle':
            row = actor_update(actor, optimizer, base, models, c, z, xi, oracle=True)
        else:
            with forbid_truth():
                row = actor_update(actor, optimizer, base, models, c, z, xi, oracle=False)
        row.update(update=step+1)
        update_rows.append(row)
    assert hashes == [model_hash(m) for m in [base, *models]], 'frozen models changed'
    pd.DataFrame(update_rows).to_csv(directory/'updates.csv', index=False)
    torch.save(actor.state_dict(), directory/'final.pt')
    save_json(directory/'budget.json', dict(seed=seed, method=method, train_K=c.steps,
              gradient_source=gradient_source, batch_size=batch_size,
              updates=c.actor_updates, training_paths=c.actor_updates*batch_size,
              generated_time_steps=c.actor_updates*batch_size*c.steps,
              actor_true_reward_queries=0,
              actor_true_gradient_points=c.actor_updates*batch_size if gradient_source == 'oracle' else 0,
              wall_seconds=time.monotonic()-start, extra_diagnostic_seconds=diagnostic_seconds,
              frozen_hashes_before=hashes,
              frozen_hashes_after=[model_hash(m) for m in [base, *models]], final_checkpoint_rule='fixed final update'))
    return actor, curve, diag, propagation, conditional


def paired_summary(frame, metric, sampler='SDE', eval_grid=None):
    selected = frame[frame.sampler.eq(sampler) & frame.experiment.eq('main')]
    if eval_grid is not None:
        selected = selected[selected.eval_K.eq(eval_grid)]
    pivot = selected.pivot(index='seed', columns='method', values=metric)
    d = pivot['AM-oracle']-pivot['AM-learned']
    mean, sd = float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.
    sem = sd/math.sqrt(len(d))
    half = float(student_t.ppf(.975, len(d)-1)*sem) if len(d) > 1 else None
    return dict(metric=metric, sampler=sampler, seeds=len(d), paired_deltas=d.to_dict(),
                mean_oracle_minus_learned=mean, seed_sd=sd, seed_se=sem,
                ci95_low=mean-half if half is not None else None,
                ci95_high=mean+half if half is not None else None,
                favorable_seeds=int((d > 0).sum() if metric == 'reward' else (d < 0).sum()))


def run_suite(c, out):
    records, curves, fits, quality, diagnostic_rows, time_rows, conditional_rows, audits = [], [], [], [], [], [], [], []
    visual = None
    record_event(out, 'start', config_sha256=hashlib.sha256(json.dumps(asdict(c), sort_keys=True).encode()).hexdigest(),
                 note='No formal configuration changes or win-based selection allowed')
    try:
        for seed in c.seeds:
            directory = out/f'seed-{seed}'; directory.mkdir()
            data = make_data(c, seed)
            np.savez_compressed(directory/'data.npz', **{k: v.numpy() for k, v in data.items()})
            record_event(out, 'pretrain_start', seed=seed)
            base, fh, selection = fit_base(c, data, seed)
            with forbid_truth():
                models, rh, selections = fit_rewards(c, data, seed)
            torch.save(base.state_dict(), directory/'base.pt')
            for j, model in enumerate(models):
                torch.save(model.state_dict(), directory/f'reward-{j}.pt')
            save_json(directory/'pretraining-selection.json', dict(base=selection, rewards=selections,
                      base_hash=model_hash(base), reward_hashes=[model_hash(m) for m in models],
                      data_hashes={k: tensor_hash(v) for k, v in data.items()}, true_scalar_observations=c.reward_train+c.reward_valid))
            fits.extend([dict(seed=seed, **row) for row in fh+rh])
            seed_audits = implementation_audit(base, models, c)
            audits.extend([dict(seed=seed, **row) for row in seed_audits])
            save_json(directory/'implementation-tests.json', seed_audits)
            z, xi = noise(c, c.final_paths, seed_key(seed, 700000))
            base_ode = ode_sample(base, z, c.ode_steps)
            q = dict(seed=seed, **selection, **mode_metrics(base_ode, c),
                     sliced_w2_behavior=sliced_distance(base_ode, data['heldout'], seed_key(seed, 99)))
            quality.append(q)
            actor_map = {'Frozen Flow': base}
            print(f'{seed}: FM selected step {selection["selected_step"]}; validation={selection["validation_loss"]:.4f}; '
                  f'base ODE coverage={q["within_radius_08"]:.3f}', flush=True)
            for method in ('AM-learned', 'AM-oracle', 'AM-learned-2x'):
                target = directory/method; target.mkdir()
                actor, curve, diag, pr, cond = train_actor(base, models, c, c, seed, method, target, out,
                                                          diagnostic=method == 'AM-learned')
                actor_map[method] = actor
                curves.extend(curve); diagnostic_rows.extend(diag); time_rows.extend(pr); conditional_rows.extend(cond)
                record_event(out, 'actor_completed', seed=seed, method=method, train_K=c.steps)
                print(f'{seed}: {method} completed {c.actor_updates} updates', flush=True)
            terminals, paths_for_figure = {}, {}
            for method, actor in actor_map.items():
                for steps in (c.steps, 2*c.steps):
                    z, xi = noise(c, c.final_paths, seed_key(seed, 700000), steps=steps)
                    row, arrays = evaluate(actor, base, replace(c, steps=steps), z, xi)
                    row.update(seed=seed, method=method, experiment='main', sampler='SDE', eval_K=steps,
                               train_K=0 if method == 'Frozen Flow' else c.steps)
                    records.append(row)
                    safe_name = method.replace(' ', '-')
                    np.savez_compressed(directory/f'eval-{safe_name}-K{steps}.npz',
                                        terminal=arrays['paths'][:, -1].numpy(), reward=arrays['reward'].numpy(),
                                        cost=arrays['cost'].numpy(), energy=arrays['energy'].numpy())
                    if steps == c.steps:
                        terminals[method+' SDE'] = arrays['paths'][:, -1]
                        paths_for_figure[method] = arrays['paths'][:64]
                        if method == 'Frozen Flow':
                            for update in range(0, c.actor_updates+1, c.curve_every):
                                curves.append(dict(row, update=update))
                action = ode_sample(actor, z, c.ode_steps)
                reward, se = mean_se(true_reward(action, c))
                records.append(dict(seed=seed, method=method, experiment='main', sampler='ODE', eval_K=c.ode_steps,
                                    train_K=0 if method == 'Frozen Flow' else c.steps,
                                    reward=reward, reward_mc_se=se, **mode_metrics(action, c),
                                    sliced_w2_SDE_ODE=sliced_distance(terminals[method+' SDE'], action, seed_key(seed, 98))))
                terminals[method+' ODE'] = action
                np.savez_compressed(directory/f'eval-{safe_name}-ODE.npz', terminal=action.numpy(), reward=true_reward(action, c).numpy())
            # Only pre-specified first seed gets a complete 2K training comparison.
            if c.fine_retrain_first_seed and seed == c.seeds[0]:
                fine = replace(c, steps=2*c.steps)
                for method in ('AM-learned', 'AM-oracle'):
                    target = directory/(method+'-retrain-2K'); target.mkdir()
                    actor, curve, _, _, _ = train_actor(base, models, fine, c, seed, method, target, out)
                    curves.extend(curve)
                    z, xi = noise(c, c.final_paths, seed_key(seed, 700000), steps=fine.steps)
                    row, arrays = evaluate(actor, base, fine, z, xi)
                    records.append(dict(row, seed=seed, method=method, experiment='retrain-2K', sampler='SDE',
                                        train_K=fine.steps, eval_K=fine.steps))
                    np.savez_compressed(target/'eval.npz', terminal=arrays['paths'][:, -1].numpy(),
                                        reward=arrays['reward'].numpy(), cost=arrays['cost'].numpy(), energy=arrays['energy'].numpy())
                    record_event(out, 'actor_completed', seed=seed, method=method, train_K=fine.steps)
            if seed == c.seeds[0]:
                visual = dict(data=data, base=base, models=models, base_ode=base_ode, terminals=terminals, initial_noise=z[:64],
                              paths=paths_for_figure)
            # Incremental persistence: failed later seeds cannot erase previous results.
            for name, rows in [('final', records), ('learning-curves', curves), ('pretraining', fits),
                               ('base-quality', quality), ('diagnostics', diagnostic_rows),
                               ('propagation', time_rows), ('conditional', conditional_rows)]:
                pd.DataFrame(rows).to_csv(out/f'{name}.csv', index=False)
            save_json(out/'implementation-tests.json', audits)
            record_event(out, 'seed_completed', seed=seed)
        frame = pd.DataFrame(records)
        paired = [paired_summary(frame, metric, sampler, c.steps if sampler == 'SDE' else c.ode_steps)
                  for sampler, metric in [('SDE', 'reward'), ('SDE', 'cost'), ('ODE', 'reward')]]
        save_json(out/'paired-seed-summary.json', paired)
        record_event(out, 'suite_completed', actor_runs=3*len(c.seeds)+(2 if c.fine_retrain_first_seed else 0))
        return dict(final=frame, curves=pd.DataFrame(curves), fits=pd.DataFrame(fits), quality=pd.DataFrame(quality),
                    diagnostic=pd.DataFrame(diagnostic_rows), propagation=pd.DataFrame(time_rows),
                    conditional=pd.DataFrame(conditional_rows), audits=audits, paired=paired, visual=visual)
    except BaseException as exc:
        record_event(out, 'failure', exception=repr(exc), traceback=traceback.format_exc())
        raise


START = time.monotonic()
RESULTS = run_suite(CFG, OUT)
display(RESULTS['quality'])
display(pd.DataFrame(RESULTS['audits']))

# %% [markdown]
# ## 4. 实际结果与可视化
#
# 所有图均为 noise→action generation。模式占比采用最近预定中心分配，另报半径0.8内比例。
# 学习曲线阴影为独立训练 seed 的标准差；不是把数千评估动作当作独立实验。
# 表中的 `*_mc_se` 是同一固定 actor 内的积分误差，不能替代 seed 级不确定性。
# Conditional MC 只固定中间状态并续采未来，报告均值向量各分量 SE、方差迹和逐路径RMS；
# ensemble disagreement 单列，未校准为真实梯度误差。

# %%
def make_figures(result, c, out):
    saved = []
    def finish(fig, name):
        fig.suptitle('noise→action generation — '+name.replace('_', ' '), fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(out/(name+'.png'), dpi=150, bbox_inches='tight')
        saved.append(name+'.png')
        display(fig)
        plt.close(fig)
    visual = result['visual']
    means = np.asarray(c.means)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for axis, data, title in zip(ax, (visual['data']['heldout'], visual['base_ode'], visual['terminals']['Frozen Flow SDE']),
                                 ('held-out behavior', 'pretrained base ODE', 'pretrained base SDE')):
        axis.scatter(*data[:2000].T, s=3, alpha=.25)
        axis.scatter(*means.T, marker='x', c='black')
        axis.set(title=title, xlabel='action coordinate 1', ylabel='action coordinate 2', xlim=(-3, 3), ylim=(-3, 3), aspect='equal')
    finish(fig, '01_behavior_and_pretrained_flow')
    line = torch.linspace(-3, 3, 100)
    yy, xx = torch.meshgrid(line, line, indexing='ij')
    mesh = torch.stack((xx.flatten(), yy.flatten()), 1)
    with torch.no_grad():
        truth, learned = true_reward(mesh, c), reward_mean(visual['models'], mesh)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for axis, values, title in zip(ax, (truth, learned, learned-truth), ('true reward', 'frozen scalar ensemble', 'value error (learned − true)')):
        im = axis.imshow(values.reshape(100, 100), extent=(-3, 3, -3, 3), origin='lower', cmap='coolwarm')
        axis.set(title=title, xlabel='action coordinate 1', ylabel='action coordinate 2')
        fig.colorbar(im, ax=axis, shrink=.8)
    finish(fig, '02_reward_and_value_error')
    paths = visual['paths']['AM-learned']
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.3))
    for axis, data, when in zip(ax[:3], (visual['initial_noise'], paths[:, c.steps//2], paths[:, -1]),
                                (0., (1+c.epsilon)/2, 1.)):
        axis.scatter(*data.T, s=12)
        axis.set(title=f't={when:.3f}', xlabel='generation coordinate 1', ylabel='generation coordinate 2')
    for path in paths[:12]:
        ax[3].plot(*path.T, alpha=.6, lw=.8)
    ax[3].set(title='stochastic generation paths (not motion)', xlabel='generation coordinate 1', ylabel='generation coordinate 2')
    finish(fig, '03_initial_middle_terminal_states')
    curve = result['curves'][result['curves'].train_K.isin((0, c.steps))]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for method, rows in curve.groupby('method'):
        groups = rows.groupby('update').reward
        mean, sd = groups.mean(), groups.std().fillna(0.)
        ax[0].plot(mean.index, mean, label=method)
        ax[0].fill_between(mean.index, mean-sd, mean+sd, alpha=.15)
    ax[0].set(xlabel='actor updates', ylabel='true SDE reward', title='mean ± training-seed SD')
    for (seed, stage, member), rows in result['fits'].groupby(['seed', 'stage', 'member']):
        if seed == c.seeds[0]:
            ax[1].plot(rows.step, rows.valid_mse, label=f'{stage} {member}')
    ax[1].set(xlabel='pretraining updates', ylabel='held-out MSE', yscale='log', title='fixed validation selection')
    for axis in ax:
        axis.legend(fontsize=8)
    finish(fig, '04_reward_learning_and_pretraining')
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for method, rows in curve.groupby('method'):
        for axis, metric in zip(ax, ('cost', 'energy')):
            group = rows.groupby('update')[metric]
            mean, sd = group.mean(), group.std().fillna(0)
            axis.plot(mean.index, mean, label=method)
            axis.fill_between(mean.index, mean-sd, mean+sd, alpha=.15)
            axis.set(xlabel='actor updates', ylabel=metric)
    ax[0].set_title('true discrete regularized J (lower better)')
    ax[1].set_title('MC estimate of exact finite-chain path KL')
    ax[0].legend(fontsize=8)
    finish(fig, '05_regularized_objective_and_path_KL')
    pr = result['propagation']
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for checkpoint, rows in pr.groupby('checkpoint'):
        average = rows.groupby('time').mean(numeric_only=True)
        for axis, metric in zip(ax, ('adjoint_error_rms', 'label_error_rms', 'gain_max_mean')):
            axis.plot(average.index, average[metric], label=f'update {checkpoint}')
            axis.set(xlabel='generation time', ylabel=metric)
    ax[2].axhline(1, color='black', ls=':', lw=1)
    for axis in ax:
        axis.legend(fontsize=8)
    finish(fig, '06_adjoint_error_and_propagation_gain')
    diag = result['diagnostic']
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for seed, rows in diag.groupby('seed'):
        ax[0].plot(rows.checkpoint, rows.q_cosine, '-o', label=f'{seed} q')
        ax[0].plot(rows.checkpoint, rows.step_cosine, '--x', label=f'{seed} Adam')
        ax[1].plot(rows.checkpoint, rows.q_norm_ratio, '-o', label=f'{seed} q')
        ax[1].plot(rows.checkpoint, rows.step_norm_ratio, '--x', label=f'{seed} Adam')
        ax[2].plot(rows.checkpoint, rows.virtual_oracle_cost_delta-rows.virtual_learned_cost_delta, '-o', label=str(seed))
    ax[0].set(ylabel='learned / oracle cosine')
    ax[1].set(ylabel='learned / oracle norm ratio')
    ax[2].set(ylabel='virtual ΔJ oracle − learned (lower better)')
    for axis in ax:
        axis.set_xlabel('common AM-learned checkpoint')
        axis.legend(fontsize=7)
    finish(fig, '07_parameter_gradient_and_actual_Adam_step')
    final = result['final']
    main = final[final.experiment.eq('main') & final.sampler.eq('SDE') & final.eval_K.eq(c.steps)]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for seed, rows in main.groupby('seed'):
        ax[0].plot(rows.method, rows.reward, '-o', label=str(seed))
        ax[1].plot(rows.method, rows.cost, '-o', label=str(seed))
    ax[0].set_ylabel('final true reward (higher better)')
    ax[1].set_ylabel('final true J (lower better)')
    ode = final[final.experiment.eq('main') & final.sampler.eq('ODE')]
    methods = main.method.unique()
    positions = np.arange(len(methods))
    ax[2].bar(positions-.17, [main[main.method.eq(m)].reward.mean() for m in methods], .34, label='SDE')
    ax[2].bar(positions+.17, [ode[ode.method.eq(m)].reward.mean() for m in methods], .34, label='ODE')
    ax[2].set_xticks(positions, methods)
    ax[2].set_ylabel('same actor, different sampler: true reward')
    for axis in ax:
        axis.tick_params(axis='x', rotation=22)
        axis.legend(fontsize=8)
    finish(fig, '08_final_seed_results_and_sampler_gap')
    fig, ax = plt.subplots(2, 4, figsize=(14, 6.6))
    for column, method in enumerate(methods):
        for row, sampler in enumerate(('SDE', 'ODE')):
            action = visual['terminals'][method+' '+sampler]
            ax[row, column].scatter(*action[:1500].T, s=2, alpha=.2)
            ax[row, column].set(title=method+' '+sampler, xlim=(-3, 3), ylim=(-3, 3), aspect='equal')
    finish(fig, '09_SDE_and_ODE_distributions')
    cond = result['conditional']
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for checkpoint, rows in cond.groupby('checkpoint'):
        a = rows.groupby('time').mean(numeric_only=True)
        ax[0].plot(a.index, a.pathwise_rms, '-o', label=f'{checkpoint} path RMS')
        ax[0].plot(a.index, a.conditional_mean_norm, '--x', label=f'{checkpoint} conditional mean norm')
        ax[1].plot(a.index, a.variance_trace, '-o', label=f'update {checkpoint}')
    ax[0].set(xlabel='fixed generation time', ylabel='matching residual-label error')
    ax[1].set(xlabel='fixed generation time', ylabel='conditional variance trace')
    for axis in ax:
        axis.legend(fontsize=8)
    finish(fig, '10_pathwise_vs_conditional_label_error')
    return saved


FIGURES = make_figures(RESULTS, CFG, OUT)
main = RESULTS['final'].query('experiment == "main" and sampler == "SDE" and eval_K == @CFG.steps')
display(main[['seed', 'method', 'reward', 'reward_mc_se', 'cost', 'cost_mc_se', 'energy', 'energy_mc_se']])
display(main.groupby('method')[['reward', 'cost', 'energy']].agg(['mean', 'std']))
display(RESULTS['final'][['seed', 'method', 'experiment', 'sampler', 'train_K', 'eval_K',
                           'reward', 'cost', 'mode_0', 'mode_1', 'mode_2', 'mode_3']])
display(RESULTS['diagnostic'])
display(RESULTS['conditional'])
display(pd.DataFrame(RESULTS['paired']))

# %% [markdown]
# ## 5. 由结果自动生成结论，保留不确定性
#
# Oracle 与 learned 的三 seed 差值才是统计单位。Student-t 区间只是 n=3 的初步摘要。
# 无论结果如何，此实验不能证明传播感知抑制优于一般梯度处理、不能建立普适性、
# 不能把 lean 梯度叫真实策略梯度，也没有测量 ODE action KL。
# 条件平均、参数投影、Adam history 都可能改变对逐路径误差的判断。
# 下方报告使用实际结果，不以候选获胜为验收条件。

# %%
def make_report(result, c, elapsed):
    lines = [f'# Flow AM execution report ({c.profile})', '',
             f'Seeds: {list(c.seeds)}. {3*len(c.seeds)} main fine-tunes + '
             f'{2 if c.fine_retrain_first_seed else 0} pre-specified first-seed 2K retrains; '
             f'{len(c.seeds)} frozen bases. Fixed final step {c.actor_updates}. Runtime {elapsed:.1f}s.', '',
             'Primary sampler: shared frozen-base RK2 warmup to epsilon, then all-Gaussian EM chain. '
             'This is a truncated-start, finite-step approximation, not exact time-zero memoryless AM. '
             'The complete implemented-chain KL equals expected conditional Gaussian KL; numerical estimates have MC error. '
             'The separate ODE integrates fine velocity from 0 to 1 and has no reported KL.', '',
             'Oracle minus learned paired differences (positive reward is good; negative cost is good):']
    for row in result['paired']:
        lines.extend(['', f'- {row["sampler"]} {row["metric"]}: mean {row["mean_oracle_minus_learned"]:.6f}, '
                      f'seed SD {row["seed_sd"]:.6f}; 95% t interval '
                      f'[{row["ci95_low"]}, {row["ci95_high"]}], favorable seeds {row["favorable_seeds"]}/{row["seeds"]}.'])
    primary = result['paired'][1]
    if primary['ci95_high'] is not None and primary['ci95_high'] < 0:
        interpretation = '该设置观察到可恢复的终端梯度估计损失（按真实离散 J）；尚未证明传播感知处理优于普通梯度处理。'
    elif primary['ci95_low'] is not None and primary['ci95_low'] > 0:
        interpretation = 'Oracle 的真实离散 J 更差；有限预算、优化尺度、离散化和梯度噪声可能交互，不能自动判 learned 实现有错。'
    else:
        interpretation = '三 seed 的真实离散 J 差异区间包含零，尚未确认稳定收益空间；这既不是普适证明，也不否定理论动机。'
    lines.extend(['', interpretation, '',
                  'Reward, regularized J and ODE/SDE outcomes must be interpreted separately. '
                  'Ensemble disagreement is not a calibrated error estimate; all mechanism diagnostics use common learned-actor checkpoints. '
                  'Virtual Adam steps clone the current optimizer state and use fresh common evaluation noise without changing formal training.', '',
                  'No new candidate, covariance regularizer, damping, guard, adaptive beta or oracle-dependent checkpoint selection was introduced. '
                  'Learned branches made zero new true reward/gradient queries during updates. '
                  'The first-seed K/2K retrain is not a three-seed convergence study. '
                  'Diagnostics query truth only outside the learned-training guard.', '',
                  f'Implementation checks: {sum(x["passed"] for x in result["audits"])}/{len(result["audits"])}. '
                  'Failed executions, if any, remain in their separate events.jsonl; no trajectory clipping or nan_to_num.'])
    return '\n'.join(lines)+'\n'


REPORT = make_report(RESULTS, CFG, time.monotonic()-START)
(OUT/'report.md').write_text(REPORT)
save_json(OUT/'completion.json', dict(status='complete', config_sha256=CONFIG_SHA, profile=CFG.profile,
          seeds=list(CFG.seeds), figures=FIGURES, checks=len(RESULTS['audits']),
          elapsed_seconds=time.monotonic()-START, source_sha256=hashlib.sha256(
              (ROOT/'notebooks/02_flow_am_terminal_gradient.py').read_bytes()).hexdigest()))
display(Markdown(REPORT))
print('All outputs:', OUT.relative_to(ROOT))

# %% [markdown]
# ## 5. 为什么 Oracle AM 收益有限：预定的收益空间诊断
#
# 这是既有三个开发 seed 的追加诊断，不是独立确认实验。固定同一真实目标、
# 预训练模型、Lean adjoint、matching 权重、Adam 与 K=32 主协议。
# 四臂为 learned/oracle × B64/B128；B128 的前64条训练噪声与 B64 对齐，
# 但不同 actor 的状态轨迹在训练分化后并不相同。
# 比较固定300次更新，也比较同为19200条训练轨迹的 B64/300 与 B128/150；
# 后者更新次数不同。重复梯度诊断只在共同 checkpoint 上使用相同状态路径，
# 因而不能把其 oracle Matching 梯度称为真实目标的参数梯度。

# %%
@dataclass(frozen=True)
class OracleDiagnosticConfig:
    repeats: int = 32
    virtual_paths: int = 1024
    train_noise_offset: int = 1000000
    checkpoint_methods: tuple = (('AM-learned', 0), ('AM-learned', 150),
                                 ('AM-learned', 300), ('AM-oracle', 300))
    batches: tuple = (64, 128)
    previous_run: str = 'runs/02-flow-am-formal-drdkjy4g'


def oracle_diagnostic_configuration(c):
    d = OracleDiagnosticConfig()
    if c.profile == 'smoke':
        d = replace(d, repeats=2, virtual_paths=64,
                    checkpoint_methods=(('AM-learned', 0), ('AM-learned', c.actor_updates),
                                        ('AM-oracle', c.actor_updates)))
    if tuple(d.batches) != (c.actor_batch, 2*c.actor_batch):
        raise AssertionError('batch protocol changed')
    return d


def load_velocity(path, c, frozen=False):
    model = Velocity(c.width)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return freeze(model) if frozen else model


def load_reward_models(directory, c):
    models = []
    for j in range(c.members):
        model = RewardModel(c.width)
        model.load_state_dict(torch.load(directory/f'reward-{j}.pt', map_location='cpu', weights_only=True))
        models.append(freeze(model))
    return models


def load_actor_checkpoint(path, c, base=None):
    item = torch.load(path, map_location='cpu', weights_only=False)
    actor = actor_copy(base) if base is not None else Velocity(c.width)
    actor.load_state_dict(item['actor'])
    optimizer = adam_for(actor, c)
    optimizer.load_state_dict(item['optimizer'])
    return actor, optimizer, item


def exact_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact_tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def paired_difference(oracle_values, learned_values):
    # The SE belongs to the per-path *difference*, not a sum of two independent SEs.
    difference = (oracle_values-learned_values).detach().double()
    return mean_se(difference)


def trace_covariance(x, y=None):
    x = x.double()
    y = x if y is None else y.double()
    if x.shape != y.shape or len(x) < 2:
        raise ValueError('Need aligned repeated gradients')
    xc, yc = x-x.mean(0), y-y.mean(0)
    return float((xc*yc).sum()/(len(x)-1))


def gradient_pair(actor, base, models, c, z, xi):
    paths, _ = rollout(actor, base, c, z, xi)
    f = base_jacobians(base, paths, c)
    endpoint = paths[:, -1]
    p_learned = lean_adjoint(f, terminal_label(models, endpoint, c, oracle=False))
    p_oracle = lean_adjoint(f, terminal_label(models, endpoint, c, oracle=True))
    q_learned = flat_gradient(matching_loss(actor, base, paths, p_learned, c), actor)
    q_oracle = flat_gradient(matching_loss(actor, base, paths, p_oracle, c), actor)
    return q_learned, q_oracle


def virtual_adam_pair(actor, optimizer, base, c, q_learned, q_oracle, z_eval, xi_eval):
    params_before = flat_parameters(actor)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    rng_before = torch.get_rng_state().clone()
    clones, clipped, steps, evaluations = [], [], [], []
    for q in (q_learned, q_oracle):
        candidate = copy.deepcopy(actor)
        candidate_optimizer = adam_for(candidate, c)
        candidate_optimizer.load_state_dict(copy.deepcopy(optimizer_before))
        before = flat_parameters(candidate)
        clipped.append(apply_gradient(candidate, candidate_optimizer, q, c))
        steps.append(flat_parameters(candidate)-before)
        evaluations.append(evaluate(candidate, base, c, z_eval, xi_eval))
        clones.append(candidate)
    if not torch.equal(params_before, flat_parameters(actor)):
        raise AssertionError('Virtual step changed formal actor')
    if not exact_tree_equal(optimizer_before, optimizer.state_dict()):
        raise AssertionError('Virtual step changed formal Adam history')
    if not torch.equal(rng_before, torch.get_rng_state()):
        raise AssertionError('Virtual step changed global training RNG')
    _, learned_arrays = evaluations[0]
    _, oracle_arrays = evaluations[1]
    dj, dj_se = paired_difference(oracle_arrays['cost'], learned_arrays['cost'])
    dr, dr_se = paired_difference(oracle_arrays['reward'], learned_arrays['reward'])
    return dict(raw_cosine=cosine(q_learned, q_oracle),
                raw_norm_learned=float(q_learned.norm()), raw_norm_oracle=float(q_oracle.norm()),
                clipped_cosine=cosine(*clipped),
                clipped_norm_learned=float(clipped[0].norm()), clipped_norm_oracle=float(clipped[1].norm()),
                step_cosine=cosine(*steps),
                step_norm_learned=float(steps[0].norm()), step_norm_oracle=float(steps[1].norm()),
                step_norm_ratio=float(steps[0].norm()/steps[1].norm()) if steps[1].norm() > 1e-12 else None,
                step_relative_difference=float((steps[0]-steps[1]).norm()/steps[1].norm())
                if steps[1].norm() > 1e-12 else None,
                D_J=dj, D_J_mc_se=dj_se, D_R=dr, D_R_mc_se=dr_se), (
                    (oracle_arrays['cost']-learned_arrays['cost']).numpy(),
                    (oracle_arrays['reward']-learned_arrays['reward']).numpy())


def repeated_gradient_probe(actor, optimizer, base, models, c, seed, checkpoint_index,
                            checkpoint_name, checkpoint_step, d, directory):
    rows, summaries = [], []
    for batch in d.batches:
        probe_start = time.monotonic()
        ql_all, qo_all, dj_paths, dr_paths = [], [], [], []
        for repetition in range(d.repeats):
            # The same role is used at B64 and B128 to ensure a literal prefix.
            role = d.train_noise_offset+checkpoint_index*100000+4*repetition
            z, xi = noise(c, batch, seed_key(seed, role))
            ql, qo = gradient_pair(actor, base, models, c, z, xi)
            z_eval, xi_eval = noise(c, d.virtual_paths, seed_key(seed, role+2))
            virtual, paired_paths = virtual_adam_pair(
                actor, optimizer, base, c, ql, qo, z_eval, xi_eval)
            difference = ql-qo
            oracle_norm = float(qo.norm())
            if oracle_norm > 1e-12:
                coefficient = float(ql.double()@qo.double()/(qo.double()@qo.double()))
                perpendicular = ql-coefficient*qo
                perp_ratio = float(perpendicular.norm()/qo.norm())
            else:
                coefficient, perp_ratio = None, None
            row = dict(seed=seed, checkpoint_actor=checkpoint_name, checkpoint_step=checkpoint_step,
                       batch_size=batch, repetition=repetition, train_noise_seed=seed_key(seed, role),
                       eval_noise_seed=seed_key(seed, role+2), d_norm=float(difference.norm()),
                       q_norm_ratio=float(ql.norm()/qo.norm()) if oracle_norm > 1e-12 else None,
                       projection_c=coefficient, perpendicular_ratio=perp_ratio,
                       projection_undefined=coefficient is None, projection_negative=coefficient is not None and coefficient < 0,
                       **virtual)
            rows.append(row)
            ql_all.append(ql.numpy()); qo_all.append(qo.numpy())
            dj_paths.append(paired_paths[0]); dr_paths.append(paired_paths[1])
        ql_matrix, qo_matrix = torch.from_numpy(np.stack(ql_all)), torch.from_numpy(np.stack(qo_all))
        d_matrix = ql_matrix-qo_matrix
        var_l, var_o = trace_covariance(ql_matrix), trace_covariance(qo_matrix)
        var_d, cross = trace_covariance(d_matrix), trace_covariance(qo_matrix, d_matrix)
        residual = var_l-(var_o+var_d+2*cross)
        if abs(residual) > 1e-8*max(1., abs(var_l)):
            raise AssertionError('Missing qO,d cross covariance')
        subset = pd.DataFrame(rows[-d.repeats:])
        summaries.append(dict(seed=seed, checkpoint_actor=checkpoint_name, checkpoint_step=checkpoint_step,
                              batch_size=batch, repeats=d.repeats,
                              probe_wall_seconds=time.monotonic()-probe_start,
                              mean_d_norm=float(d_matrix.double().mean(0).norm()),
                              d_norm_repeat_mean=float(subset.d_norm.mean()),
                              d_norm_repeat_sd=float(subset.d_norm.std(ddof=1)),
                              trace_cov_qL=var_l, trace_cov_qO=var_o, trace_cov_d=var_d,
                              trace_cross_cov_qO_d=cross, variance_identity_residual=residual,
                              raw_cosine_mean=float(subset.raw_cosine.mean()),
                              q_norm_ratio_mean=float(subset.q_norm_ratio.mean()),
                              projection_c_mean=float(subset.projection_c.mean()),
                              perpendicular_ratio_mean=float(subset.perpendicular_ratio.mean()),
                              clipped_cosine_mean=float(subset.clipped_cosine.mean()),
                              step_cosine_mean=float(subset.step_cosine.mean()),
                              step_norm_ratio_mean=float(subset.step_norm_ratio.mean()),
                              step_relative_difference_mean=float(subset.step_relative_difference.mean()),
                              D_J_mean=float(subset.D_J.mean()), D_J_repeat_sd=float(subset.D_J.std(ddof=1)),
                              D_J_eval_mc_se_rms=float(np.sqrt(np.mean(subset.D_J_mc_se.to_numpy()**2))),
                              D_R_mean=float(subset.D_R.mean()), D_R_repeat_sd=float(subset.D_R.std(ddof=1)),
                              D_R_eval_mc_se_rms=float(np.sqrt(np.mean(subset.D_R_mc_se.to_numpy()**2))),
                              projection_undefined_count=int(subset.projection_undefined.sum()),
                              projection_negative_count=int(subset.projection_negative.sum())))
        np.savez_compressed(directory/f'probe-{checkpoint_name}-{checkpoint_step}-B{batch}.npz',
                            q_learned=np.stack(ql_all), q_oracle=np.stack(qo_all),
                            D_J_path=np.stack(dj_paths), D_R_path=np.stack(dr_paths))
    return rows, summaries


def seed_effects(checkpoint_frame, metric, schedule, final_update=300):
    if schedule == 'equal_updates':
        chosen = checkpoint_frame[checkpoint_frame['update'].eq(final_update)]
    elif schedule == 'equal_rollouts':
        chosen = checkpoint_frame[((checkpoint_frame.batch_size.eq(64) & checkpoint_frame['update'].eq(final_update)) |
                                   (checkpoint_frame.batch_size.eq(128) & checkpoint_frame['update'].eq(final_update//2)))]
    else:
        raise ValueError(schedule)
    pivot = chosen.pivot(index='seed', columns='method', values=metric)
    labels = ('AM-learned-B64', 'AM-oracle-B64', 'AM-learned-B128', 'AM-oracle-B128')
    if set(pivot.columns) != set(labels):
        raise AssertionError('Incomplete 2x2')
    l64, o64, l128, o128 = (pivot[name] for name in labels)
    if metric == 'reward':
        effects = dict(A64=o64-l64, A128=o128-l128, S_learned=l128-l64, S_oracle=o128-o64)
        effects['I'] = effects['A64']-effects['A128']
        identity = effects['S_learned']-effects['S_oracle']
    else:
        effects = dict(A64=l64-o64, A128=l128-o128, S_learned=l64-l128, S_oracle=o64-o128)
        effects['I'] = effects['A64']-effects['A128']
        identity = effects['S_learned']-effects['S_oracle']
    if not np.allclose(effects['I'], identity, atol=1e-9, rtol=1e-9):
        raise AssertionError('2x2 interaction sign error')
    detail, summary = [], []
    for effect, values in effects.items():
        for seed, value in values.items():
            detail.append(dict(schedule=schedule, metric=metric, seed=int(seed),
                               effect=effect, value=float(value)))
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        half = float(student_t.ppf(.975, len(values)-1)*sd/math.sqrt(len(values)))
        summary.append(dict(schedule=schedule, metric=metric, effect=effect, seeds=len(values),
                            mean=mean, seed_sd=sd, descriptive_ci95_low=mean-half,
                            descriptive_ci95_high=mean+half, positive_seeds=int((values > 0).sum())))
    return detail, summary


def replay_prior_baselines(c, out, d):
    if c.profile != 'formal':
        return [dict(status='smoke; historical comparison intentionally skipped')]
    previous = ROOT/d.previous_run
    if not previous.exists():
        raise FileNotFoundError(previous)
    if json.loads((previous/'config.json').read_text()) != json.loads(json.dumps(asdict(c))):
        raise AssertionError('Historical config differs')
    audit = []
    for seed in c.seeds:
        now, old = out/f'seed-{seed}', previous/f'seed-{seed}'
        new_data, old_data = np.load(now/'data.npz'), np.load(old/'data.npz')
        if set(new_data.files) != set(old_data.files) or not all(
                np.array_equal(new_data[k], old_data[k]) for k in new_data.files):
            raise AssertionError('Historical data do not replay')
        for name in ['base.pt', *[f'reward-{j}.pt' for j in range(c.members)]]:
            current, reference = torch.load(now/name, weights_only=True), torch.load(old/name, weights_only=True)
            if not exact_tree_equal(current, reference):
                raise AssertionError(f'Historical pretrain mismatch: {seed} {name}')
        for method in ('AM-learned', 'AM-oracle', 'AM-learned-2x'):
            for step in c.diagnostic_steps:
                filename = f'{method}/checkpoint-{step}.pt'
                current = torch.load(now/filename, weights_only=False)
                reference = torch.load(old/filename, weights_only=False)
                if not exact_tree_equal(current['actor'], reference['actor']) or not exact_tree_equal(
                        current['optimizer'], reference['optimizer']):
                    raise AssertionError(f'Historical training mismatch: {seed} {filename}')
            for grid_steps in (c.steps, 2*c.steps):
                filename = f'eval-{method}-K{grid_steps}.npz'
                current, reference = np.load(now/filename), np.load(old/filename)
                if not all(np.array_equal(current[k], reference[k]) for k in ('terminal', 'reward', 'cost', 'energy')):
                    raise AssertionError(f'Historical evaluation mismatch: {seed} {filename}')
            audit.append(dict(seed=seed, method=method, status='retrained and exact-replay-verified',
                              previous_run=d.previous_run, config_sha256=CONFIG_SHA,
                              base_hash=json.loads((now/'pretraining-selection.json').read_text())['base_hash'],
                              final_actor_hash=hashlib.sha256(
                                  b''.join(v.numpy().tobytes() for v in
                                           torch.load(now/method/'checkpoint-300.pt', weights_only=False)['actor'].values())).hexdigest()))
    save_json(out/'oracle-diagnostic'/'historical-replay-audit.json', audit)
    return audit


def oracle_method_location(seed_directory, diagnostic_seed_directory, method):
    names = dict(AM_learned_B64='AM-learned', AM_oracle_B64='AM-oracle',
                 AM_learned_B128='AM-learned-2x')
    key = method.replace('-', '_')
    return seed_directory/names[key] if key in names else diagnostic_seed_directory/'AM-oracle-B128'


def run_oracle_diagnostic(c, out, d):
    diagnostic_root = out/'oracle-diagnostic'
    diagnostic_root.mkdir()
    formal_spec_path = ROOT/'configs/oracle_am_diagnostic.json'
    formal_spec = json.loads(formal_spec_path.read_text())
    if c.profile == 'formal':
        expected = dict(seeds=list(c.seeds), main_train_K=c.steps, sensitivity_train_K=2*c.steps,
                        epsilon=c.epsilon, beta=c.beta, actor_updates=c.actor_updates,
                        batch_sizes=list(d.batches), probe_repeats=d.repeats,
                        virtual_pair_eval_paths=d.virtual_paths, final_eval_paths=c.final_paths,
                        train_noise_role_offset=d.train_noise_offset)
        if any(formal_spec[key] != value for key, value in expected.items()):
            raise AssertionError('Formal oracle diagnostic config differs from fixed spec')
    save_json(diagnostic_root/'config.json', dict(flow=asdict(c), diagnostic=asdict(d),
              flow_config_sha256=CONFIG_SHA,
              formal_spec_sha256=hashlib.sha256(formal_spec_path.read_bytes()).hexdigest(),
              source_sha256=hashlib.sha256(
                  (ROOT/'notebooks/02_flow_am_terminal_gradient.py').read_bytes()).hexdigest(),
              previous_run=d.previous_run, status='existing development seeds; not confirmation'))
    replay = replay_prior_baselines(c, out, d)
    checkpoint_rows, budget_rows, probe_rows, probe_summaries, grid_rows, frozen_rows = [], [], [], [], [], []
    frozen_hash_rows = []
    record_event(out, 'oracle_diagnostic_start', repeats=d.repeats, virtual_paths=d.virtual_paths)
    for seed in c.seeds:
        source = out/f'seed-{seed}'
        target = diagnostic_root/f'seed-{seed}'
        target.mkdir()
        base = load_velocity(source/'base.pt', c, frozen=True)
        models = load_reward_models(source, c)
        frozen_before = [model_hash(m) for m in [base, *models]]
        pretrain = json.loads((source/'pretraining-selection.json').read_text())
        if frozen_before != [pretrain['base_hash'], *pretrain['reward_hashes']]:
            raise AssertionError('Frozen pretraining hashes differ')
        initial_hash = model_hash(actor_copy(base))
        if initial_hash != model_hash(base):
            raise AssertionError('Actor initialization is not the base')
        initial_optimizer_state = adam_for(actor_copy(base), c).state_dict()
        new_oracle_dir = target/'AM-oracle-B128'
        new_oracle_dir.mkdir()
        train_actor(base, models, c, c, seed, 'AM-oracle-B128', new_oracle_dir, out,
                    gradient_source='oracle', batch_size=128)
        record_event(out, 'oracle_diagnostic_actor_completed', seed=seed, method='AM-oracle-B128', train_K=32)
        print(f'{seed}: AM-oracle-B128 completed', flush=True)
        if seed != c.seeds[0] or not c.fine_retrain_first_seed:
            fine = replace(c, steps=2*c.steps)
            for source_name in ('learned', 'oracle'):
                name = 'AM-'+source_name+'-B64-K64'
                location = target/name
                location.mkdir()
                train_actor(base, models, fine, c, seed, name, location, out,
                            gradient_source=source_name, batch_size=64)
                record_event(out, 'oracle_diagnostic_actor_completed', seed=seed, method=name, train_K=64)
        z_frozen, xi_frozen = noise(c, c.final_paths, seed_key(seed, 700000))
        frozen_metric, frozen_arrays = evaluate(base, base, c, z_frozen, xi_frozen)
        frozen_rows.append(dict(seed=seed, method='Frozen Flow', train_K=0, eval_K=c.steps,
                                **frozen_metric))
        np.savez_compressed(target/'eval-Frozen-Flow-K32.npz',
                            terminal=frozen_arrays['paths'][:, -1].numpy(),
                            reward=frozen_arrays['reward'].numpy(),
                            energy=frozen_arrays['energy'].numpy(),
                            cost=frozen_arrays['cost'].numpy())
        for method in ('AM-learned-B64', 'AM-oracle-B64', 'AM-learned-B128', 'AM-oracle-B128'):
            location = oracle_method_location(source, target, method)
            for step in (0, c.actor_updates//2, c.actor_updates):
                actor, _, checkpoint = load_actor_checkpoint(location/f'checkpoint-{step}.pt', c, base)
                if step == 0 and model_hash(actor) != initial_hash:
                    raise AssertionError(f'Nonidentical initial actor: {seed} {method}')
                if step == 0 and not exact_tree_equal(checkpoint['optimizer'], initial_optimizer_state):
                    raise AssertionError(f'Nonidentical initial Adam state: {seed} {method}')
                z, xi = noise(c, c.final_paths, seed_key(seed, 700000))
                row, arrays = evaluate(actor, base, c, z, xi)
                row.update(seed=seed, method=method, gradient_source='oracle' if '-oracle-' in method else 'learned',
                           batch_size=128 if method.endswith('B128') else 64, update=step, train_K=c.steps,
                           eval_K=c.steps, sampler='SDE')
                row['terminal_shift_from_frozen'], row['terminal_shift_from_frozen_mc_se'] = mean_se(
                    (arrays['paths'][:, -1]-frozen_arrays['paths'][:, -1]).norm(dim=1))
                checkpoint_rows.append(row)
                np.savez_compressed(target/f'eval-{method}-step{step}-K32.npz',
                                    terminal=arrays['paths'][:, -1].numpy(),
                                    reward=arrays['reward'].numpy(), energy=arrays['energy'].numpy(),
                                    cost=arrays['cost'].numpy())
                budget_rows.append(dict(seed=seed, method=method, train_K=c.steps,
                                        update=step, batch_size=row['batch_size'],
                                        training_paths=step*row['batch_size'],
                                        generated_time_steps=step*row['batch_size']*c.steps,
                                        actor_updates=step,
                                        actor_true_gradient_points=step*row['batch_size']
                                        if row['gradient_source'] == 'oracle' else 0,
                                        checkpoint_elapsed_seconds=checkpoint['elapsed_seconds'],
                                        extra_diagnostic_seconds=checkpoint['extra_diagnostic_seconds'],
                                        elapsed_minus_extra_diagnostic_seconds=
                                        checkpoint['elapsed_seconds']-checkpoint['extra_diagnostic_seconds']))
                if step == c.actor_updates:
                    for eval_steps in (c.steps, 2*c.steps):
                        z_grid, xi_grid = noise(c, c.final_paths, seed_key(seed, 700000), steps=eval_steps)
                        grid_metric, grid_arrays = evaluate(actor, base, replace(c, steps=eval_steps), z_grid, xi_grid)
                        grid_rows.append(dict(seed=seed, method=method, train_K=c.steps,
                                              eval_K=eval_steps, **grid_metric))
                        np.savez_compressed(target/f'grid-{method}-train32-eval{eval_steps}.npz',
                                            reward=grid_arrays['reward'].numpy(),
                                            energy=grid_arrays['energy'].numpy(),
                                            cost=grid_arrays['cost'].numpy())
        # Both K64 arms have the same evaluation noise; first seed was trained
        # in the baseline suite, the other two are genuinely new here.
        fine = replace(c, steps=2*c.steps)
        for source_name in ('learned', 'oracle'):
            location = ((source/f'AM-{source_name}-retrain-2K')
                        if seed == c.seeds[0] and c.fine_retrain_first_seed
                        else (target/f'AM-{source_name}-B64-K64'))
            actor, _, checkpoint = load_actor_checkpoint(location/f'checkpoint-{c.actor_updates}.pt', fine, base)
            z, xi = noise(c, c.final_paths, seed_key(seed, 700000), steps=fine.steps)
            metric, arrays = evaluate(actor, base, fine, z, xi)
            grid_rows.append(dict(seed=seed, method=f'AM-{source_name}-B64', train_K=fine.steps,
                                  eval_K=fine.steps, **metric))
            np.savez_compressed(target/f'grid-AM-{source_name}-B64-train64-eval64.npz',
                                reward=arrays['reward'].numpy(), energy=arrays['energy'].numpy(),
                                cost=arrays['cost'].numpy())
            budget_rows.append(dict(seed=seed, method=f'AM-{source_name}-B64-K64',
                                    train_K=fine.steps, update=c.actor_updates, batch_size=64,
                                    training_paths=c.actor_updates*64,
                                    generated_time_steps=c.actor_updates*64*fine.steps,
                                    actor_updates=c.actor_updates,
                                    actor_true_gradient_points=c.actor_updates*64 if source_name == 'oracle' else 0,
                                    checkpoint_elapsed_seconds=checkpoint['elapsed_seconds'],
                                    extra_diagnostic_seconds=checkpoint['extra_diagnostic_seconds'],
                                    elapsed_minus_extra_diagnostic_seconds=
                                    checkpoint['elapsed_seconds']-checkpoint['extra_diagnostic_seconds']))
        for checkpoint_index, (name, step) in enumerate(d.checkpoint_methods):
            actor, optimizer, _ = load_actor_checkpoint(source/name/f'checkpoint-{step}.pt', c, base)
            rows, summaries = repeated_gradient_probe(actor, optimizer, base, models, c, seed,
                                                       checkpoint_index, name, step, d, target)
            probe_rows.extend(rows); probe_summaries.extend(summaries)
            pd.DataFrame(probe_rows).to_csv(diagnostic_root/'probe-repetitions.csv', index=False)
            pd.DataFrame(probe_summaries).to_csv(diagnostic_root/'probe-summary.csv', index=False)
            record_event(out, 'oracle_probe_completed', seed=seed, checkpoint=name, step=step)
        frozen_after = [model_hash(m) for m in [base, *models]]
        if frozen_before != frozen_after:
            raise AssertionError('Diagnostic changed frozen models')
        frozen_hash_rows.append(dict(seed=seed, initial_actor_hash=initial_hash,
                                     frozen_hashes_before=frozen_before, frozen_hashes_after=frozen_after))
        for filename, data in [('checkpoint-evaluations.csv', checkpoint_rows),
                               ('budgets.csv', budget_rows), ('grid-sensitivity.csv', grid_rows)]:
            pd.DataFrame(data).to_csv(diagnostic_root/filename, index=False)
        pd.DataFrame(frozen_rows).to_csv(diagnostic_root/'frozen-reference.csv', index=False)
        save_json(diagnostic_root/'frozen-hash-audit.json', frozen_hash_rows)
        record_event(out, 'oracle_diagnostic_seed_completed', seed=seed)
        print(f'{seed}: Oracle gain-space probes completed', flush=True)
    checkpoints = pd.DataFrame(checkpoint_rows)
    effects, effect_summaries = [], []
    for metric in ('cost', 'reward', 'energy'):
        for schedule in ('equal_updates', 'equal_rollouts'):
            detail, summary = seed_effects(checkpoints, metric, schedule, c.actor_updates)
            effects.extend(detail); effect_summaries.extend(summary)
    pd.DataFrame(effects).to_csv(diagnostic_root/'paired-effects-by-seed.csv', index=False)
    pd.DataFrame(effect_summaries).to_csv(diagnostic_root/'paired-effects-summary.csv', index=False)
    record_event(out, 'oracle_diagnostic_complete', seeds=len(c.seeds), repeats=d.repeats)
    return dict(checkpoints=checkpoints, budgets=pd.DataFrame(budget_rows),
                probes=pd.DataFrame(probe_rows), probe_summaries=pd.DataFrame(probe_summaries),
                grid=pd.DataFrame(grid_rows), frozen=pd.DataFrame(frozen_rows), effects=pd.DataFrame(effects),
                effect_summaries=pd.DataFrame(effect_summaries), replay=replay,
                frozen_hashes=frozen_hash_rows, directory=diagnostic_root)


def complete_oracle_curves(c, out, diagnostic_root):
    former = pd.read_csv(out/'learning-curves.csv')
    labels = {'AM-learned': 'AM-learned-B64', 'AM-oracle': 'AM-oracle-B64',
              'AM-learned-2x': 'AM-learned-B128', 'Frozen Flow': 'Frozen Flow'}
    former = former[former.method.isin(labels) &
                    (former.train_K.eq(c.steps) | former.method.eq('Frozen Flow'))].copy()
    former['method'] = former.method.map(labels)
    pieces = [former]
    for seed in c.seeds:
        new = pd.read_csv(diagnostic_root/f'seed-{seed}'/'AM-oracle-B128'/'curve.csv')
        new['method'] = 'AM-oracle-B128'
        pieces.append(new)
    combined = pd.concat(pieces, ignore_index=True)
    combined.to_csv(diagnostic_root/'complete-learning-curves.csv', index=False)
    return combined


def make_oracle_diagnostic_figures(result, c, out):
    directory = result['directory']
    curves = complete_oracle_curves(c, out, directory)
    names = ('AM-learned-B64', 'AM-oracle-B64', 'AM-learned-B128', 'AM-oracle-B128')
    colors = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red')
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for ax, metric, ylabel in zip(axes, ('reward', 'energy', 'cost'),
                                  ('True reward (higher better)', 'Path KL / energy', 'True J (lower better)')):
        for name, color in zip(names, colors):
            subset = curves[curves.method.eq(name)]
            grouped = subset.groupby('update')[metric].agg(['mean', 'std'])
            x = grouped.index.to_numpy()
            y = grouped['mean'].to_numpy()
            sd = grouped['std'].fillna(0).to_numpy()
            ax.plot(x, y, label=name, color=color)
            ax.fill_between(x, y-sd, y+sd, color=color, alpha=.12)
        baseline = curves[curves.method.eq('Frozen Flow')].groupby('update')[metric].mean()
        ax.plot(baseline.index, baseline.values, color='0.35', ls='--', label='Frozen Flow')
        ax.set_xlabel('Actor updates')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('K32: complete four-arm learning curves; band = training-seed SD')
    fig.tight_layout()
    fig.savefig(directory/'four-arm-learning-curves.png', dpi=160, bbox_inches='tight')
    display(fig); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    effects = result['effect_summaries']
    for ax, metric, title in zip(axes, ('cost', 'reward', 'energy'), ('True J', 'True reward', 'Path KL')):
        selected = effects[effects.metric.eq(metric) & effects.schedule.eq('equal_updates')]
        ordered = [selected[selected.effect.eq(name)].iloc[0] for name in
                   ('A64', 'A128', 'S_learned', 'S_oracle', 'I')]
        ax.bar(range(len(ordered)), [row['mean'] for row in ordered], color=colors[:2]+('0.4', '0.6', '0.8'))
        ax.set_xticks(range(len(ordered)), ('A64', 'A128', 'S-L', 'S-O', 'I'))
        ax.axhline(0, color='black', lw=.8)
        ax.set_title(title+' paired effects')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('300 updates; descriptive means across existing 3 development seeds')
    fig.tight_layout()
    fig.savefig(directory/'paired-effects.png', dpi=160, bbox_inches='tight')
    display(fig); plt.close(fig)
    summaries = result['probe_summaries'].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for batch, color in zip((64, 128), ('tab:blue', 'tab:orange')):
        subset = summaries[summaries.batch_size.eq(batch)]
        grouped = subset.groupby(['checkpoint_actor', 'checkpoint_step'])[
            ['raw_cosine_mean', 'clipped_cosine_mean', 'step_cosine_mean',
             'D_J_mean', 'D_R_mean']].mean()
        x = np.arange(len(grouped))+(0 if batch == 64 else .12)
        for metric, marker, style in [('raw_cosine_mean', 'o', '-'),
                                      ('clipped_cosine_mean', 's', '--'),
                                      ('step_cosine_mean', '^', ':')]:
            axes[0].plot(x, grouped[metric], marker=marker, linestyle=style,
                         color=color, label=f'B{batch} {metric}')
        axes[1].plot(x, grouped.D_J_mean, 'o-', color=color, label=f'B{batch}: D_J')
        axes[1].plot(x, grouped.D_R_mean, 's--', color=color, label=f'B{batch}: D_R')
        labels = [f'{name}/{step}' for name, step in grouped.index]
        for ax in axes:
            ax.set_xticks(np.arange(len(labels))+.06, labels, rotation=20, ha='right')
            ax.grid(alpha=.2)
    axes[0].set_title('Raw → clipped → Adam step cosine')
    axes[1].set_title('Paired virtual-step effect (oracle minus learned)')
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory/'gradient-adam-virtual-step.png', dpi=160, bbox_inches='tight')
    display(fig); plt.close(fig)
    grid = result['grid']
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = [(c.steps, c.steps), (c.steps, 2*c.steps), (2*c.steps, 2*c.steps)]
    for j, (train_k, eval_k) in enumerate(positions):
        selected = grid[(grid.train_K.eq(train_k)) & (grid.eval_K.eq(eval_k)) &
                        (grid.method.isin(('AM-learned-B64', 'AM-oracle-B64')))]
        pivot = selected.pivot(index='seed', columns='method', values='cost')
        values = pivot['AM-learned-B64']-pivot['AM-oracle-B64']
        ax.scatter(np.full(len(values), j)+np.linspace(-.06, .06, len(values)), values,
                   color='tab:purple', label='Each seed' if j == 0 else None)
        ax.plot(j, values.mean(), 'kD', label='Seed mean' if j == 0 else None)
    ax.axhline(0, color='black', lw=.8)
    ax.set_xticks(range(3), [f'train{tr}/eval{ev}' for tr, ev in positions])
    ax.set_ylabel('J learned − J oracle (positive favors oracle)')
    ax.set_title('K32/K64 grid sensitivity: B64 only')
    ax.legend()
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(directory/'grid-sensitivity.png', dpi=160, bbox_inches='tight')
    display(fig); plt.close(fig)
    return ['four-arm-learning-curves.png', 'paired-effects.png',
            'gradient-adam-virtual-step.png', 'grid-sensitivity.png']


def markdown_table(frame, columns=None, max_rows=None):
    selected = frame if columns is None else frame[columns]
    if max_rows is not None:
        selected = selected.head(max_rows)
    selected = selected.copy()
    for column in selected.select_dtypes(include='number').columns:
        selected[column] = selected[column].map(lambda x: f'{x:.6g}' if pd.notna(x) else 'NA')
    header = '| ' + ' | '.join(selected.columns) + ' |'
    rule = '| ' + ' | '.join('---' for _ in selected.columns) + ' |'
    body = ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in selected.itertuples(index=False, name=None)]
    return '\n'.join([header, rule, *body])


def make_oracle_diagnostic_report(result, c, d, figures):
    end = result['checkpoints'][result['checkpoints']['update'].eq(c.actor_updates)]
    effects = result['effect_summaries']
    main = effects[(effects.schedule.eq('equal_updates')) & effects.metric.eq('cost')].set_index('effect')
    reward = effects[(effects.schedule.eq('equal_updates')) & effects.metric.eq('reward')].set_index('effect')
    energy = effects[(effects.schedule.eq('equal_updates')) & effects.metric.eq('energy')].set_index('effect')
    a64, a128 = main.loc['A64'], main.loc['A128']
    r64, r128 = reward.loc['A64'], reward.loc['A128']
    e64, e128 = energy.loc['A64'], energy.loc['A128']
    if a64.descriptive_ci95_low > 0 and a128.descriptive_ci95_low > 0:
        opening = ('Both batches show descriptive seed-level J improvement for Oracle, '
                   'supporting recoverable gradient-label loss in this task; three examined seeds '
                   'do not establish a general algorithmic advantage.')
    elif r64['mean'] > 0 and r128['mean'] > 0 and e64['mean'] < 0 and e128['mean'] < 0:
        opening = ('The clearest observed pattern is an Oracle reward gain accompanied by '
                   'additional path KL/energy. Its net J effect must be judged separately; '
                   'the present three examined seeds do not isolate label error from '
                   'sampling and optimization limitations.')
    else:
        opening = ('The current examined task does not exhibit a clear, stable recoverable '
                   'J gain from Oracle labels across both batches. This is not evidence '
                   'that terminal-gradient errors never matter.')
    grid = result['grid']
    grid_effects = []
    for train_k, eval_k in ((c.steps, c.steps), (c.steps, 2*c.steps), (2*c.steps, 2*c.steps)):
        selected = grid[grid.train_K.eq(train_k) & grid.eval_K.eq(eval_k) &
                        grid.method.isin(('AM-learned-B64', 'AM-oracle-B64'))]
        pivot = selected.pivot(index='seed', columns='method', values='cost')
        for seed, value in (pivot['AM-learned-B64']-pivot['AM-oracle-B64']).items():
            grid_effects.append(dict(seed=seed, train_K=train_k, eval_K=eval_k, J_oracle_gain=value))
    grid_effects = pd.DataFrame(grid_effects)
    signs = grid_effects.pivot(index='seed', columns=['train_K', 'eval_K'], values='J_oracle_gain')
    stable_grid = all((row > 0).all() or (row < 0).all() for _, row in signs.iterrows())
    if not stable_grid:
        grid_note = 'Oracle–learned J signs differ across grid settings for at least one seed; no grid-stable conclusion.'
    else:
        grid_note = ('Signs agree within each seed across the measured grid settings, '
                     'but K64 is only a B64 sensitivity study, not a full grid×batch interaction.')
    probe = result['probe_summaries']
    raw_cos, step_cos = float(probe.raw_cosine_mean.mean()), float(probe.step_cosine_mean.mean())
    if step_cos > raw_cos:
        adam_note = ('At these shared-history virtual updates, Adam-step directions are on average '
                     'more aligned than raw gradients; this does not imply independent long-run '
                     'learned/oracle histories converge.')
    else:
        adam_note = ('At these shared-history virtual updates, Adam does not on average increase '
                     'directional alignment over the raw gradients.')
    lines = [
        '# Oracle AM gain-space diagnosis — executed Notebook 2 extension',
        '',
        f'Existing development seeds {list(c.seeds)}; K32, {c.actor_updates} Adam updates, beta={c.beta}. '
        f'Each checkpoint/batch uses {d.repeats} independent training-noise probes and '
        f'{d.virtual_paths} fresh common-noise paths per virtual pair. '
        'These probes are not independent training seeds or confirmatory trials.',
        '',
        '## Direct answer', '',
        opening, '',
        f'Equal-update paired J oracle gain: A64={a64["mean"]:.6g} '
        f'(descriptive 95% t interval [{a64.descriptive_ci95_low:.6g}, {a64.descriptive_ci95_high:.6g}]); '
        f'A128={a128["mean"]:.6g} '
        f'([{a128.descriptive_ci95_low:.6g}, {a128.descriptive_ci95_high:.6g}]). '
        'Positive means lower true J for Oracle. Intervals summarize training-seed variation, '
        'not 4096 evaluation paths or 32 probes as independent seeds.', '',
        f'Oracle reward effects: B64={r64["mean"]:.6g}, B128={r128["mean"]:.6g}; '
        f'energy effects (Oracle minus learned): B64={-e64["mean"]:.6g}, '
        f'B128={-e128["mean"]:.6g}. True J = full finite-chain path KL − beta × true reward.',
        '',
        '## Four arms at fixed update 300', '',
        markdown_table(end, ['seed', 'method', 'reward', 'reward_mc_se', 'energy',
                             'energy_mc_se', 'cost', 'cost_mc_se',
                             'terminal_shift_from_frozen']), '',
        'Frozen Flow reference on the identical SDE evaluation noise:', '',
        markdown_table(result['frozen'], ['seed', 'reward', 'energy', 'cost']), '',
        '## Budget (checkpoint elapsed includes fixed curve evaluations; extra diagnostics separately counted)',
        '',
        markdown_table(result['budgets'], ['seed', 'method', 'train_K', 'update', 'training_paths',
                                           'generated_time_steps', 'actor_updates',
                                           'actor_true_gradient_points', 'checkpoint_elapsed_seconds',
                                           'extra_diagnostic_seconds']), '',
        'B64/300 and B128/150 both use 19200 training paths, but they have different '
        'optimizer-step counts and not equal wall-clock or total compute. '
        'B128/300 uses twice the trajectories of B64/300; batch effects are not a pure variance estimate.',
        '',
        '## Equal-rollout-budget results (B64/300 versus B128/150)', '',
        markdown_table(result['checkpoints'][
            ((result['checkpoints'].batch_size.eq(64) & result['checkpoints']['update'].eq(c.actor_updates)) |
             (result['checkpoints'].batch_size.eq(128) & result['checkpoints']['update'].eq(c.actor_updates//2)))],
            ['seed', 'method', 'update', 'reward', 'energy', 'cost']), '',
        '## Paired 2×2 effects', '',
        'For J: A_B = J_learned,B − J_oracle,B, S_source = J_source,64 − J_source,128, '
        'I=A64−A128=S_learned−S_oracle. For reward, higher-better reverses the '
        'first two signs and I=S_learned−S_oracle. The energy table uses lower-better convention.',
        '',
        markdown_table(result['effect_summaries'], ['schedule', 'metric', 'effect', 'mean',
                                                    'seed_sd', 'descriptive_ci95_low',
                                                    'descriptive_ci95_high', 'positive_seeds']), '',
        'Per-seed effects are in paired-effects-by-seed.csv; all paired final evaluation arrays '
        'and checkpoint 0/150/300 results are retained.', '',
        '## Repeated common-checkpoint gradient, Adam and paired virtual-step evidence', '',
        markdown_table(probe, ['seed', 'checkpoint_actor', 'checkpoint_step', 'batch_size',
                               'mean_d_norm', 'd_norm_repeat_sd', 'trace_cov_qL',
                               'trace_cov_qO', 'trace_cov_d', 'trace_cross_cov_qO_d',
                               'variance_identity_residual', 'raw_cosine_mean',
                               'q_norm_ratio_mean', 'projection_c_mean',
                               'perpendicular_ratio_mean', 'clipped_cosine_mean',
                               'step_cosine_mean', 'step_norm_ratio_mean',
                               'step_relative_difference_mean',
                               'D_J_mean', 'D_J_repeat_sd', 'D_J_eval_mc_se_rms',
                               'D_R_mean', 'D_R_repeat_sd', 'D_R_eval_mc_se_rms',
                               'projection_undefined_count', 'projection_negative_count']), '',
        'The mean of d=qL−qO is a finite-repetition estimate, not known population bias. '
        'The exact covariance identity includes 2 tr Cov(qO,d); no independence '
        'assumption is made. Probe-repetitions.csv gives every raw/clipped/step norm, '
        'cosine, projection coefficient/perpendicular ratio, and paired D_J/D_R. '
        'Their Monte Carlo SE is computed from per-path differences, whereas '
        'repeat SD describes variation across fresh training batches at a fixed actor. '
        'Near-zero qO projections are marked undefined and all repeats remain counted.',
        '',
        adam_note, '',
        '## Grid sensitivity', '',
        markdown_table(grid_effects), '', grid_note, '',
        'K32 actors are evaluated under both K32 and K64 finite chains; K64 actors '
        'were retrained at K64 for all three seeds with B64. Training grid and '
        'evaluation grid are distinct variables. K64 does not provide a complete 2×2.',
        '',
        '## Provenance and limits', '',
        'The three original arms and first-seed K64 arms were retrained into a new unique '
        'run; original historical outputs were not overwritten or pooled as extra seeds. '
        'The historical baseline replay audit compares data, frozen checkpoint tensors, '
        'actor and Adam states at 0/150/300, and saved paired evaluation arrays. '
        'Oracle-B128 and K64 seeds 2–3 are new training. All repeated-gradient and '
        'virtual-step measurements are new diagnostics. Initial actor and frozen-model '
        'hashes are audited for every seed.', '',
        'This is a fixed objective with a learned terminal-label gradient, not an uncertain '
        'true terminal cost. No new uncertainty algorithm was added. '
        'A small or interval-overlapping effect does not demonstrate equivalence. '
        'The 32 probes on shared states isolate local update mechanisms, but cannot '
        'identify the causal contribution of independent long-run optimizer histories. '
        'Three previously inspected training seeds are an exploratory diagnosis.', '',
        'Figures: '+', '.join(figures)+'. Full numerical outputs are in this run directory.'
    ]
    return '\n'.join(lines)+'\n'


ORACLE_CFG = oracle_diagnostic_configuration(CFG)
ORACLE_RESULT = run_oracle_diagnostic(CFG, OUT, ORACLE_CFG)
ORACLE_FIGURES = make_oracle_diagnostic_figures(ORACLE_RESULT, CFG, OUT)
ORACLE_REPORT = make_oracle_diagnostic_report(ORACLE_RESULT, CFG, ORACLE_CFG, ORACLE_FIGURES)
(ORACLE_RESULT['directory']/'report.md').write_text(ORACLE_REPORT)
save_json(ORACLE_RESULT['directory']/'completion.json', dict(
    status='complete', profile=CFG.profile, seeds=list(CFG.seeds), figures=ORACLE_FIGURES,
    repeats=ORACLE_CFG.repeats, virtual_paths=ORACLE_CFG.virtual_paths,
    new_actor_runs=len(CFG.seeds)+2*(len(CFG.seeds)-1),
    source_sha256=hashlib.sha256((ROOT/'notebooks/02_flow_am_terminal_gradient.py').read_bytes()).hexdigest(),
    baseline_retrained_and_replayed=len(ORACLE_RESULT['replay']),
    total_probe_groups=len(ORACLE_RESULT['probe_summaries']),
    total_probe_pairs=len(ORACLE_RESULT['probes'])))
display(Markdown(ORACLE_REPORT))
print('Oracle diagnosis:', ORACLE_RESULT['directory'].relative_to(ROOT))
