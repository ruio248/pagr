"""All-lean terminal-consistent minimax AM comparison.

No full-adjoint branch exists in matching_gradients or update. This reuses the
previous convex adversary, but not its full-adjoint training update. A local
minimax certificate concerns lean matching gradients, not true-J gradients.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'am_terminal_consistent'
sys.path.insert(0, str(PREVIOUS))
import tc_am as base
rh = base.rh
Adversary = base.Adversary
sys.path.insert(0, str(ROOT))


@dataclass
class Options(base.Options):
    adjoint: str = 'lean'


def matching_gradients(actor, paths, terminals, task, cfg):
    """Only lean: a_k=F.T a_{k+1}+dt*ell_x, target=-S.T a_{k+1}.

    Parameter gradient of the frozen-path matching objective, NOT generally
    the gradient of true control cost. Time integration matches previous AM.
    """
    u, jac = base.prediction(actor, paths, cfg, spatial_jacobian=False)
    assert jac is None
    f, sigma = base.coefficients(task, cfg)
    running = (torch.zeros_like(paths[:, :-1]) if task is rh.arm else
               task.running_cost_and_gradient(paths[:, :-1], cfg)[1])
    with torch.no_grad():
        a = terminals.clone()
        targets = []
        for k in range(cfg.steps-1, -1, -1):
            targets.append(-a @ sigma)
            a = a @ f + cfg.dt*running[:, k][None]
        targets = torch.stack(targets[::-1], 2)
    losses = .5*(u[None]-targets).square().sum(-1).sum(-1).mean(-1)*cfg.dt
    qs = torch.stack([rh.rel.flat_grad(loss, actor, retain_graph=i+1<len(losses))
                     for i, loss in enumerate(losses)])
    return qs.detach(), targets, losses.detach()


def update(actor, optimizer, paths, critics, task, cfg, options, state, adversary, step):
    assert options.adjoint == 'lean'
    assert options.method in ('am_adam', 'am_rms', 'robust', 'norm_matched')
    terminals = torch.stack([task.raw_gradient(c, paths[:, -1]) for c in critics])
    if options.method == 'am_adam':
        terminals = terminals.mean(0, keepdim=True)
    qs, _, losses = matching_gradients(actor, paths, terminals, task, cfg)
    mean = qs.mean(0)
    info = dict(adjoint='lean', matching_loss=losses.mean().item(), gradient_norm=mean.norm().item())
    if options.method == 'am_adam':
        optimizer.zero_grad()
        offset = 0
        for p in actor.parameters():
            p.grad = mean[offset:offset+p.numel()].reshape_as(p).clone()
            offset += p.numel()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.)
        optimizer.step()
        return info
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


def load_inputs(name, seed):
    previous_protocol = json.loads((PREVIOUS/'protocol.json').read_text())
    task = rh.get_task(name)
    cfg = task.Config(**previous_protocol['configs'][name])
    folder = PREVIOUS/'pilot'/name/f'seed_{seed}'
    manifest = json.loads((folder/'input_manifest.json').read_text())
    resolved = {}
    for file, digest in manifest.items():
        p = folder/'critics'/file
        assert rh.rel.sha(p)==digest
        resolved[str(p)] = digest
    critics = []
    for m in range(5):
        critic = task.TerminalCritic(cfg.critic_width)
        critic.load_state_dict(torch.load(folder/'critics'/f'critic_{m}.pt', weights_only=True)['state_dict'])
        critic.eval()
        for p in critic.parameters():
            p.requires_grad_(False)
        critics.append(critic)
    c = base.validation_matrix(critics, folder/'critics/data.npz')
    return task, cfg, critics, c, resolved


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
    print(name, seed, options.method, options.rho, 'J', round(final['cost'],6),
          'seconds', round(result['seconds'],1), flush=True)
    return result
