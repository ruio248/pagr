# %% [markdown]
# # Notebook 2b — current-policy Q-gradient measurement during joint TD + AM
#
# This is a **read-only extension** of Notebook 2's completed three-seed joint
# experiment. It loads the saved N=0/150/300 joint Actor, TD critic, and Adam
# state. It does not retrain, replace the old run, or use the fixed-policy
# analytic Q^{pi_ref} as the current-policy truth.
#
# At fixed (k,y,a), Q^{pi_theta} is the expected terminal reward when the first
# action is held at a and all later actions come from the saved current Actor.
# Because y_next = y + B a + sigma_env * xi, its action gradient obeys
#
#     grad_a Q^{pi_theta} = B^T / sigma_env * E[xi * G].
#
# We estimate this expectation with two independent Monte Carlo splits. The
# last environment noise is analytically integrated for k<H-1 to reduce
# variance. A leave-one-out (sample covariance) baseline stays unbiased.
# At k=H-1, the exact policy-independent value/gradient is used **only for
# calibration of the Monte Carlo estimator**, not as a label elsewhere.
#
# The study compares critic and reference gradients at the *same* Actor and
# Flow paths, propagates them through the same Lean adjoint, applies separate
# virtual Adam steps from identical optimizer histories, and evaluates the
# one-step branches on paired closed-loop episodes. Only three diagnostic
# states per snapshot are used, not a real training batch; this is a mechanism
# probe, not a new or sample-efficient QAM algorithm. A finite-sample MC
# reference is not an exact Oracle, and inconclusive intervals remain so.

# %%
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import secrets
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ncx2
import torch


ROOT = Path.cwd()
CONFIG = ROOT/'configs/qam_joint_mc_diagnostic.json'
CORE_SOURCE = ROOT/'notebooks/02_qam_td_terminal_error.py'


def load_core():
    spec = importlib.util.spec_from_file_location('pagr_qam_core', CORE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


N = load_core()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_config(run):
    raw = json.loads((run/'config-effective.json').read_text())
    fields = N.Config.__dataclass_fields__
    return N.Config(**{k: tuple(v) if k.endswith('_updates') and isinstance(v,list) else v
                       for k,v in raw.items() if k in fields})


def load_snapshot(run, seed, step, c):
    sub = run/f'seed-{seed}'
    checkpoint = sub/f'joint-B-N{step}.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    base = N.ConditionalVelocity(c)
    base.load_state_dict(torch.load(sub/'base.pt', map_location='cpu', weights_only=True))
    actor, critic = N.ConditionalVelocity(c), N.Critic(c)
    actor.load_state_dict(saved['actor'])
    critic.load_state_dict(saved['critic'])
    actor.eval(); critic.eval(); base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(actor.parameters(),lr=c.actor_lr)
    optimizer.load_state_dict(saved['actor_optimizer'])
    return base,actor,critic,optimizer,checkpoint


def selected_states(run, seed, times, per_time):
    """Pre-specified evaluation states; no reward or gradient-based selection."""
    data = np.load(run/f'seed-{seed}'/'eval-transitions.npz')
    rng = np.random.default_rng(N.seed_for(seed, 1100))
    rows=[]
    for k in times:
        candidates=np.flatnonzero(data['k']==k)
        assert len(candidates)>=per_time
        for index in rng.choice(candidates,size=per_time,replace=False):
            rows.append((int(k),int(index),torch.from_numpy(data['y'][index]).clone()))
    return rows


@torch.no_grad()
def mc_current_policy_reference(task, actor, base, c, k, y, action,
                                samples, seed, chunk=256):
    """Independent current-policy Q/gradient MC; no learned Q or Q_ref access.

    Only the first environment transition's Gaussian score is differentiated.
    At the final future action, integrate the last environment noise exactly.
    The unbiased sample-covariance baseline uses n/(n-1), not a plug-in mean.
    """
    if c.sigma_env<=0 or samples<2 or not (0<=k<c.horizon):
        raise ValueError('Gaussian score requires positive noise, n>=2 and valid k')
    if y.shape!=(c.dimension,) or action.shape!=(c.dimension,):
        raise ValueError('one state and one fixed action are required')
    rng=N.generator(seed)
    all_noise=[];all_return=[]
    discount=c.gamma**(c.horizon-1-k)
    for start in range(0,samples,chunk):
        count=min(chunk,samples-start)
        xi0=torch.randn((count,c.dimension),generator=rng)
        state,reward,_=task.step(k,y.expand(count,-1),action.expand(count,-1),xi0)
        if k==c.horizon-1:
            terminal=reward.double().numpy()
        else:
            for j in range(k+1,c.horizon):
                kk=torch.full((count,),j,dtype=torch.long)
                z=torch.randn((count,c.dimension),generator=rng)
                flow_noise=torch.randn((count,c.flow_steps,c.dimension),generator=rng)
                next_action=N.rollout(actor,base,c,kk,state,z,flow_noise)[0][:,-1]
                if j==c.horizon-1:
                    mu=(state+next_action@task.B.T).double().numpy()
                    noncentrality=np.einsum('ij,ij->i',mu,mu)/c.sigma_env**2
                    terminal=ncx2.cdf(task.radius**2/c.sigma_env**2,
                                      c.dimension,noncentrality)
                else:
                    noise=torch.randn((count,c.dimension),generator=rng)
                    state,_,_=task.step(j,state,next_action,noise)
        all_noise.append(xi0.double().numpy())
        all_return.append(discount*np.asarray(terminal,dtype=np.float64))
    xi=np.concatenate(all_noise)
    returns=np.concatenate(all_return)
    assert len(xi)==samples and np.isfinite(returns).all()
    centered=(xi-xi.mean(0))*(returns-returns.mean())[:,None]
    # E[xi]=0. The n/(n-1) factor corrects the sample-mean baseline bias.
    influence=(samples/(samples-1))*centered@task.B64/c.sigma_env
    gradient=influence.mean(0)
    gradient_se=influence.std(0,ddof=1)/math.sqrt(samples)
    result=dict(q=float(returns.mean()),q_se=float(returns.std(ddof=1)/math.sqrt(samples)),
                gradient=gradient,gradient_se=gradient_se,samples=samples,
                return_mean=float(returns.mean()))
    if not (np.isfinite(gradient).all() and np.isfinite(gradient_se).all()):
        raise FloatingPointError('current-policy MC score gradient')
    return result


def virtual_actor(actor, optimizer, gradient, c):
    branch=copy.deepcopy(actor)
    branch_optimizer=torch.optim.Adam(branch.parameters(),lr=c.actor_lr)
    branch_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    before=N.parameter_vector(branch).clone()
    N.set_gradient(branch,gradient)
    torch.nn.utils.clip_grad_norm_(branch.parameters(),c.actor_clip,error_if_nonfinite=True)
    branch_optimizer.step()
    branch.zero_grad(set_to_none=True)
    return branch,N.parameter_vector(branch)-before


@torch.no_grad()
def paired_one_step_evaluation(task, base, c, actors, episodes, seed):
    """Shared initial and exogenous noise, separate closed-loop states."""
    rng=N.generator(seed)
    initial=torch.randn((episodes,c.dimension),generator=rng)
    states={name:initial.clone() for name in actors}
    terminal={}
    for k in range(c.horizon):
        kk=torch.full((episodes,),k,dtype=torch.long)
        z=torch.randn((episodes,c.dimension),generator=rng)
        flow_noise=torch.randn((episodes,c.flow_steps,c.dimension),generator=rng)
        env_noise=torch.randn((episodes,c.dimension),generator=rng)
        for name,actor in actors.items():
            action=N.rollout(actor,base,c,kk,states[name],z,flow_noise)[0][:,-1]
            states[name],terminal[name],_=task.step(k,states[name],action,env_noise)
    learned=terminal['learned'].numpy()
    reference=terminal['mc_reference'].numpy()
    paired=reference-learned
    return dict(success_before=float(terminal['before'].mean()),
                success_learned=float(learned.mean()),
                success_mc_reference=float(reference.mean()),
                delta_success_reference_minus_learned=float(paired.mean()),
                delta_success_se=float(paired.std(ddof=1)/math.sqrt(episodes)),
                eval_episodes=episodes)


# %% [markdown]
# ## Fixed-snapshot measurement
#
# One state at each k=0/4/7 is drawn from the held-out evaluation transitions
# before looking at either gradient. The same state is reused across N, while
# the current Actor generates its own action endpoint. Each endpoint gets two
# independent reference estimates. We compare their disagreement with the
# learned-vs-reference discrepancy, but do not call a finite-sample estimate
# ground truth. The three-state microbatch is intentionally **not** the B=64
# batch that was used in actual training.

# %%
def diagnose_snapshot(task, run, seed, step, c, design, output):
    base,actor,critic,optimizer,checkpoint=load_snapshot(run,seed,step,c)
    before=N.parameter_vector(actor).clone()
    selected=selected_states(run,seed,design['environment_times'],design['states_per_time'])
    k=torch.tensor([item[0] for item in selected],dtype=torch.long)
    y=torch.stack([item[2] for item in selected])
    z,flow_noise=N.paired_generation_noise(c,len(k),N.seed_for(seed,1400+step))
    paths,_=N.rollout(actor,base,c,k,y,z,flow_noise)
    endpoint=paths[:,-1]
    learned_grad=-N.terminal_label(task,critic,c,k,y,endpoint,'learned')/c.beta
    with torch.no_grad():
        learned_q=critic(k,y,endpoint).mean(-1).numpy()
    split_grad=[];point_rows=[]
    for i,(time,index,_) in enumerate(selected):
        estimates=[]
        for split in (0,1):
            estimate=mc_current_policy_reference(
                task,actor,base,c,time,y[i],endpoint[i],
                design['mc_samples_per_split'],
                N.seed_for(seed,1600+step*100+time*10+i*2+split),
                design['mc_chunk'])
            estimates.append(estimate)
        g1,g2=(e['gradient'] for e in estimates)
        g=(g1+g2)/2
        se=np.sqrt(estimates[0]['gradient_se']**2+
                   estimates[1]['gradient_se']**2)/2
        q=(estimates[0]['q']+estimates[1]['q'])/2
        q_se=math.hypot(estimates[0]['q_se'],estimates[1]['q_se'])/2
        gl=learned_grad[i].numpy()
        # k=H-1 has no future policy. Its analytic Q is a calibration check only.
        exact_q=exact_gradient_error=float('nan')
        if time==c.horizon-1:
            eq,eg=N.oracle_torch(task,k[i:i+1],y[i:i+1],endpoint[i:i+1])
            exact_q=float(eq[0])
            exact_gradient_error=float(np.linalg.norm(g-eg[0].numpy()))
        point_rows.append(dict(seed=seed,N=step,k=time,eval_transition_index=index,
                               action_norm=float(endpoint[i].norm()),
                               q_critic=float(learned_q[i]),q_mc=q,q_mc_se=q_se,
                               q_critic_minus_mc=float(learned_q[i]-q),
                               gradient_critic_norm=float(np.linalg.norm(gl)),
                               gradient_mc_norm=float(np.linalg.norm(g)),
                               gradient_critic_minus_mc_norm=float(np.linalg.norm(gl-g)),
                               gradient_mc_split_difference_norm=float(np.linalg.norm(g1-g2)),
                               gradient_mc_se_l2=float(np.linalg.norm(se)),
                               terminal_exact_q_calibration=exact_q,
                               terminal_exact_gradient_error=exact_gradient_error,
                               mc_samples=2*design['mc_samples_per_split']))
        split_grad.append((g1,g2))
    grad1=torch.as_tensor(np.stack([pair[0] for pair in split_grad]),dtype=y.dtype)
    grad2=torch.as_tensor(np.stack([pair[1] for pair in split_grad]),dtype=y.dtype)
    grad_ref=(grad1+grad2)/2
    p_learned=N.lean_adjoint(base,c,k,y,paths,-c.beta*learned_grad)
    p_reference=N.lean_adjoint(base,c,k,y,paths,-c.beta*grad_ref)
    p_split1=N.lean_adjoint(base,c,k,y,paths,-c.beta*grad1)
    p_split2=N.lean_adjoint(base,c,k,y,paths,-c.beta*grad2)
    dp=p_learned-p_reference
    recurrence=N.lean_adjoint(base,c,k,y,paths,-c.beta*(learned_grad-grad_ref))
    recurrence_error=float((dp-recurrence).abs().max())
    assert recurrence_error<3e-5,recurrence_error
    _,nu,_=N.flow_grid(c)
    label_diff=nu[None,:,None]*dp[:,1:]
    q_learned,_=N.actor_gradient(actor,base,c,k,y,paths,p_learned)
    q_reference,_=N.actor_gradient(actor,base,c,k,y,paths,p_reference)
    q_split1,_=N.actor_gradient(actor,base,c,k,y,paths,p_split1)
    q_split2,_=N.actor_gradient(actor,base,c,k,y,paths,p_split2)
    actor_learned,step_learned=virtual_actor(actor,optimizer,q_learned,c)
    actor_reference,step_reference=virtual_actor(actor,optimizer,q_reference,c)
    step_split1=N.virtual_adam(actor,optimizer,q_split1,c)
    step_split2=N.virtual_adam(actor,optimizer,q_split2,c)
    paired=paired_one_step_evaluation(
        task,base,c,dict(before=actor,learned=actor_learned,
                         mc_reference=actor_reference),
        design['paired_eval_episodes'],N.seed_for(seed,1900+step))
    file=output/f'seed-{seed}-N{step}-mechanism.npz'
    np.savez_compressed(file,k=k.numpy(),state=y.numpy(),action=endpoint.numpy(),
                        critic_gradient=learned_grad.numpy(),reference_gradient=grad_ref.numpy(),
                        reference_split1=grad1.numpy(),reference_split2=grad2.numpy(),
                        adjoint_difference=dp.numpy(),matching_label_difference=label_diff.numpy(),
                        actor_parameter_gradient_difference=(q_learned-q_reference).numpy(),
                        adam_step_difference=(step_learned-step_reference).numpy(),
                        label_rms_by_flow_time=label_diff.square().mean((0,2)).sqrt().numpy())
    assert torch.equal(before,N.parameter_vector(actor))
    row=dict(seed=seed,N=step,states=len(k),
             gradient_error_rms=float((learned_grad-grad_ref).square().mean().sqrt()),
             gradient_mc_split_rms=float((grad1-grad2).square().mean().sqrt()),
             adjoint_difference_rms=float(dp.square().mean().sqrt()),
             matching_label_difference_rms=float(label_diff.square().mean().sqrt()),
             matching_parameter_gradient_difference_norm=float((q_learned-q_reference).norm()),
             matching_parameter_gradient_mc_split_norm=float((q_split1-q_split2).norm()),
             adam_step_difference_norm=float((step_learned-step_reference).norm()),
             adam_step_mc_split_norm=float((step_split1-step_split2).norm()),
             recurrence_max_error=recurrence_error,mechanism_npz=file.name,
             checkpoint=checkpoint.relative_to(ROOT).as_posix(),**paired)
    return point_rows,row


def make_report(output, design, points, snapshots):
    def table(frame,columns):
        def cell(value):
            return f'{value:.4g}' if isinstance(value,(float,np.floating)) else str(value)
        rows=['| '+' | '.join(columns)+' |',
              '| '+' | '.join(['---']*len(columns))+' |']
        rows += ['| '+' | '.join(cell(value) for value in row)+' |'
                 for row in frame[columns].itertuples(index=False,name=None)]
        return '\n'.join(rows)
    positive=int((snapshots.delta_success_reference_minus_learned>0).sum())
    resolved=int((snapshots.delta_success_reference_minus_learned.abs()>
                  1.96*snapshots.delta_success_se).sum())
    q_mae=float(points.q_critic_minus_mc.abs().mean())
    q_mean=float(points.q_mc.mean())
    gradient_over_split=int((points.gradient_critic_minus_mc_norm>
                             points.gradient_mc_split_difference_norm).sum())
    lines=[
        '# Joint TD+AM current-policy MC gradient diagnostic','',
        'This is a **post-hoc, fixed-snapshot exploratory diagnostic**, not a '
        'second joint-training run or an algorithm win. Training checkpoints are '
        f'reused unchanged from `{design["source_run"]}`. No retraining took place.','',
        '## Estimator and scope','',
        r'For the first transition $y_{k+1}=y_k+Ba+\sigma_{env}\xi$, '
        r'$\nabla_a Q^{\pi_\theta}=B^\top\mathbb E[\xi G]/\sigma_{env}$. '
        'All subsequent actions use the **saved current Actor**, not pi_ref. '
        'The terminal environment noise is Rao-Blackwellized when k<H-1; '
        'k=H-1 uses the sampled indicator. The leave-one-out covariance '
        'baseline preserves unbiasedness for the raw score expectation. '
        f'Each point has 2 independent splits of {design["mc_samples_per_split"]} '
        'rollouts; split differences and estimated standard errors measure '
        'reference uncertainty. The estimator is finite-sample, not exact Oracle.','',
        'States are selected from held-out evaluation transitions by a fixed '
        'RNG, one at k=0/4/7 per seed and checkpoint; the same state is reused '
        'at N=0/150/300. The Actor endpoint changes with N. The 3-state '
        'microbatch is not the actual B=64 training batch.','',
        '## Pointwise Q and action-gradient measurements','',
        'The gradient error norm should be compared with its MC split difference '
        'and SE; neither is a formal simultaneous confidence bound. At k=7, '
        'the exact policy-independent Q/gradient is used solely to calibrate '
        'the MC estimator. Fixed pi_ref Q is never used for k<7.','',
        table(points,['seed','N','k','q_critic','q_mc','q_mc_se',
                      'gradient_critic_minus_mc_norm','gradient_mc_split_difference_norm',
                      'gradient_mc_se_l2','terminal_exact_gradient_error']),'',
        f'Across these selected points, critic Q mean absolute error versus MC '
        f'is {q_mae:.4g}, while mean MC Q is {q_mean:.4g}. The action-gradient '
        f'discrepancy exceeds the two-split gap at {gradient_over_split}/{len(points)} '
        'points. This is **not** the desired clean case of accurate Q values '
        'but inaccurate gradients: value errors are also substantial, and '
        'the point selection is small.','',
        '## Propagation and virtual update','',
        'Both gradients enter the same frozen-base Lean adjoint, same Flow paths '
        'and same actor optimizer state. The split-reference parameter and Adam '
        'differences indicate how much these downstream comparisons themselves '
        'move under MC noise. Virtual steps do not alter saved checkpoints.','',
        table(snapshots,['seed','N','gradient_error_rms','gradient_mc_split_rms',
                         'matching_label_difference_rms',
                         'matching_parameter_gradient_difference_norm',
                         'matching_parameter_gradient_mc_split_norm',
                         'adam_step_difference_norm','adam_step_mc_split_norm']),'',
        '## Paired closed-loop outcome after one virtual step','',
        'The same initial states, Flow noise, and environment noise are shared '
        'across the unmodified, learned-step and MC-reference-step actors. '
        'Each row is one 3-state microbatch intervention, **not** a full '
        'retraining comparison. Delta is MC-reference minus learned; the '
        'interval is the uncorrected paired episode mean ±1.96 SE. Episodes '
        'are not independent training seeds.','',
        table(snapshots,['seed','N','success_before','success_learned',
                         'success_mc_reference','delta_success_reference_minus_learned',
                         'delta_success_se','eval_episodes']),'',
        f'Positive paired point estimates: {positive}/{len(snapshots)}; '
        f'uncorrected intervals excluding zero: {resolved}/{len(snapshots)}. '
        'These are descriptive, with no multiple-testing correction and no '
        'claim that the reference is a deployable algorithm. A one-step '
        'microbatch intervention is not evidence of better final training.','',
        '## Interpretation limits','',
        'This shows whether the *measured current-policy* gradient discrepancy '
        'survives adjoint propagation and changes an Actor update. It does not '
        'by itself establish a repeatable gain in final trained performance. '
        'The MC reference needs a simulator and known Gaussian transition, uses '
        'many more rollouts than ordinary TD+AM, and can remain noisy for sparse '
        'terminal rewards. Baseline TD uses fixed offline transitions, so '
        'distribution shift is also possible. No true Q for the evolving '
        'current Actor is asserted.'
    ]
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def make_figure(output, points, snapshots):
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    x=np.arange(len(points))
    axes[0].plot(x,points.gradient_critic_minus_mc_norm,'o',label='critic − MC')
    axes[0].plot(x,points.gradient_mc_split_difference_norm,'x',label='MC split gap')
    axes[0].set(title='Action-gradient discrepancy',xlabel='seed × N × k point',ylabel='L2 norm')
    for row in snapshots.itertuples():
        arr=np.load(output/row.mechanism_npz)['label_rms_by_flow_time']
        axes[1].plot(np.linspace(0,1,len(arr)),arr,alpha=.55,
                     label=f'{row.seed} N{row.N}')
    axes[1].set(title='Propagated AM-label discrepancy',xlabel='Flow time',ylabel='RMS')
    axes[2].errorbar(np.arange(len(snapshots)),snapshots.delta_success_reference_minus_learned,
                     yerr=1.96*snapshots.delta_success_se,fmt='o',capsize=3)
    axes[2].axhline(0,color='black',lw=1)
    axes[2].set(title='One virtual step: paired success delta',xlabel='seed × N snapshot',
                ylabel='MC-reference − learned')
    for ax in axes:ax.grid(alpha=.25)
    axes[0].legend();axes[1].legend(fontsize=6,ncol=2)
    fig.tight_layout()
    fig.savefig(output/'joint-mc-diagnostic.png',dpi=160)
    plt.close(fig)


def run_diagnostic():
    design=json.loads(CONFIG.read_text())
    run=ROOT/design['source_run']
    provenance=json.loads((run/'provenance.json').read_text())
    assert design['seeds']==provenance['seeds']
    assert set(design['actor_updates']).issubset(set(json.loads(
        (run/'config-effective.json').read_text())['curve_updates']))
    c=source_config(run)
    task=N.SparseReachTask(c)
    saved_task=json.loads((run/'task.json').read_text())
    assert np.array_equal(task.B64,np.asarray(saved_task['B']))
    assert task.radius==saved_task['radius']
    output=ROOT/'runs'/f'02-qam-joint-mc-diagnostic-{secrets.token_hex(5)}'
    output.mkdir(parents=True,exist_ok=False)
    provenance_out=dict(profile=design['profile'],source_run=design['source_run'],
                        source_run_provenance_sha256=sha256(run/'provenance.json'),
                        source_script_sha256=sha256(ROOT/'notebooks/02b_joint_mc_reference_diagnostic.py'),
                        core_script_sha256=sha256(CORE_SOURCE),config_sha256=sha256(CONFIG),
                        checkpoints={})
    all_points=[];all_snapshots=[]
    for seed in design['seeds']:
        for step in design['actor_updates']:
            points,snapshot=diagnose_snapshot(task,run,seed,step,c,design,output)
            all_points.extend(points);all_snapshots.append(snapshot)
            provenance_out['checkpoints'][f'{seed}/N{step}']=sha256(ROOT/snapshot['checkpoint'])
            print('diagnosed',seed,step,'gradient error RMS',
                  snapshot['gradient_error_rms'],'MC split RMS',
                  snapshot['gradient_mc_split_rms'],flush=True)
    points=pd.DataFrame(all_points)
    snapshots=pd.DataFrame(all_snapshots)
    points.to_csv(output/'pointwise.csv',index=False)
    snapshots.to_csv(output/'snapshots.csv',index=False)
    (output/'provenance.json').write_text(json.dumps(provenance_out,indent=2))
    (output/'config-effective.json').write_text(json.dumps(design,indent=2))
    make_report(output,design,points,snapshots)
    make_figure(output,points,snapshots)
    print('OUTPUT',output,flush=True)
    return output


# %% [markdown]
# Run from the repository root. The unique output directory contains pointwise
# estimates, snapshot/one-step outcome tables, NPZ mechanisms, a figure, a
# reproducibility manifest, and the interpretation report. It is deliberately
# separate from Notebook 2's frozen training run.

# %%
if __name__=='__main__':
    DIAGNOSTIC_OUTPUT=run_diagnostic()
    print((DIAGNOSTIC_OUTPUT/'report.md').read_text()[:1800])

