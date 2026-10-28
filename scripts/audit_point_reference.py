"""Independent, fail-closed precision audit of the frozen point-obstacle policy.

This is a privileged *diagnostic*, not a deployable Actor method.  It never
reads test outcomes or changes the environment, reward, QAM, or optimizer.
The point list has its own prior Git commit and is copied into the run before
any numerical result is computed.  Completed policy tables are cached with
source/Actor/config hashes so an interrupted long audit can safely resume.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'notebooks/02_qam_point_obstacle.py'
POINTS = ROOT / 'configs/reference_audit_points.json'
SMOKE = ROOT / 'runs/02-qam-point-obstacle-smoke-e6eff27267'
SPEC = importlib.util.spec_from_file_location('point_obstacle_audit_source', SOURCE)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

# Nested Sobol prefixes isolate each numerical axis.  The last two levels
# differ ONLY in transition quadrature; the previous steps isolate grid and
# policy expectation.  Q-gradient quadrature is varied separately below.
LEVELS = (
    ('L0', 41, 8, 128),
    ('G1', 61, 8, 128),
    ('P1', 61, 16, 128),
    ('T1', 61, 16, 256),
    ('G2', 81, 16, 256),
    ('P2', 81, 32, 256),
    ('T2', 81, 32, 512),
)
GRADIENT_SAMPLES = (2048, 4096, 8192)
MC_EPISODES_PER_SPLIT = 8192
FD_STEPS = (.04, .02, .01)
NONDEGENERATE_NORM = .03


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float)**2)))


def angle(x):
    return float(np.degrees(np.arctan2(x[1], x[0])))


def cosine(x, y):
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return float(np.dot(x, y) / (nx * ny)) if nx >= NONDEGENERATE_NORM and ny >= NONDEGENERATE_NORM else None


def point_arrays(points):
    k = np.array([v['k'] for v in points], dtype=int)
    p = np.array([[v['p_x'], v['p_y']] for v in points], dtype=float)
    a = np.array([[v['a_x'], v['a_y']] for v in points], dtype=float)
    return k, p, a


def cache_key(c, actor, base, source, point_bytes, level):
    return dict(actor_sha256=M.model_hash(actor), base_sha256=M.model_hash(base),
                source_sha256=M.sha(source), core_sha256=M.sha(M.CORE_PATH),
                config_sha256=M.sha(M.CONFIG_PATH), points_sha256=M.sha(point_bytes),
                level=level)


def get_table(output, task, actor, base, level, point_bytes):
    name, grid, policy, transition = level
    cache = output / 'tables' / f'{name}.npz'
    metadata = json.loads(json.dumps(cache_key(task.c, actor, base, SOURCE,
                                               point_bytes, level)))
    if cache.exists():
        with np.load(cache, allow_pickle=False) as archive:
            old = json.loads(str(archive['metadata'].item()))
            if old != metadata:
                raise RuntimeError(f'Cached table provenance changed: {cache}')
            values = archive['values'].copy()
            seconds = float(archive['seconds'].item())
        table = M.PolicyTable.__new__(M.PolicyTable)
        table.task, table.c, table.values = task, task.c, values
        table.actor_hash, table.base_hash = metadata['actor_sha256'], metadata['base_sha256']
        table.gradient_quad = M.normal_quadrature(GRADIENT_SAMPLES[0], seed=1994)
        table.grid, table.samples, table.transition_samples = grid, policy, transition
        table.seconds = seconds
        print(f'{name}: reused matching cached table ({seconds:.1f} s originally)', flush=True)
        return table
    start = time.monotonic()
    table = M.PolicyTable(task, actor, base, grid, policy, transition,
                          GRADIENT_SAMPLES[0])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, values=table.values,
                        metadata=json.dumps(metadata, sort_keys=True), seconds=table.seconds)
    print(f'{name}: built {grid}^2 / {policy} policy / {transition} transition '
          f'in {time.monotonic()-start:.1f} s', flush=True)
    return table


def table_at_points(table, k, p, a, gradient_samples):
    table.gradient_quad = M.normal_quadrature(gradient_samples, seed=1994)
    return table.q_gradient(k, p, a)


def plot_convergence(output, convergence):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for _, group in convergence.groupby('point_id'):
        group = group.set_index('level').loc[[x[0] for x in LEVELS]].reset_index()
        for ax, field in zip(axes, ('q', 'gradient_norm', 'gradient_angle_deg')):
            ax.plot(group['level'], group[field], marker='.', lw=.8, alpha=.55)
    for ax, title in zip(axes, ('Q', 'Gradient norm', 'Gradient direction (degrees)')):
        ax.set(xlabel='Nested numerical level', title=title)
        ax.grid(alpha=.2)
        ax.tick_params(axis='x', labelrotation=45)
    axes[2].set_ylim(-180, 180)
    fig.suptitle('Fixed Actor; all 17 predeclared points (direction undefined for near-zero gradients)')
    fig.tight_layout()
    fig.savefig(output/'reference_convergence.png', dpi=170)
    plt.close(fig)


def plot_arrows(output, points, critic, reference, mc):
    fig, axes = plt.subplots(4, 5, figsize=(16, 11))
    scale = max(.1, np.percentile(np.linalg.norm(np.vstack((critic, reference, mc)),axis=1), 90))
    for i, ax in enumerate(axes.flat):
        if i >= len(points):
            ax.axis('off'); continue
        for vec, color, label in ((critic[i], '#2878b5', 'Critic'),
                                  (reference[i], '#d27500', 'Table'),
                                  (mc[i], '#268348', 'MC')):
            ax.arrow(0, 0, vec[0], vec[1], color=color, width=.007*scale,
                     length_includes_head=True, alpha=.8, label=label)
        ax.set(xlim=(-1.6*scale,1.6*scale),ylim=(-1.6*scale,1.6*scale),
               aspect='equal',title=points[i]['id'],xlabel='action gradient x',ylabel='y')
        ax.axhline(0,color='0.8',lw=.5);ax.axvline(0,color='0.8',lw=.5)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Action-gradient arrows on the same fixed Actor and audit points')
    fig.tight_layout();fig.savefig(output/'critic_vs_reference_gradient.png',dpi=160);plt.close(fig)


def plot_gap(output, points, self_error, critic_gap):
    x=np.arange(len(points));w=.38
    fig,ax=plt.subplots(figsize=(13,4))
    ax.bar(x-w/2,self_error,w,label='Table high→higher drift')
    ax.bar(x+w/2,critic_gap,w,label='Critic–higher table gap')
    ax.set_yscale('log')
    ax.set_ylim(1e-6,max(3.,1.5*max(np.max(self_error),np.max(critic_gap))))
    ax.set(xticks=x,xticklabels=[p['id'] for p in points],ylabel='Euclidean action-gradient difference (log scale)',
           xlabel='Predeclared audit point')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(output/'reference_error_vs_critic_gap.png',dpi=170);plt.close(fig)


def run(output, source_run, seed, N):
    source_run=source_run.resolve();output=output.resolve()
    if output == source_run or source_run in output.parents:
        raise ValueError('Audit output must not overwrite the original smoke run')
    point_bytes=POINTS.read_bytes()
    point_protocol=json.loads(point_bytes)
    points=point_protocol['points']
    if not 12 <= len(points) <= 20 or len({p['id'] for p in points}) != len(points):
        raise ValueError('Expected 12–20 unique predeclared points')
    c,_=M.load_config();task=M.PointObstacle(c)
    sub=source_run/f'seed-{seed}'
    actor=M.load_actor(sub/f'learned-N{N}.pt',c)
    base=M.load_base(sub/'shared-initialization.pt',c)
    saved=torch.load(sub/f'learned-N{N}.pt',map_location='cpu',weights_only=True)
    critic=M.core.Critic(c);critic.load_state_dict(saved['critic']);critic.eval()
    output.mkdir(parents=True,exist_ok=True)
    audit=output/'reference-audit';audit.mkdir(exist_ok=True)
    copied=audit/'audit_points.json'
    if copied.exists() and copied.read_bytes()!=point_bytes:
        raise RuntimeError('Audit points changed on resume')
    if not copied.exists():copied.write_bytes(point_bytes)
    provenance=dict(source_run=str(source_run.relative_to(ROOT)),seed=seed,N=N,
                    points_sha256=M.sha(point_bytes),actor_sha256=M.model_hash(actor),
                    critic_sha256=M.model_hash(critic),base_sha256=M.model_hash(base),
                    config_sha256=M.sha(M.CONFIG_PATH),source_sha256=M.sha(SOURCE),
                    core_sha256=M.sha(M.CORE_PATH),audit_code_sha256=M.sha(Path(__file__)),
                    levels=LEVELS,gradient_samples=GRADIENT_SAMPLES,
                    mc_episodes_per_split=MC_EPISODES_PER_SPLIT,fd_steps=FD_STEPS)
    provenance=json.loads(json.dumps(provenance))
    prov_path=audit/'provenance.json'
    if prov_path.exists():
        old_provenance=json.loads(prov_path.read_text())
        changed={key for key in provenance if old_provenance.get(key)!=provenance[key]}
        if changed-{'audit_code_sha256'}:
            raise RuntimeError(f'Audit scientific provenance changed on resume: {changed}')
        if changed:
            history_path=audit/'runner-revision-history.json'
            history=json.loads(history_path.read_text()) if history_path.exists() else []
            revision=dict(previous_runner_sha256=old_provenance['audit_code_sha256'],
                          resumed_runner_sha256=provenance['audit_code_sha256'],
                          reason='Audit runner revisions after cached tables: cache metadata '
                                 'normalization, empty MC CSV handling, and absolute-error '
                                 'fallback for exactly zero MC standard error. PointObstacle, '
                                 'PolicyTable, actor, config, and table cache keys unchanged.')
            if revision not in history:
                history.append(revision)
                history_path.write_text(json.dumps(history,indent=2))
            print('Runner source changed; scientific hashes match; revision recorded',flush=True)
    else:
        prov_path.write_text(json.dumps(provenance,indent=2))
    k,p,a=point_arrays(points)
    with torch.enable_grad():
        ta=torch.tensor(a,dtype=torch.float32,requires_grad=True)
        cq=critic(torch.tensor(k,dtype=torch.long),torch.tensor(p,dtype=torch.float32),ta).mean(-1)
        gc=torch.autograd.grad(cq.sum(),ta)[0].detach().numpy()
    convergence=[];grads={};values={};tables={};table=None
    for level in LEVELS:
        name,grid,policy,transition=level
        table=get_table(audit,task,actor,base,level,point_bytes)
        tables[name]=table
        table.assert_current(actor,base)
        q,g=table_at_points(table,k,p,a,GRADIENT_SAMPLES[0])
        values[name]=q;grads[name]=g
        for i,point in enumerate(points):
            convergence.append(dict(point_id=point['id'],level=name,grid=grid,
                                    policy_samples=policy,transition_samples=transition,
                                    gradient_samples=GRADIENT_SAMPLES[0],q=q[i],
                                    grad_x=g[i,0],grad_y=g[i,1],gradient_norm=np.linalg.norm(g[i]),
                                    gradient_angle_deg=angle(g[i]) if np.linalg.norm(g[i])>=NONDEGENERATE_NORM else np.nan))
        pd.DataFrame(convergence).to_csv(audit/'convergence.csv',index=False)
    plot_convergence(audit,pd.DataFrame(convergence))
    grad_quad={}
    for samples in GRADIENT_SAMPLES:
        q,g=table_at_points(table,k,p,a,samples)
        grad_quad[samples]=g
    # In this run P2 is the "high" table and T2 the "higher" table; they
    # differ only in transition quadrature.  The final point gradients use
    # 8192 Gaussian Sobol nodes, with a separate 4096→8192 stability check.
    _,g_hi=table_at_points(tables['P2'],k,p,a,GRADIENT_SAMPLES[-1])
    g_higher=grad_quad[8192];q_higher=q
    mc_rows=[];mc_avg=[];mc_se=[];mc_q=[];mc_q_se=[];split_z=[]
    mc_partial=audit/'mc_crosscheck.csv'
    if mc_partial.exists():
        previous=pd.read_csv(mc_partial)
        if len(previous) and not set(previous['point_id']).issubset({v['id'] for v in points}):
            raise RuntimeError('MC cache contains different points')
    else:previous=pd.DataFrame()
    for i,point in enumerate(points):
        existing=previous[previous.point_id==point['id']] if len(previous) else pd.DataFrame()
        if len(existing)==2 and set(existing['split'])=={'A','B'}:
            runs=[dict(q=float(row.q_mc),q_se=float(row.q_mc_se),
                       gradient=[float(row.gradient_mc_x),float(row.gradient_mc_y)],
                       gradient_se=[float(row.gradient_mc_se_x),float(row.gradient_mc_se_y)])
                  for row in existing.itertuples()]
        else:
            runs=[]
            for split,base_seed in (('A',410000),('B',720000)):
                result=M.mc_fixed_q(task,actor,base,int(k[i]),p[i],a[i],MC_EPISODES_PER_SPLIT,
                                    base_seed+7919*i)
                runs.append(result)
                mc_rows.append(dict(point_id=point['id'],split=split,episodes=MC_EPISODES_PER_SPLIT,
                                    mc_seed=base_seed+7919*i,
                                    k=int(k[i]),p_x=p[i,0],p_y=p[i,1],a_x=a[i,0],a_y=a[i,1],
                                    q_mc=result['q'],q_mc_se=result['q_se'],
                                    gradient_mc_x=result['gradient'][0],gradient_mc_y=result['gradient'][1],
                                    gradient_mc_se_x=result['gradient_se'][0],
                                    gradient_mc_se_y=result['gradient_se'][1],
                                    gradient_split_difference=np.nan))
            combined=pd.concat([previous,pd.DataFrame(mc_rows)],ignore_index=True)
            combined.to_csv(mc_partial,index=False)
            print(f'MC {point["id"]}: split A/B complete ({i+1}/{len(points)})',flush=True)
        ga,gb=np.asarray(runs[0]['gradient']),np.asarray(runs[1]['gradient'])
        sa,sb=np.asarray(runs[0]['gradient_se']),np.asarray(runs[1]['gradient_se'])
        mc_avg.append((ga+gb)/2);mc_se.append(np.hypot(sa,sb)/2)
        mc_q.append((runs[0]['q']+runs[1]['q'])/2)
        mc_q_se.append(math.hypot(runs[0]['q_se'],runs[1]['q_se'])/2)
        split_z.append(np.linalg.norm((ga-gb)/np.maximum(np.hypot(sa,sb),1e-12)))
    mc_avg=np.asarray(mc_avg);mc_se=np.asarray(mc_se)
    mc_q=np.asarray(mc_q);mc_q_se=np.asarray(mc_q_se)
    # Complete the split-difference field in both rows, preserving split data.
    mc_frame=pd.read_csv(mc_partial)
    for i,point in enumerate(points):
        entries=mc_frame.index[mc_frame.point_id==point['id']]
        mc_frame.loc[(mc_frame.point_id==point['id'])&(mc_frame.split=='A'),'mc_seed']=410000+7919*i
        mc_frame.loc[(mc_frame.point_id==point['id'])&(mc_frame.split=='B'),'mc_seed']=720000+7919*i
        aa=mc_frame.loc[(mc_frame.point_id==point['id'])&(mc_frame.split=='A'),
                        ['gradient_mc_x','gradient_mc_y']].to_numpy()[0]
        bb=mc_frame.loc[(mc_frame.point_id==point['id'])&(mc_frame.split=='B'),
                        ['gradient_mc_x','gradient_mc_y']].to_numpy()[0]
        mc_frame.loc[entries,'gradient_split_difference']=np.linalg.norm(aa-bb)
    mc_frame['mc_seed']=mc_frame['mc_seed'].astype(int)
    mc_frame.to_csv(mc_partial,index=False)
    fd_rows=[];fd_last=np.zeros_like(g_higher)
    for h in FD_STEPS:
        fd=np.zeros_like(g_higher)
        for dim in range(2):
            da=np.zeros_like(a);da[:,dim]=h
            plus=table_at_points(table,k,p,a+da,GRADIENT_SAMPLES[-1])[0]
            minus=table_at_points(table,k,p,a-da,GRADIENT_SAMPLES[-1])[0]
            fd[:,dim]=(plus-minus)/(2*h)
        for i,point in enumerate(points):
            fd_rows.append(dict(point_id=point['id'],h=h,fd_x=fd[i,0],fd_y=fd[i,1],
                                table_x=g_higher[i,0],table_y=g_higher[i,1],
                                absolute_error=np.linalg.norm(fd[i]-g_higher[i]),
                                cosine=cosine(fd[i],g_higher[i])))
        fd_last=fd
    pd.DataFrame(fd_rows).to_csv(audit/'finite_difference.csv',index=False)
    rows=[]
    for i,point in enumerate(points):
        nondeg=np.linalg.norm(g_higher[i])>=NONDEGENERATE_NORM
        rows.append(dict(point_id=point['id'],geometry_category=point['geometry_category'],
                         k=int(k[i]),p_x=p[i,0],p_y=p[i,1],a_x=a[i,0],a_y=a[i,1],
                         nondegenerate=nondeg,near_zero_absolute_fallback=not nondeg,
                         critic_q=float(cq.detach().numpy()[i]),reference_q=q_higher[i],
                         mc_q=mc_q[i],mc_q_se=mc_q_se[i],
                         critic_grad_x=gc[i,0],critic_grad_y=gc[i,1],
                         high_grad_x=g_hi[i,0],high_grad_y=g_hi[i,1],
                         reference_grad_x=g_higher[i,0],reference_grad_y=g_higher[i,1],
                         mc_grad_x=mc_avg[i,0],mc_grad_y=mc_avg[i,1],
                         mc_se_x=mc_se[i,0],mc_se_y=mc_se[i,1],
                         fd_grad_x=fd_last[i,0],fd_grad_y=fd_last[i,1],
                         reference_self_error=np.linalg.norm(g_hi[i]-g_higher[i]),
                         critic_reference_gap=np.linalg.norm(gc[i]-g_higher[i]),
                         mc_reference_gap=np.linalg.norm(mc_avg[i]-g_higher[i]),
                         critic_reference_cosine=cosine(gc[i],g_higher[i]),
                         mc_reference_cosine=cosine(mc_avg[i],g_higher[i]),
                         mc_reference_angle_error_deg=(math.degrees(math.acos(np.clip(cosine(mc_avg[i],g_higher[i]),-1,1)))
                                                       if cosine(mc_avg[i],g_higher[i]) is not None else np.nan)))
    comparison=pd.DataFrame(rows);comparison.to_csv(audit/'gradient_comparison.csv',index=False)
    plot_arrows(audit,points,gc,g_higher,mc_avg)
    plot_gap(audit,points,np.linalg.norm(g_hi-g_higher,axis=1),np.linalg.norm(gc-g_higher,axis=1))
    nondeg=comparison['nondegenerate'].to_numpy(dtype=bool)
    if not np.any(nondeg):
        raise RuntimeError('No nondegenerate audit points; precision ratio undefined')
    e_ref=rms((g_hi-g_higher)[nondeg]);e_critic=rms((gc-g_higher)[nondeg])
    axis_drift=dict(grid=rms((grads['G2']-grads['T1'])[nondeg]),
                    policy=rms((grads['P2']-grads['G2'])[nondeg]),
                    transition=rms((grads['T2']-grads['P2'])[nondeg]),
                    gradient_quadrature=rms((grad_quad[8192]-grad_quad[4096])[nondeg]))
    # A deterministic zero-return geometry can give exactly zero empirical MC
    # SE.  A z-score there is undefined, not millions of standard errors.
    # Predeclared absolute fallbacks retain those points in the gate.
    q_zero=mc_q_se<1e-8
    gradient_zero=mc_se<1e-8
    q_abs=np.abs(q_higher-mc_q)
    gradient_abs=np.abs(g_higher-mc_avg)
    q_z=q_abs[~q_zero]/mc_q_se[~q_zero]
    gradient_z=gradient_abs[~gradient_zero]/mc_se[~gradient_zero]
    q_fallback_ok=bool(np.all(q_abs[q_zero]<=.001))
    gradient_fallback_ok=bool(np.all(gradient_abs[gradient_zero]<=.01))
    fd_error=fd_last-g_higher
    opposite=int(np.sum([(cosine(x,y) or 0)<-.2 for x,y in zip(g_higher[nondeg],mc_avg[nondeg])]))
    gates=dict(reference_to_critic_ratio=e_ref/(e_critic+1e-12)<=.25,
               all_axis_drift=max(axis_drift.values())/(e_critic+1e-12)<=.25,
               mc_q_consistency=q_fallback_ok and float(np.sqrt(np.mean(q_z**2)))<=2 and float(np.max(q_z))<=4,
               mc_gradient_consistency=gradient_fallback_ok and float(np.sqrt(np.mean(gradient_z**2)))<=2.5 and float(np.max(gradient_z))<=5,
               mc_gradient_precision=rms(mc_se)/(e_critic+1e-12)<=.25,
               finite_difference=rms(fd_error)<=.05 and float(np.max(np.abs(fd_error)))<=.15,
               direction=opposite<=1,
               convergence_trend=rms(grads['T2']-grads['P2'])<=rms(grads['P2']-grads['G2']))
    summary=dict(reference_validated=all(gates.values()),gates=gates,
                 seed=seed,N=N,audit_points=len(points),nondegenerate_points=int(nondeg.sum()),
                 near_zero_absolute_fallback_points=comparison.loc[~nondeg,'point_id'].tolist(),
                 zero_mc_q_se_absolute_fallback_points=[points[i]['id'] for i in np.flatnonzero(q_zero)],
                 zero_mc_gradient_se_absolute_fallback_components=[f'{points[i]["id"]}:{"xy"[j]}'
                   for i,j in zip(*np.where(gradient_zero))],
                 zero_mc_q_se_absolute_fallback_pass=q_fallback_ok,
                 zero_mc_gradient_se_absolute_fallback_pass=gradient_fallback_ok,
                 E_ref=e_ref,E_critic=e_critic,reference_to_critic_error_ratio=e_ref/(e_critic+1e-12),
                 axis_drift=axis_drift,all_axis_drift_ratio=max(axis_drift.values())/(e_critic+1e-12),
                 mc_gradient_se_rms=rms(mc_se),mc_gradient_se_to_critic_ratio=rms(mc_se)/(e_critic+1e-12),
                 mc_split_difference_rms=rms(mc_frame.groupby('point_id').gradient_split_difference.first()),
                 mc_split_z_max=float(np.max(split_z)),mc_q_z_rms=rms(q_z),mc_q_z_max=float(np.max(q_z)),
                 mc_gradient_z_rms=rms(gradient_z),mc_gradient_z_max=float(np.max(gradient_z)),
                 fd_error_rms=rms(fd_error),fd_error_max=float(np.max(np.abs(fd_error))),
                 opposite_direction_points=opposite,
                 high_level='P2: 81^2/32 policy/256 transition',
                 higher_level='T2: 81^2/32 policy/512 transition',
                 gradient_quadrature_final=8192,
                 protocol_note='Gate applies to frozen Ordinary N=0 Actor only; passing it would not automatically certify every formal Actor update.')
    (audit/'precision_summary.json').write_text(json.dumps(summary,indent=2))
    status='PASS' if summary['reference_validated'] else 'FAIL'
    report=f'''# Current-policy Q-gradient reference precision audit — {status}

The 17 geometry-based audit points were fixed in `configs/reference_audit_points.json`
at commit `af127527efb7ce787f7f6c2e7c66716fb6efc8d1`, before this audit.
This run freezes the **ordinary QAM Actor/Critic at seed {seed}, N={N}**.
The numerical table evaluates its full current ODE deployment sampler; independent
Monte Carlo uses real simulator transitions and two independent {MC_EPISODES_PER_SPLIT}-rollout
streams per point, without reading Critic or table Q. The finite differences use
shared deterministic QMC nodes at h={FD_STEPS}.

## Predeclared numerical study

Nested levels are {', '.join(x[0]+':'+str(x[1:]) for x in LEVELS)}. `G`, `P`, and
`T` steps separately change grid, policy samples, and transition samples.
Pointwise action-gradient quadrature uses {GRADIENT_SAMPLES} Sobol nodes. The
highest two *table* levels P2→T2 differ only in transition integration.

On the {int(nondeg.sum())} nondegenerate points, E_ref = RMS(g_P2−g_T2) = **{e_ref:.6g}**;
E_critic = RMS(g_critic−g_T2) =
**{e_critic:.6g}**; ratio = **{summary['reference_to_critic_error_ratio']:.4g}**
(required ≤ 0.25). This ratio alone cannot prove reference accuracy: the
grid/policy and independent MC/FD checks are additional mandatory gates.
Last-axis drifts: {axis_drift}. Max-axis/critic ratio =
{summary['all_axis_drift_ratio']:.4g}. Near-zero gradients use absolute error;
the affected point IDs are {summary['near_zero_absolute_fallback_points']}.

Independent MC Q z RMS/max (positive-SE points only) = {summary['mc_q_z_rms']:.3g}/
{summary['mc_q_z_max']:.3g}; gradient z RMS/max =
{summary['mc_gradient_z_rms']:.3g}/{summary['mc_gradient_z_max']:.3g}.
Zero-SE MC entries use fixed absolute fallbacks (Q ≤ 0.001, gradient
component ≤ 0.01), not meaningless relative/z errors: Q points
{summary['zero_mc_q_se_absolute_fallback_points']}, gradient components
{summary['zero_mc_gradient_se_absolute_fallback_components']}.
MC gradient SE RMS / critic gap = {summary['mc_gradient_se_to_critic_ratio']:.3g};
the two MC splits' gradient differences are stored per point. The h=0.01
finite-difference gradient RMS/max-coordinate error is
{summary['fd_error_rms']:.3g}/{summary['fd_error_max']:.3g}.
Opposite table-vs-MC directions at nondegenerate points: {opposite}.

Automatic gate results: `{json.dumps(gates,sort_keys=True)}`.

## Decision

{'The frozen N=0 reference passed this audit, but every updated Actor still needs validation before a formal claim.' if summary['reference_validated'] else 'Reference precision failed. Formal Oracle-gap experiment not executed.'}

No formal 3-seed training, 50-pair trajectory test, or causal Q-gradient
performance-gap conclusion is reported here. Those conditional downstream
steps remain blocked unless the reference is certified for updated policies.
The original single-seed smoke and all earlier research artifacts are untouched.
'''
    (audit/'precision_report.md').write_text(report)
    (output/'report.md').write_text(report)
    print(f'PRECISION GATE: {status}; ratio={summary["reference_to_critic_error_ratio"]:.4g}; '
          f'failed={[k for k,v in gates.items() if not v]}',flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-run',type=Path,default=SMOKE)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=59000)
    parser.add_argument('--N',type=int,default=0)
    args=parser.parse_args()
    run(args.output,args.source_run,args.seed,args.N)
