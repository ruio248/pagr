"""Post-gate, fixed-point numerical-sensitivity probe; never grants PASS.

The primary audit points and thresholds are frozen.  This separate follow-up
only diagnoses whether a single-axis increase in grid or policy-action Sobol
samples reduces its failed independent-MC Q discrepancies.  It cannot alter
precision_summary.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('point_reference_audit_probe_source',
                                            ROOT/'scripts/audit_point_reference.py')
A=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=A
SPEC.loader.exec_module(A)


def run(output,policy_samples,transition_samples,grid=81):
    audit=Path(output).resolve()/'reference-audit'
    primary=json.loads((audit/'precision_summary.json').read_text())
    if primary['reference_validated']:
        raise RuntimeError('This post-failure probe is not needed for a passing audit')
    provenance=json.loads((audit/'provenance.json').read_text())
    point_bytes=(audit/'audit_points.json').read_bytes()
    if A.M.sha(point_bytes)!=provenance['points_sha256']:
        raise RuntimeError('Fixed audit points changed')
    if A.M.sha(A.SOURCE)!=provenance['source_sha256']:
        raise RuntimeError('Numerical source changed since audit')
    if A.M.sha(A.M.CONFIG_PATH)!=provenance['config_sha256']:
        raise RuntimeError('Task configuration changed since audit')
    c,_=A.M.load_config();task=A.M.PointObstacle(c)
    sub=ROOT/provenance['source_run']/f'seed-{provenance["seed"]}'
    actor=A.M.load_actor(sub/f'learned-N{provenance["N"]}.pt',c)
    base=A.M.load_base(sub/'shared-initialization.pt',c)
    if A.M.model_hash(actor)!=provenance['actor_sha256']:
        raise RuntimeError('Actor changed')
    points=json.loads(point_bytes)['points'];k,p,a=A.point_arrays(points)
    name=f'followup-g{grid}-p{policy_samples}-t{transition_samples}'
    table=A.get_table(audit,task,actor,base,(name,grid,policy_samples,transition_samples),point_bytes)
    q,g=A.table_at_points(table,k,p,a,A.GRADIENT_SAMPLES[-1])
    old=pd.read_csv(audit/'gradient_comparison.csv').set_index('point_id').loc[[v['id'] for v in points]]
    previous=(pd.read_csv(audit/'convergence.csv').query('level == "P2"')
              .set_index('point_id').loc[[v['id'] for v in points]])
    q_previous=previous.q.to_numpy()
    g_previous=previous[['grad_x','grad_y']].to_numpy()
    q_mc=old.mc_q.to_numpy();q_se=old.mc_q_se.to_numpy()
    mc_g=old[['mc_grad_x','mc_grad_y']].to_numpy()
    mc_g_se=old[['mc_se_x','mc_se_y']].to_numpy()
    critic_g=old[['critic_grad_x','critic_grad_y']].to_numpy()
    valid=q_se>=1e-8
    z_old=np.abs(q_previous[valid]-q_mc[valid])/q_se[valid]
    z_new=np.abs(q[valid]-q_mc[valid])/q_se[valid]
    grad_valid=mc_g_se>=1e-8
    grad_z=np.abs(g-mc_g)[grad_valid]/mc_g_se[grad_valid]
    nondeg=np.linalg.norm(g,axis=1)>=A.NONDEGENERATE_NORM
    fd=np.zeros_like(g);h=.01
    for dim in range(2):
        da=np.zeros_like(a);da[:,dim]=h
        plus=A.table_at_points(table,k,p,a+da,A.GRADIENT_SAMPLES[-1])[0]
        minus=A.table_at_points(table,k,p,a-da,A.GRADIENT_SAMPLES[-1])[0]
        fd[:,dim]=(plus-minus)/(2*h)
    e_ref=A.rms((g-g_previous)[nondeg])
    e_critic=A.rms((g-critic_g)[nondeg])
    frame=pd.DataFrame(dict(point_id=[v['id'] for v in points],
                            q_new=q, q_previous=q_previous,
                            q_mc=q_mc,q_mc_se=q_se,
                            q_delta=q-q_previous,
                            q_mc_z=np.where(valid,np.abs(q-q_mc)/np.maximum(q_se,1e-12),np.nan),
                            grad_x_new=g[:,0],grad_y_new=g[:,1],
                            grad_delta_norm=np.linalg.norm(g-g_previous,axis=1),
                            grad_mc_z_x=np.abs(g[:,0]-mc_g[:,0])/np.maximum(mc_g_se[:,0],1e-12),
                            grad_mc_z_y=np.abs(g[:,1]-mc_g[:,1])/np.maximum(mc_g_se[:,1],1e-12),
                            fd_x=fd[:,0],fd_y=fd[:,1]))
    frame.to_csv(audit/f'{name}.csv',index=False)
    result=dict(status='post-failure sensitivity only; original gate unchanged',
                name=name,grid=grid,policy_samples=policy_samples,
                transition_samples=transition_samples,actor_sha256=provenance['actor_sha256'],
                seconds=table.seconds,
                q_abs_mc_error_mean_before=float(np.mean(np.abs(q_previous[valid]-q_mc[valid]))),
                q_abs_mc_error_mean_after=float(np.mean(np.abs(q[valid]-q_mc[valid]))),
                q_mc_z_rms_before=A.rms(z_old),q_mc_z_rms_after=A.rms(z_new),
                q_mc_z_max_before=float(np.max(z_old)),q_mc_z_max_after=float(np.max(z_new)),
                q_drift_rms=A.rms(q-q_previous),
                gradient_drift_rms=A.rms(g-g_previous),
                gradient_mc_z_rms=A.rms(grad_z),gradient_mc_z_max=float(np.max(grad_z)),
                finite_difference_error_rms=A.rms(fd-g),
                E_ref_nondegenerate=e_ref,E_critic_nondegenerate=e_critic,
                self_to_critic_ratio=e_ref/(e_critic+1e-12),
                caveat='Post-failure settings chosen after seeing initial audit; no formal precision PASS or formal Actor-update certification.')
    (audit/f'{name}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grid',type=int,default=81)
    parser.add_argument('--policy-samples',type=int,default=64)
    parser.add_argument('--transition-samples',type=int,default=256)
    args=parser.parse_args()
    run(args.output,args.policy_samples,args.transition_samples,args.grid)
