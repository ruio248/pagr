"""Read-only completeness and fail-closed checks for a reference audit run."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('point_reference_validator_audit_source',
                                            ROOT/'scripts/audit_point_reference.py')
A=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=A
SPEC.loader.exec_module(A)


def validate(run):
    audit=Path(run).resolve()/'reference-audit'
    points_bytes=(audit/'audit_points.json').read_bytes()
    assert points_bytes==A.POINTS.read_bytes(), 'predeclared point list changed'
    points=json.loads(points_bytes)['points'];ids={p['id'] for p in points}
    assert len(ids)==len(points)==17
    provenance=json.loads((audit/'provenance.json').read_text())
    assert provenance['points_sha256']==A.M.sha(points_bytes)
    for key,path in [('source_sha256',A.SOURCE),('core_sha256',A.M.CORE_PATH),
                     ('config_sha256',A.M.CONFIG_PATH)]:
        assert provenance[key]==A.M.sha(path),f'{key} differs from audited source'
    history_path=audit/'runner-revision-history.json'
    runner_hash=A.M.sha(Path(A.__file__))
    history=json.loads(history_path.read_text()) if history_path.exists() else []
    allowed={provenance['audit_code_sha256']}
    allowed.update(entry['resumed_runner_sha256'] for entry in history)
    assert runner_hash in allowed,'runner revision missing from audit history'
    convergence=pd.read_csv(audit/'convergence.csv')
    expected_levels={x[0] for x in A.LEVELS}
    assert set(convergence.point_id)==ids
    assert set(convergence.level)==expected_levels
    assert len(convergence)==len(ids)*len(A.LEVELS)
    assert not convergence[['point_id','level']].duplicated().any()
    for name,grid,policy,transition in A.LEVELS:
        with np.load(audit/'tables'/f'{name}.npz',allow_pickle=False) as cache:
            assert cache['values'].shape==(28+1,grid,grid)
            metadata=json.loads(str(cache['metadata'].item()))
            assert metadata['actor_sha256']==provenance['actor_sha256']
            assert metadata['source_sha256']==provenance['source_sha256']
            assert metadata['level']==[name,grid,policy,transition]
    mc=pd.read_csv(audit/'mc_crosscheck.csv')
    assert len(mc)==len(ids)*2
    assert not mc[['point_id','split']].duplicated().any()
    assert set(mc.point_id)==ids and set(mc.split)=={'A','B'}
    assert (mc.episodes==A.MC_EPISODES_PER_SPLIT).all()
    assert (mc[['q_mc_se','gradient_mc_se_x','gradient_mc_se_y']]>=0).all().all()
    for i,point in enumerate(points):
        two=mc[mc.point_id==point['id']].set_index('split')
        assert int(two.loc['A','mc_seed'])==410000+7919*i
        assert int(two.loc['B','mc_seed'])==720000+7919*i
        aa=two.loc['A',['gradient_mc_x','gradient_mc_y']].to_numpy(dtype=float)
        bb=two.loc['B',['gradient_mc_x','gradient_mc_y']].to_numpy(dtype=float)
        assert np.allclose(two.gradient_split_difference,np.linalg.norm(aa-bb),atol=1e-12)
    fd=pd.read_csv(audit/'finite_difference.csv')
    assert len(fd)==len(ids)*len(A.FD_STEPS)
    assert set(fd.point_id)==ids and set(fd.h)==set(A.FD_STEPS)
    assert not fd[['point_id','h']].duplicated().any()
    comparison=pd.read_csv(audit/'gradient_comparison.csv')
    assert len(comparison)==len(ids) and set(comparison.point_id)==ids
    assert not comparison.point_id.duplicated().any()
    summary=json.loads((audit/'precision_summary.json').read_text())
    ordered=comparison.set_index('point_id').loc[[p['id'] for p in points]]
    nondeg=ordered.nondegenerate.to_numpy(dtype=bool)
    hi=ordered[['high_grad_x','high_grad_y']].to_numpy()
    higher=ordered[['reference_grad_x','reference_grad_y']].to_numpy()
    critic=ordered[['critic_grad_x','critic_grad_y']].to_numpy()
    e_ref=A.rms((hi-higher)[nondeg]);e_critic=A.rms((critic-higher)[nondeg])
    assert np.isclose(summary['E_ref'],e_ref,rtol=1e-10)
    assert np.isclose(summary['E_critic'],e_critic,rtol=1e-10)
    assert np.isclose(summary['reference_to_critic_error_ratio'],e_ref/(e_critic+1e-12),rtol=1e-10)
    assert summary['reference_validated']==all(summary['gates'].values())
    if not summary['reference_validated']:
        assert 'Formal Oracle-gap experiment not executed' in (run/'report.md').read_text()
    for name in ('reference_convergence.png','critic_vs_reference_gradient.png',
                 'reference_error_vs_critic_gap.png'):
        assert (audit/name).stat().st_size>10_000,name
    followup_names=('followup-g81-p64-t256','followup-g121-p32-t256')
    found=sum((audit/f'{name}.json').exists() for name in followup_names)
    assert found in (0,2),'partial post-failure followup set'
    if found:
        for name in followup_names:
            followup=json.loads((audit/f'{name}.json').read_text())
            followup_rows=pd.read_csv(audit/f'{name}.csv')
            assert followup['actor_sha256']==provenance['actor_sha256']
            assert followup['status'].startswith('post-failure')
            assert set(followup_rows.point_id)==ids and len(followup_rows)==len(ids)
            assert not followup_rows.point_id.duplicated().any()
    return dict(passed=True,reference_validated=summary['reference_validated'],
                audit_points=len(ids),levels=len(A.LEVELS),mc_split_rows=len(mc),
                finite_difference_rows=len(fd),
                post_failure_followups=found,
                failed_gates=[k for k,v in summary['gates'].items() if not v])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    print(json.dumps(validate(args.run),indent=2))
