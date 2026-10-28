"""Read-only structural and provenance audit for Notebook 2b's diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(output):
    output=Path(output).resolve()
    manifest=json.loads((output/'provenance.json').read_text())
    design=json.loads((output/'config-effective.json').read_text())
    source=ROOT/manifest['source_run']
    source_config=json.loads((source/'config-effective.json').read_text())
    dimension=source_config['dimension']
    flow_steps=source_config['flow_steps']
    assert manifest['source_run_provenance_sha256']==sha(source/'provenance.json')
    assert manifest['source_script_sha256']==sha(
        ROOT/'notebooks/02b_joint_mc_reference_diagnostic.py')
    assert manifest['core_script_sha256']==sha(ROOT/'notebooks/02_qam_td_terminal_error.py')
    assert manifest['config_sha256']==sha(ROOT/'configs/qam_joint_mc_diagnostic.json')
    assert design==json.loads((ROOT/'configs/qam_joint_mc_diagnostic.json').read_text())
    points=pd.read_csv(output/'pointwise.csv')
    snapshots=pd.read_csv(output/'snapshots.csv')
    expected={(seed,n,k) for seed in design['seeds']
              for n in design['actor_updates'] for k in design['environment_times']}
    assert set(map(tuple,points[['seed','N','k']].to_numpy()))==expected
    assert len(points)==len(expected)
    expected_snapshots={(seed,n) for seed in design['seeds'] for n in design['actor_updates']}
    assert set(map(tuple,snapshots[['seed','N']].to_numpy()))==expected_snapshots
    assert len(snapshots)==len(expected_snapshots)
    assert (points.mc_samples==2*design['mc_samples_per_split']).all()
    assert (snapshots.eval_episodes==design['paired_eval_episodes']).all()
    assert (snapshots.recurrence_max_error<3e-5).all()
    assert ((points.q_mc>=0)&(points.q_mc<=1)).all()
    assert ((points.q_mc_se>=0)&(points.gradient_mc_se_l2>=0)).all()
    for field in ('gradient_critic_minus_mc_norm','gradient_mc_split_difference_norm',
                  'gradient_mc_se_l2'):
        assert np.isfinite(points[field]).all(),field
    assert points.loc[points.k<max(design['environment_times']),
                      'terminal_exact_q_calibration'].isna().all()
    assert points.loc[points.k==max(design['environment_times']),
                      'terminal_exact_q_calibration'].notna().all()
    for row in snapshots.itertuples():
        checkpoint=ROOT/row.checkpoint
        assert sha(checkpoint)==manifest['checkpoints'][f'{row.seed}/N{row.N}']
        arrays=np.load(output/row.mechanism_npz)
        assert arrays['k'].shape==(row.states,)
        assert arrays['reference_gradient'].shape==(row.states,dimension)
        assert arrays['adjoint_difference'].shape[1]==flow_steps+1
        assert arrays['matching_label_difference'].shape[1]==flow_steps
        assert np.allclose(arrays['reference_gradient'],
                           .5*(arrays['reference_split1']+arrays['reference_split2']))
        for key in arrays.files:
            assert np.isfinite(arrays[key]).all(),(row.seed,row.N,key)
    for file in ('report.md','joint-mc-diagnostic.png'):
        assert (output/file).is_file()
    return dict(passed=True,source_run=manifest['source_run'],
                seeds=design['seeds'],snapshots=len(snapshots),points=len(points),
                mc_rollouts_per_point=2*design['mc_samples_per_split'],
                paired_eval_episodes_per_snapshot=design['paired_eval_episodes'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--write',action='store_true',help='write validation.json in output')
    args=parser.parse_args()
    result=audit(args.output)
    if args.write:
        path=args.output/'validation.json'
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
