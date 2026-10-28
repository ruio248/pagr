"""Read-only structural/numerical audit of a completed TD/QAM run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT=Path(__file__).resolve().parents[1]
OLD={
    'notebooks/02_flow_am_dimension_coupling.py': '8f2d57e4cdc604ae8b145e819f86ef82a346edb468498f231b226f9b4094fb03',
    'notebooks/02_flow_am_dimension_coupling.ipynb': '4e4a9a2bd65b7ff21f720df495ce46c79d5058ae0e9e452880d0becf4a4dd508',
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(run):
    run=Path(run).resolve()
    assert run.is_dir(),run
    provenance=json.loads((run/'provenance.json').read_text())
    config=json.loads((run/'config-effective.json').read_text())
    assert provenance['source_sha256']==sha(ROOT/'notebooks/02_qam_td_terminal_error.py')
    assert provenance['config_sha256']==sha(ROOT/'configs/qam_td_terminal_error.json')
    for name,digest in OLD.items():assert sha(ROOT/name)==digest,name
    seeds=provenance['seeds'];formal=provenance['profile']=='formal'
    if formal:assert seeds==[49001,49002,49003]
    N=set(config['curve_updates']);K=config['flow_steps'];d=config['dimension'];H=config['horizon']
    assert K==64 and d==16 and H==8 if formal else True
    counts={}
    for seed in seeds:
        sub=run/f'seed-{seed}'
        data=np.load(sub/'train-transitions.npz')
        valid=np.load(sub/'valid-transitions.npz')
        eval_data=np.load(sub/'eval-transitions.npz')
        for label,arr in [('train',data),('valid',valid),('eval',eval_data)]:
            k=arr['k'];r=arr['r'];done=arr['done'];kn=arr['kn']
            assert (kn==k+1).all() and (done==(k==H-1)).all(),(seed,label)
            assert (r[k<H-1]==0).all(),(seed,label)
            assert len(k)%H==0 and set(k)==set(range(H)),(seed,label)
        assert len(data['k'])==config['train_episodes']*H
        assert len(valid['k'])==config['valid_episodes']*H
        count=int((data['r']>0).sum());counts[str(seed)]=count
        check=json.loads((sub/'future-value-check.json').read_text())
        assert check['immediate_reward']==0 and check['Q_action_perturb_mean_abs_change']>0
        assert (sub/'base.pt').is_file() and (sub/'critic-A.pt').is_file()
        a=pd.read_csv(sub/'experiment-A-curves.csv')
        b=pd.read_csv(sub/'experiment-B-curves.csv')
        m=pd.read_csv(sub/'experiment-A-mechanism.csv')
        assert set(a.N)==N and set(a.method)=={'learned','oracle'}
        assert set(b.N)==N and set(b.method)=={'joint_learned'}
        assert len(a)==2*len(N) and len(b)==len(N)
        at_zero=a[a.N==0].set_index('method')
        for metric in ('extract_J','extract_KL','extract_Q_ref','success_rate','discounted_return'):
            assert at_zero.loc['learned',metric]==at_zero.loc['oracle',metric],(seed,metric)
        assert set(m.N)==set(config['diagnostic_updates'])
        for n in N:
            for method in ('learned','oracle'):
                checkpoint=sub/f'actor-A-{method}-N{n}.pt'
                assert checkpoint.is_file()
                saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                steps=[float(item['step']) for item in saved['optimizer']['state'].values()]
                assert (not steps if n==0 else steps and all(x==n for x in steps)),(seed,method,n)
            checkpoint=sub/f'joint-B-N{n}.pt'
            assert checkpoint.is_file()
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            for key in ('actor_optimizer','critic_optimizer'):
                steps=[float(item['step']) for item in saved[key]['state'].values()]
                assert (not steps if n==0 else steps and all(x==n for x in steps)),(seed,key,n)
        for row in m.itertuples():
            assert row.recurrence_max_error<3e-5 and row.parameter_projection_max_error<3e-5
            arr=np.load(sub/row.diagnostic_npz)
            assert arr['adjoint_error'].shape[1]==K+1
            assert arr['matching_label_error'].shape[1]==K
            assert np.isfinite(arr['matching_parameter_gradient_error']).all()
            assert np.isfinite(arr['adam_step_error']).all()
        print('audited',seed,'successes',count,flush=True)
    for filename in ('experiment-A-curves.csv','experiment-B-curves.csv',
                     'experiment-A-mechanism.csv','critic-A-diagnostics.csv','data-summary.csv',
                     'final-independent-evaluation.csv','oracle-checks.csv','report.md',
                     'fixed-N-and-propagation.png'):
        assert (run/filename).is_file(),filename
    final=pd.read_csv(run/'final-independent-evaluation.csv')
    assert set(final.seed)==set(seeds)
    assert set(final.method)=={'A-learned','A-oracle','B-joint'}
    assert (final.eval_episodes==config['eval_episodes_final']).all()
    result={'passed':True,'profile':provenance['profile'],'seeds':seeds,
            'train_successes':counts,'fixed_N':sorted(N),'flow_steps':K,
            'checks':'source/config hashes, historical bytes, transition semantics, paired N0, exact optimizer counts, per-seed checkpoints/curves/mechanism, final evaluation'}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=audit(args.run)
    if args.output:
        if args.output.exists():raise FileExistsError(args.output)
        args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
