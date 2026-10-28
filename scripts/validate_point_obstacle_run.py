"""Audit a point-obstacle run without retraining or claiming reference accuracy."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def validate(path):
    path=Path(path)
    c=json.loads((path/'config-effective.json').read_text())
    provenance=json.loads((path/'provenance.json').read_text())
    curve=pd.read_csv(path/'curves.csv')
    final=pd.read_csv(path/'final-independent.csv')
    seeds=provenance['seeds']
    methods=('learned','reference')
    updates=tuple(c.get('smoke_overrides',{}).get('curve_updates',c['curve_updates']))
    N=int(c.get('smoke_overrides',{}).get('actor_updates',c['actor_updates']))
    curve_episodes=int(c.get('smoke_overrides',{}).get('curve_episodes',c['curve_episodes']))
    final_episodes=int(c.get('smoke_overrides',{}).get('final_episodes',c['final_episodes']))
    assert set(curve.method)==set(methods)
    assert len(curve)==len(seeds)*len(methods)*len(updates)
    assert len(final)==len(seeds)*len(methods)
    for seed in seeds:
        sub=path/f'seed-{seed}'
        summary=json.loads((sub/'data-summary.json').read_text())
        for split in ('train','valid'):
            data=np.load(sub/f'{split}-transitions.npz')
            assert len(data['k'])==summary[split]['transitions']
            assert np.array_equal(data['kn'],data['k']+1)
            assert len(np.unique(data['episode_id']))==summary[split]['episodes']
            assert np.all((data['k']>=0)&(data['k']<c['horizon']))
            assert np.array_equal(data['done'],(data['r']!=0)|(data['k']==c['horizon']-1))
        for method in methods:
            rows=curve[(curve.seed==seed)&(curve.method==method)].sort_values('N')
            assert tuple(rows.N)==updates
            assert set(rows.critic_updates)==set(updates)
            assert set(rows.episodes)=={curve_episodes}
            assert set(final[(final.seed==seed)&(final.method==method)].episodes)=={final_episodes}
            for _,row in rows.iterrows():
                step=int(row.N)
                saved=torch.load(sub/f'{method}-N{step}.pt',map_location='cpu',weights_only=True)
                assert saved['N']==step and saved['method']==method
                for optimizer_name in ('actor_optimizer','critic_optimizer'):
                    states=saved[optimizer_name]['state'].values()
                    if step==0:
                        assert len(list(states))==0
                    else:
                        assert all(int(state['step'])==step for state in states)
                assert abs(row.success_rate+row.collision_rate+row.timeout_rate-1)<1e-10
        initial=[torch.load(sub/f'{method}-N0.pt',map_location='cpu',weights_only=True)
                 for method in methods]
        for key in ('actor','critic','target'):
            assert all(torch.equal(initial[0][key][name],initial[1][key][name])
                       for name in initial[0][key])
    for name in ('learning-curves.png','showcase-trajectories.png','showcase.gif',
                 'mechanism-smoke.json','mechanism-smoke.png','report.md'):
        assert (path/name).is_file() and (path/name).stat().st_size>0
    mechanism=json.loads((path/'mechanism-smoke.json').read_text())
    assert mechanism['reference_validated'] is False
    return dict(passed=True,profile='smoke only',seeds=seeds,N=N,
                reference_precision_passed=False,formal_claim_allowed=False,
                curve_episodes=curve_episodes,final_episodes=final_episodes)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    print(json.dumps(validate(args.run),indent=2))
