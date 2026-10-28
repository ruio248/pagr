"""Measure extra analytic-Oracle label cost separately from training outcomes."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('qam_notebook',ROOT/'notebooks/02_qam_td_terminal_error.py')
N=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=N
SPEC.loader.exec_module(N)


def benchmark(run,repeats=100):
    run=Path(run).resolve()
    c,_=N.load_config();task=N.SparseReachTask(c)
    rows=[]
    for seed in json.loads((run/'provenance.json').read_text())['seeds']:
        sub=run/f'seed-{seed}'
        saved=torch.load(sub/'critic-A.pt',map_location='cpu',weights_only=True)
        critic=N.Critic(c);critic.load_state_dict(saved['critic'])
        data=np.load(sub/'valid-transitions.npz')
        k=torch.from_numpy(data['k'][:64]);y=torch.from_numpy(data['y'][:64]);a=torch.from_numpy(data['a'][:64])
        # Warm both paths before timing, same endpoints, same dtype.
        N.terminal_label(task,critic,c,k,y,a,'learned')
        N.terminal_label(task,critic,c,k,y,a,'oracle')
        for source in ('learned','oracle'):
            start=time.perf_counter()
            for _ in range(repeats):N.terminal_label(task,critic,c,k,y,a,source)
            elapsed=(time.perf_counter()-start)/repeats
            rows.append(dict(seed=seed,source=source,batch=64,repeats=repeats,
                             seconds_per_label_batch=elapsed))
    result={'scope':'terminal-label generation only; excludes critic training, actor rollouts and other compute',
            'hardware':'local CPU, not a method-matched wall-clock study','rows':rows}
    target=run/'terminal-label-compute-cost.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    args=parser.parse_args()
    print(json.dumps(benchmark(args.run),indent=2))
