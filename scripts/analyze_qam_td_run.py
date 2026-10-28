"""Read saved common-path diagnostics; no retraining or checkpoint selection."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze(run):
    run=Path(run).resolve()
    provenance=json.loads((run/'provenance.json').read_text())
    rows=[]
    for seed in provenance['seeds']:
        for file in sorted((run/f'seed-{seed}').glob('mechanism-seed*-N*.npz')):
            n=int(file.stem.split('-N')[-1])
            data=np.load(file)
            err=data['terminal_gradient_error'].astype(np.float64)
            ref=data['terminal_gradient_ref'].astype(np.float64)
            refnorm=np.linalg.norm(ref,axis=-1)
            unit=ref/np.maximum(refnorm[:,None],1e-12)
            parallel=(err*unit).sum(-1)
            orthogonal=err-parallel[:,None]*unit
            for j in range(err.shape[-1]):
                rows.append(dict(seed=seed,N=n,dimension=j,
                                 gradient_error_rms=float(np.sqrt(np.mean(err[:,j]**2))),
                                 oracle_gradient_rms=float(np.sqrt(np.mean(ref[:,j]**2))),
                                 parallel_error_rms=float(np.sqrt(np.mean(parallel**2))),
                                 orthogonal_error_rms=float(np.sqrt(np.mean(orthogonal**2))),
                                 q_value_error_rms=float(np.sqrt(np.mean(data['q_value_error']**2)))))
    result=pd.DataFrame(rows)
    result.to_csv(run/'gradient-dimension-direction-diagnostics.csv',index=False)
    end=result[result.N==result.N.max()]
    groups=end.groupby('dimension').gradient_error_rms.agg(['mean','std'])
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(groups.index,groups['mean'],yerr=groups['std'].fillna(0),capsize=2)
    ax.set(xlabel='action dimension',ylabel='terminal Q-gradient error RMS',
           title='A: learned TD critic vs exact fixed-policy Q gradient, final N')
    ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(run/'terminal-gradient-per-dimension.png',dpi=160);plt.close(fig)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    args=parser.parse_args()
    table=analyze(args.run)
    print(table.groupby(['N'])[['parallel_error_rms','orthogonal_error_rms']].mean().to_string())
