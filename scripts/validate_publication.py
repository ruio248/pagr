"""Collect verifiable evidence from local replay/prepare/notebook runs.

Run after the documented two replay suites and one arm/18001 prepare. This
does not launch or extend scientific experiments; it checks existing outputs.
"""
import argparse
import json
from pathlib import Path
import platform

import nbformat
import numpy as np
import scipy
import torch
from pagr import runtime as rt


def exact_nested(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys()==right.keys() and all(exact_nested(left[k],right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left)==len(right) and all(exact_nested(a,b) for a,b in zip(left,right))
    return left==right


def collect(root):
    suites=[json.loads((root/'runs'/f'public_replay_{task}'/'suite_comparison.json').read_text()) for task in rt.TASKS]
    assert sum(s['runs'] for s in suites)==42
    assert sum(s['exact_runs'] for s in suites)==42
    same_actors=0
    for task in rt.TASKS:
        for seed in rt.SEEDS:
            for method in rt.METHODS:
                relative=Path(task)/f'seed_{seed}'/method/'actor.pt'
                replay=torch.load(root/'runs'/f'public_replay_{task}'/relative,weights_only=True)
                reference=torch.load(rt.ASSETS/'reference/followup'/relative,weights_only=True)
                assert exact_nested(replay,reference)
                same_actors+=1
    generated=root/'runs/regenerated'
    original=Path('original/pilot/arm/seed_18001/critics')
    a=np.load(generated/original/'data.npz',allow_pickle=False)
    b=np.load(rt.ASSETS/original/'data.npz',allow_pickle=False)
    assert set(a.files)==set(b.files)
    assert all(np.array_equal(a[k],b[k]) for k in a.files)
    paths=[original/f'critic_{m}.pt' for m in range(5)]
    paths += [Path('anchored/models/arm/seed_18001/silu')/f'critic_{m}.pt' for m in range(5)]
    paths += [Path('anchored/models/arm/seed_18001')/method/'model.pt' for method in ('smooth','residual')]
    for path in paths:
        assert exact_nested(torch.load(generated/path,weights_only=True),torch.load(rt.ASSETS/path,weights_only=True)),str(path)
    notebook=nbformat.read(root/'notebooks/reproduce.ipynb',as_version=4)
    code=[c for c in notebook.cells if c.cell_type=='code']
    assert code and all(c.execution_count is not None for c in code)
    errors=[o for c in code for o in c.outputs if o.output_type=='error']
    assert not errors
    return {
        'date':'2026-10-27',
        'environment':{'python':platform.python_version(),'os':platform.system(),'arch':platform.machine(),
                       'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__},
        'local_environment_note':'Isolated project venv with system-site-packages using the existing pinned scientific environment; not a clean dependency-resolution test. GitHub CI installs separately.',
        'bundled_integrity':rt.verify(),
        'replays':{'runs':42,'exact_final_metric_runs':42,'max_final_metric_error':max(s['max_final_metric_error'] for s in suites),'exact_final_actor_state_dicts':same_actors,
                   'rows':[r for s in suites for r in s['comparisons']]},
        'from_scratch':{'task':'arm','seed':18001,'offline_arrays_exact':len(a.files),'serialized_model_contents_exact':len(paths),'new_offline_outcomes':6144,'other_seeds_regenerated':False},
        'notebook':{'executed_cells':len(code),'error_outputs':len(errors),'fresh_actor_replays':2},
        'research_boundary':'Packaging validation, not new-seed confirmation or algorithm superiority. Six historical solver warnings and overall passed=false remain unchanged.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('provenance/publication_validation.json'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    result=collect(root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='replays'},indent=2))
