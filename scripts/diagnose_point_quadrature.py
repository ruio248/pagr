"""Diagnose constant-function leakage in finite Gaussian score quadrature.

This does not edit a gate, a checkpoint, or an existing experiment result.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from point_reference_accelerated import M, AcceleratedPolicyTable


def run(snapshot, output):
    output.mkdir(parents=True, exist_ok=False)
    sub = snapshot.parent
    c, _ = M.load_config()
    actor = M.load_actor(snapshot, c)
    base = M.load_base(sub/'shared-initialization.pt', c)
    audit = sub/'audit-reference-N50'
    points = json.loads((audit/'points.json').read_text())
    k, p, a = np.array(points['k']), np.array(points['p']), np.array(points['a'])
    old = pd.read_csv(audit/'gradients.csv')
    table = AcceleratedPolicyTable(M.PointObstacle(c), actor, base, 81, 32, 512, 8192)
    q, raw = table.q_gradient(k, p, a)
    xi = table.gradient_quad[0]
    leakage = c.delta*(1-np.tanh(a)**2)*xi.mean(0)*q[:, None]/c.sigma_env
    rows = []
    for samples in (16384, 65536):
        half = M.normal_quadrature(samples//2, seed=1994)[0]
        nodes = np.vstack((half, -half))
        table.gradient_quad = (nodes, np.full(samples, 1/samples))
        _, grad = table.q_gradient(k, p, a)
        mc = old[['mc_x', 'mc_y']].to_numpy()
        se = old[['se_x', 'se_y']].to_numpy()
        z = np.abs(grad-mc)/np.maximum(se, 1e-8)
        for i, point_id in enumerate(points['ids']):
            rows.append(dict(point_id=point_id, samples=samples,
                        raw_x=raw[i, 0], raw_y=raw[i, 1],
                        constant_leakage_x=leakage[i, 0], constant_leakage_y=leakage[i, 1],
                        antithetic_x=grad[i, 0], antithetic_y=grad[i, 1],
                        mc_x=mc[i, 0], mc_y=mc[i, 1],
                        z_x=z[i, 0], z_y=z[i, 1]))
        print(json.dumps(dict(samples=samples, gradient_z_rms=float(np.sqrt(np.mean(z*z))),
                              gradient_z_max=float(z.max()),
                              gradient_mc_rms=float(np.sqrt(np.mean((grad-mc)**2))),
                              endpoint04=dict(raw=raw[12].tolist(), leakage=leakage[12].tolist(),
                                              antithetic=grad[12].tolist(), mc=mc[12].tolist()))),
              flush=True)
    pd.DataFrame(rows).to_csv(output/'quadrature-comparison.csv', index=False)
    (output/'provenance.json').write_text(json.dumps(dict(actor_hash=M.model_hash(actor),
              original_normal_mean=xi.mean(0).tolist(),
              purpose='post-failure numerical diagnosis; old gate/results unchanged'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.snapshot, args.output)
