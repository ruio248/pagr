"""Post-training-only reference audit and report for frozen directional study.

Never import this file from the dev/formal training runner.  It reads saved
checkpoints and cannot change the frozen alpha or formal results.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_directional_robust_qam as R
from directional_robust_qam import (critic_member_action_gradients,
                                   directional_robust_gradient)
from run_point_long_training import independent_mc


def cosine(x, y):
    norm = np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.dot(x, y)/norm) if norm > 1e-12 else None


def audit(output, episodes=8192):
    p = R.protocol(); c = R.effective_config(p)
    frozen = json.loads((output/'formal-frozen-selection.json').read_text())
    if frozen['selection_sha256'] != R.digest(output/'selected-alpha.json'):
        raise RuntimeError('Formal alpha selection record changed')
    alpha = frozen['alpha']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points = R.M.fixed_reference_points()
    rows = []
    for seed in p['formal_seeds']:
        folder = output/f'seed-{seed}'
        original = torch.load(folder/'shared-initialization.pt', map_location='cpu', weights_only=True)
        base = R.M.core.ConditionalVelocity(c); base.load_state_dict(original['base'])
        task = R.M.PointObstacle(c)
        for method in ('learned','directional_robust'):
            state = torch.load(folder/method/f'{method}-N300.pt', map_location='cpu', weights_only=True)
            actor = R.M.core.ConditionalVelocity(c); actor.load_state_dict(state['actor'])
            critic = R.M.core.Critic(c); critic.load_state_dict(state['critic'])
            policy = copy.deepcopy(actor).to(device).eval()
            for index, (k, pos, action, label) in enumerate(points):
                kk = torch.tensor([k], dtype=torch.long)
                yy = torch.tensor([pos], dtype=torch.float32)
                aa = torch.tensor([action], dtype=torch.float32)
                member = critic_member_action_gradients(critic, kk, yy, aa)
                mean = member.mean(1)[0].numpy()
                filtered, sigma = directional_robust_gradient(member, alpha)
                filtered = filtered[0].numpy()
                ref = independent_mc(task, policy, base, k, np.asarray(pos), np.asarray(action),
                    episodes, R.M.seed_for(seed, 98000+100*index), device)
                true = np.asarray(ref['gradient'])
                se = np.asarray(ref['gradient_se'])
                rows.append(dict(seed=seed, method=method, point=label, k=k,
                    gbar_x=mean[0], gbar_y=mean[1], robust_x=filtered[0], robust_y=filtered[1],
                    ref_x=true[0], ref_y=true[1], se_x=se[0], se_y=se[1],
                    learned_error=float(np.linalg.norm(mean-true)),
                    robust_error=float(np.linalg.norm(filtered-true)),
                    learned_cosine=cosine(mean,true), robust_cosine=cosine(filtered,true),
                    covariance_trace=float(sigma.diagonal(dim1=-2,dim2=-1).sum()),
                    gradient_norm=float(np.linalg.norm(mean)),
                    robust_norm=float(np.linalg.norm(filtered)),
                    reference_se_norm=float(np.linalg.norm(se)), mc_episodes=episodes))
                pd.DataFrame(rows).to_csv(output/'posthoc-reference-gradient-audit.csv', index=False)
                print('AUDIT',seed,method,label,'error',rows[-1]['learned_error'],
                      rows[-1]['robust_error'],flush=True)
    return pd.DataFrame(rows)


def summary_table(frame, by):
    grouped = frame.groupby(by)
    return grouped.agg(success_mean=('success_rate','mean'), success_sd=('success_rate','std'),
        collision_mean=('collision_rate','mean'), collision_sd=('collision_rate','std'),
        timeout_mean=('timeout_rate','mean'), return_mean=('discounted_return','mean'),
        return_sd=('discounted_return','std')).reset_index()


def report(output):
    p=R.protocol(); alpha=json.loads((output/'selected-alpha.json').read_text())['alpha']
    dev=pd.read_csv(output/'development-alpha-sweep.csv')
    curves=pd.read_csv(output/'formal-curves.csv')
    finals=pd.read_csv(output/'formal-final.csv')
    pairs=json.loads((output/'paired-outcome-summary.json').read_text())
    compare=finals.pivot(index='seed',columns='method',values='success_rate')
    delta=compare['directional_robust']-compare['learned']
    means=summary_table(finals,['method']); fixed=summary_table(curves,['N','method'])
    means.to_csv(output/'formal-aggregate.csv', index=False)
    fixed.to_csv(output/'fixed-N-aggregate.csv', index=False)
    audit_path=output/'posthoc-reference-gradient-audit.csv'
    audit=pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    dev_selected=output/'seed-59000'/f'development_alpha_{str(alpha).replace(".","p")}'/'directional_robust-train.csv'
    diag=pd.read_csv(dev_selected) if dev_selected.exists() else pd.DataFrame()
    formal_logs=list(output.glob('seed-5900[1-3]/directional_robust/directional_robust-train.csv'))
    formal_diag=pd.concat([pd.read_csv(path) for path in formal_logs],ignore_index=True) if formal_logs else pd.DataFrame()
    def pct(v): return f'{100*v:.2f}'
    lines=[
        '# Directional Robust QAM v1: PointObstacle study', '',
        '## Definition and information boundary', '',
        r'At each AM endpoint $a=X_1$, obtain $g_m=\nabla_a Q_m(s,a)$ separately for every critic member. Let $\bar g=M^{-1}\sum_mg_m$ and $\Sigma_g=(M-1)^{-1}\sum_m(g_m-\bar g)(g_m-\bar g)^\top$. The algorithm uses $\hat\Sigma_g=\Sigma_g/(\|\bar g\|_2^2+10^{-12})$, $\tilde g=(I+\alpha\hat\Sigma_g)^{-1}\bar g$, and terminal label $p_1=-\beta\tilde g$. The matrix system is solved, not inverted. The ordinary branch still uses the untouched $p_1=-\beta\bar g$.', '',
        'Training uses only the TD critic ensemble and its per-member action gradients. **No reference gradient is used in training, alpha selection, or checkpoint selection.** Neither the task, Base Flow, TD loss, sampler, beta, nor Lean Adjoint was changed.', '',
        '## Development selection', '',
        'Seed 59000 only; N=50; 1024 held-out episodes; candidate set {0, 0.1, 0.3, 1, 3, 10}. Rank by success descending, collision ascending, return descending, then smaller alpha. Formal alpha was frozen before any new formal run.', '',
        '| Alpha | Success % | Collision % | Return |', '|---:|---:|---:|---:|']
    for _,row in dev.iterrows():
        lines.append(f"| {row.alpha:g} | {pct(row.success_rate)} | {pct(row.collision_rate)} | {row.discounted_return:.4f} |")
    lines += ['',f'Frozen formal alpha: **{alpha:g}**. The development advantage over alpha 0 is small and should not be overinterpreted.', '',
              '## Formal final evaluation (N=300, 4096 independent episodes per seed/method)', '',
              '| Seed | Method | Success % | Collision % | Timeout % | Return |',
              '|---:|---|---:|---:|---:|---:|']
    for _,row in finals.iterrows():
        lines.append(f"| {int(row.seed)} | {row.method} | {pct(row.success_rate)} | {pct(row.collision_rate)} | {pct(row.timeout_rate)} | {row.discounted_return:.4f} |")
    lines += ['', '| Method | Success mean ± SD % | Collision mean ± SD % | Return mean ± SD |',
              '|---|---:|---:|---:|']
    for _,row in means.iterrows():
        lines.append(f"| {row.method} | {pct(row.success_mean)} ± {pct(row.success_sd)} | {pct(row.collision_mean)} ± {pct(row.collision_sd)} | {row.return_mean:.4f} ± {row.return_sd:.4f} |")
    lines += ['',f'Mean Δ success (robust − ordinary): **{100*delta.mean():+.3f} percentage points**; per-seed: '
              +', '.join(f'{int(s)} {100*v:+.3f} pp' for s,v in delta.items())
              +f'. Robust wins {int((delta>0).sum())}/3 seeds, ties {int((delta==0).sum())}/3. The rerun Ordinary baseline reproduces the prior v2 N=300 final metrics exactly for all three seeds.', '',
              '## Fixed-N success curves (mean ± SD across three seeds; same 1024-episode bank within each seed)', '',
              '| N | Ordinary % | Robust % | Δ pp |', '|---:|---:|---:|---:|']
    for N in p['formal_checkpoints']:
        old=fixed[(fixed.N==N)&(fixed.method=='learned')].iloc[0]
        new=fixed[(fixed.N==N)&(fixed.method=='directional_robust')].iloc[0]
        lines.append(f"| {N} | {pct(old.success_mean)} ± {pct(old.success_sd)} | {pct(new.success_mean)} ± {pct(new.success_sd)} | {100*(new.success_mean-old.success_mean):+.3f} |")
    lines += ['', 'The complete fixed-N measurements are retained below and in `formal-curves.csv`.', '',
              '| Seed | Method | N | Success % | Collision % | Timeout % | Return |',
              '|---:|---|---:|---:|---:|---:|---:|']
    for _, row in curves.sort_values(['seed','N','method']).iterrows():
        lines.append(f"| {int(row.seed)} | {row.method} | {int(row.N)} | {pct(row.success_rate)} | {pct(row.collision_rate)} | {pct(row.timeout_rate)} | {row.discounted_return:.4f} |")
    aggregate=pairs['aggregate']
    lines += ['', '## Final 50 paired physical trajectories per seed', '',
        f"Across 150 paired episodes: ordinary failure→robust success **{aggregate['ordinary_failure_to_robust_success']}**, ordinary success→robust failure **{aggregate['ordinary_success_to_robust_failure']}**, both success **{aggregate['both_success']}**, both failure **{aggregate['both_failure']}**, ordinary collision→robust success **{aggregate['ordinary_collision_to_robust_success']}**.", '',
        'Each seed saves `test50/initial_states.npy`, `noise_bank.npz`, per-episode outcomes, full physical trajectories, and all/success/failure plots. The ODE deployment sampler uses the shared z noise; the saved Flow tensor is present in the bank but unused by that sampler.', '',
        '## Mechanism and post-training reference-gradient audit', '']
    if not diag.empty:
        ratio=diag.mean_robust_g_norm.mean()/diag.mean_g_norm.mean()
        lines += [f"On development diagnostic batches, mean gradient norm ratio robust/ordinary = **{ratio:.4f}**; mean cosine = **{diag.mean_cosine.mean():.4f}**; mean covariance trace = **{diag.mean_cov_trace.mean():.6g}**; mean shrink factors on principal axes = **{diag.mean_shrink_1.mean():.4f}**, **{diag.mean_shrink_2.mean():.4f}**. These are descriptive and were not used for alpha selection.", '']
    if not formal_diag.empty:
        ratio=formal_diag.mean_robust_g_norm.mean()/formal_diag.mean_g_norm.mean()
        lines += [f"Across formal diagnostic batches (logged every 10 Actor updates), norm ratio = **{ratio:.4f}**, mean cosine = **{formal_diag.mean_cosine.mean():.4f}**, covariance trace = **{formal_diag.mean_cov_trace.mean():.6g}**. Maximum pre-clip Actor gradient norm was **{formal_diag.pre_clip_norm.max():.3f}**, below the fixed clip threshold 5; clipping did not erase the observed branch difference. With only M=3 ensemble members, covariance estimation stability remains an untested limitation.", '']
    if not audit.empty:
        lines += [f"Post-training only: {len(audit)} fixed audit point/checkpoint pairs, {int(audit.mc_episodes.iloc[0])} independent Monte Carlo rollouts each. Mean Euclidean error to current-policy MC reference: ordinary mean-gradient **{audit.learned_error.mean():.4f}**, filtered gradient **{audit.robust_error.mean():.4f}**; filtered is closer on **{int((audit.robust_error<audit.learned_error).sum())}/{len(audit)}** points. Mean cosine to reference changes from **{audit.learned_cosine.mean():.4f}** to **{audit.robust_cosine.mean():.4f}** (slightly worse); mean MC gradient SE norm **{audit.reference_se_norm.mean():.4f}**. The covariance-trace versus original error correlation at these fixed points is **{audit.covariance_trace.corr(audit.learned_error):.3f}**. The tiny norm-error improvement is below MC precision and does not establish better gradient direction. These are descriptive diagnostics, not a performance/selection criterion.", '']
    else:
        lines += ['Reference audit not yet complete; no reference-proximity claim is made.', '']
    lines += ['## Interpretation and next step', '',
        'The frozen first-version directional filter **did not outperform Ordinary Learned-QAM** at equal Actor and TD update counts. Because there is no robust performance gain, a global-scale control cannot explain a gain here; it was not run on formal seeds. The current evidence does **not** justify claiming a successful propagation-aware method. The next algorithm iteration may be developed only on seed 59000, after inspecting whether ensemble covariance is informative, whether M=3 is too small, whether shrink is effectively zero/global beta scaling, and whether Actor clipping dominates.', '',
        'Research caveat: these formal seed IDs appeared in an earlier reference-gradient study, so they are not wholly unseen. The new alpha was nevertheless selected from seed 59000 only, before this formal comparison. The failed `-001` output was retained: a tuple/list manifest comparison bug stopped formal before any seed ran. The corrected and re-run experiment is `-002`; no formal alpha or seed was adjusted.', '']
    (output/'report.md').write_text('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--audit',action='store_true'); ap.add_argument('--mc-episodes',type=int,default=8192)
    args=ap.parse_args()
    if args.audit: audit(args.output,args.mc_episodes)
    report(args.output)


if __name__=='__main__': main()
