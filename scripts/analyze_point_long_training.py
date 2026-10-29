"""Render all saved learning curves and paired outcomes without checkpoint selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze(run, plots_only=False):
    run = Path(run)
    frozen = json.loads((run/'protocol.json').read_text())
    protocol = frozen['protocol']
    parts = [pd.read_csv(p) for p in sorted(run.glob('seed-*/curves.csv'))]
    if not parts:
        raise RuntimeError('No checkpoint evaluations yet')
    curves = pd.concat(parts, ignore_index=True)
    curves.to_csv(run/'curves.csv', index=False)
    names = {'learned': 'Ordinary QAM', 'reference': 'Reference-gradient QAM'}
    colors = {'learned': '#3276aa', 'reference': '#e17c24'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    metrics = [('success_rate', 'Success rate', 100),
               ('collision_rate', 'Collision / out-of-bounds rate', 100),
               ('discounted_return', 'Discounted return', 1)]
    for ax, (metric, title, scale) in zip(axes, metrics):
        for method in names:
            rows = curves[curves.method == method]
            for seed, part in rows.groupby('seed'):
                ax.plot(part.N, scale*part[metric], color=colors[method], alpha=.22, linewidth=1)
            summary = rows.groupby('N')[metric].agg(['mean', 'std', 'count']).reset_index()
            ax.plot(summary.N, scale*summary['mean'], color=colors[method], marker='o',
                    label=names[method], linewidth=2)
            ax.fill_between(summary.N, scale*(summary['mean']-summary['std'].fillna(0)),
                            scale*(summary['mean']+summary['std'].fillna(0)),
                            color=colors[method], alpha=.12)
        ax.set(xlabel='Actor optimizer updates N', ylabel=title+(' (%)' if scale == 100 else ''))
        if scale == 100:
            ax.set_ylim(0, 100)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=9)
    coverage = curves.groupby(['method', 'N']).seed.nunique()
    partial = (len(coverage) != 2*len(protocol['curve_updates']) or
               not (coverage == len(protocol['seeds'])).all())
    fig.suptitle(('PARTIAL — seed coverage varies; ' if partial else 'Equal Actor-update budgets; ')+
                 'thin lines = training seeds; band = seed SD')
    fig.tight_layout()
    fig.savefig(run/'learning-curves.png', dpi=180)
    plt.close(fig)
    paired = curves.pivot(index=['seed', 'N'], columns='method', values='success_rate').dropna()
    paired['gap_percentage_points'] = 100*(paired['reference']-paired['learned'])
    paired.reset_index().to_csv(run/'success-gap-by-seed.csv', index=False)
    aggregates = paired.groupby('N').gap_percentage_points.agg(['mean', 'std', 'count'])
    aggregates.to_csv(run/'success-gap-aggregate.csv')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seed, part in paired.reset_index().groupby('seed'):
        ax.plot(part.N, part.gap_percentage_points, marker='o', alpha=.65, label=f'Seed {seed}')
    ax.axhline(0, color='black', linewidth=1)
    ax.set(xlabel='Actor optimizer updates N', ylabel='Reference − Ordinary success (percentage points)')
    ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(run/'success-gap.png', dpi=180); plt.close(fig)
    if plots_only:
        # A lightweight local mirror need not download every remote checkpoint
        # or rollout. Keep the remotely validated full report and paired table.
        print(json.dumps(dict(plots_only=True, rows=len(curves), seeds=sorted(curves.seed.unique().tolist()))))
        return
    efficiency = []
    for (seed, method), part in curves.groupby(['seed', 'method']):
        part = part.sort_values('N')
        auc = float(np.trapezoid(part.success_rate, part.N)) if len(part) > 1 else 0.
        for threshold in protocol['success_thresholds']:
            reached = part[part.success_rate >= threshold]
            efficiency.append(dict(seed=seed, method=method, success_threshold=threshold,
                               first_observed_update=int(reached.N.min()) if len(reached) else None,
                               reached=bool(len(reached)), max_observed_update=int(part.N.max()),
                               success_auc=auc,
                               normalized_success_auc=auc/part.N.max() if part.N.max() else None))
    pd.DataFrame(efficiency).to_csv(run/'sample-efficiency.csv', index=False)
    final_files = sorted(run.glob('seed-*/final-independent.csv'))
    finals = pd.concat([pd.read_csv(p) for p in final_files], ignore_index=True) if final_files else pd.DataFrame()
    if len(finals):
        finals.to_csv(run/'final-independent.csv', index=False)
        final_aggregate = finals.groupby('method')[[
            'success_rate', 'collision_rate', 'discounted_return']].agg(['mean', 'std', 'count'])
        final_aggregate.to_csv(run/'final-aggregate.csv')
    costs = []
    for path in sorted(run.glob('seed-*/train.csv')):
        train = pd.read_csv(path)
        for method, part in train.groupby('method'):
            costs.append(dict(seed=int(part.seed.iloc[0]), method=method,
                              updates=len(part), update_seconds=float(part.update_seconds.sum()),
                              reference_seconds=float(part.reference_seconds.sum()),
                              extra_simulator_continuations=(
                                  len(part)*frozen['config']['actor_batch']*protocol['reference_mc_episodes']
                                  if method == 'reference' and protocol.get('reference_estimator') ==
                                  'iid-monte-carlo-score' else 0)))
    pd.DataFrame(costs).to_csv(run/'training-cost.csv', index=False)
    audits = []
    for path in sorted(run.glob('seed-*/audit-*/summary.json')):
        item = json.loads(path.read_text())
        audits.append(dict(seed=item['seed'], method=item['method'], N=item['N'],
                           passed=item['passed'], reference_mc_rms=item['reference_mc_rms'],
                           mc_se_rms=item['mc_se_rms'], gradient_z_rms=item['gradient_z_rms'],
                           gradient_z_max=item['gradient_z_max'], q_mc_z_rms=item['q_mc_z_rms'],
                           critic_reference_rms=item['critic_reference_rms'],
                           higher_precision_drift_rms=item['higher_precision_drift_rms'],
                           training_reference_se_rms=item.get('training_reference_se_rms', 0.),
                           estimator=item.get('estimator', 'policy-table')))
    pd.DataFrame(audits).to_csv(run/'gradient-audits.csv', index=False)
    final_pairs = []
    for seed in protocol['seeds']:
        sub = run/f'seed-{seed}'
        left, right = [sub/f'final-{method}-N{protocol["actor_updates"]}.npz'
                       for method in ('learned', 'reference')]
        if not (left.exists() and right.exists()):
            continue
        with np.load(left) as aa, np.load(right) as bb:
            success_a, success_b = aa['status'] == 'success', bb['status'] == 'success'
            d = success_b.astype(float)-success_a.astype(float)
            reward_delta = bb['returns']-aa['returns']
            final_pairs.append(dict(seed=seed, episodes=len(d),
                         repaired=int(np.sum(~success_a & success_b)),
                         regressed=int(np.sum(success_a & ~success_b)),
                         both_success=int(np.sum(success_a & success_b)),
                         both_fail=int(np.sum(~success_a & ~success_b)),
                         success_gap=float(d.mean()),
                         within_seed_paired_se=float(d.std(ddof=1)/np.sqrt(len(d))),
                         return_gap=float(reward_delta.mean())))
    pd.DataFrame(final_pairs).to_csv(run/'final-paired-outcomes.csv', index=False)
    complete = len(final_pairs) == len(protocol['seeds']) and (run/'complete.json').exists()
    all_audits_pass = bool(audits) and all(a['passed'] for a in audits)
    # N=0 reference shares the identical ordinary Actor certificate, then 3 times x 2 methods.
    expected_audits = len(protocol['seeds'])*(2*len(protocol['audit_updates'])-1)
    certified = complete and all_audits_pass and len(audits) == expected_audits
    is_mc = protocol.get('reference_estimator') == 'iid-monte-carlo-score'
    reference_description = ('The current-policy gradient is estimated locally from 8192 IID simulator '
                             'rollouts per training endpoint; no value table or learned reference Critic is used.'
                             if is_mc else 'Each reference value table is rebuilt for the current Actor before its update.')
    lines = ['# Obstacle QAM: training-budget extension', '',
             f'Status: {"complete" if complete else "in progress / partial"}.',
             f'All required snapshot gradient checks passed: {certified}.', '',
             'The user approved a new gradient-focused admission protocol. Scalar Q–MC',
             'discrepancy is reported but does not veto a gradient comparison. Historical',
             'audit FAIL records are unchanged. This is a numerical privileged-reference',
             'comparison, not an exact oracle or a new PAGR algorithm.', '',
             'Three predeclared seeds; 300 Actor updates each; the same TD update counts,',
             'initialization and training RNG streams within each pair.',
             reference_description,
             'Extra reference computation is not equalized with Ordinary QAM.', '',
             'Success and discounted return are environment-performance metrics. They',
             'are not the KL/energy-regularized AM extraction objective; this extension',
             'does not establish an improvement in that full control objective.', '',
             '## Observed success gaps', '',
             '| Actor updates | Mean reference − ordinary (pp) | Seed SD (pp) | Seeds |',
             '|---:|---:|---:|---:|']
    for N, row in aggregates.iterrows():
        sd = f'{row["std"]:.3f}' if np.isfinite(row['std']) else '—'
        lines.append(f'| {N} | {row["mean"]:.3f} | {sd} | {int(row["count"])} |')
    lines += ['', '![All learning curves](learning-curves.png)', '',
              '![Paired success gap](success-gap.png)', '',
              '## Independent final evaluation', '']
    if final_pairs:
        lines += ['| Method | Success (%) | Collision / bounds (%) | Discounted return |',
                  '|---|---:|---:|---:|']
        for method, part in finals.groupby('method'):
            def cell(metric, scale=1):
                values = part[metric]*scale
                return (f'{values.mean():.3f} ± {values.std(ddof=1):.3f}'
                        if len(values) > 1 else f'{values.mean():.3f} (one seed)')
            lines.append(f'| {names[method]} | {cell("success_rate", 100)} | '
                         f'{cell("collision_rate", 100)} | {cell("discounted_return")} |')
        lines += ['', 'Values are training-seed mean ± SD, not episode-level uncertainty.', '']
        lines += ['| Seed | Success gap (pp) | Return gap | Failures repaired | Successes lost |',
                  '|---:|---:|---:|---:|---:|']
        for row in final_pairs:
            lines.append(f'| {row["seed"]} | {100*row["success_gap"]:.3f} | {row["return_gap"]:.5f} | '
                         f'{row["repaired"]} | {row["regressed"]} |')
        delta = np.array([r['success_gap'] for r in final_pairs])
        lines += ['', f'Mean final success gap: {100*delta.mean():.3f} percentage points.',
                  'Within-seed evaluation SE is not training-seed uncertainty. Three seeds',
                  'remain a small exploratory sample; no universal advantage is inferred.']
    else:
        lines += ['Final checkpoints have not all been evaluated.']
    if costs:
        lines += ['', '## Compute accounting', '',
                  '| Method | Updates recorded | Update wall time (s) | Extra simulator continuations |',
                  '|---|---:|---:|---:|']
        for method in names:
            selected = [row for row in costs if row['method'] == method]
            lines.append(f'| {names[method]} | {sum(row["updates"] for row in selected)} | '
                         f'{sum(row["update_seconds"] for row in selected):.2f} | '
                         f'{sum(row["extra_simulator_continuations"] for row in selected)} |')
        lines += ['', 'These timings include TD and Actor/reference computation, but exclude',
                  'shared pretraining, evaluation, checkpoint serialization and precision audits.',
                  'Simulator continuations start at sampled environment times, so they have',
                  'different remaining horizons; they are not full episodes from the initial state.']
    lines += ['', 'Snapshot validation samples eight frozen geometry cases and eight actual',
              'training endpoints at N=0/50/150/300 per branch, with two independent',
              ('8192-rollout MC streams per point and an independent 32768-rollout reference.'
               if is_mc else '8192-rollout MC streams per point and a higher-resolution table.'),
              'It is sampled validation, not proof of accuracy everywhere or at every update.', '',
              'Fixed success thresholds are 60%, 70%, 80%; first observed crossings are',
              'checkpoint-grid measurements, not interpolated exact crossing times.',
              'All curves, gaps, threshold crossings, final paired episodes and audit',
              'tables are saved alongside this report.']
    if is_mc:
        lines += ['', 'The table-based v1 run paused at seed 59001 reference N=50 after a',
                  'gradient mismatch near the goal. Refining the table/quadrature did not',
                  'remove it. This v2 run restarts Actor updates from N=0 with a direct MC',
                  'reference, retaining the same seeds, initialization and acceptance',
                  'thresholds. Stochastic-reference comparison SE combines both independent',
                  'estimator variances. Seed 59001 was seen during numerical development;',
                  'this is exploratory evidence, not an untouched confirmatory seed set.']
        lines += ['', 'The saved scalar `q_mc_z_rms` diagnostic normalizes by audit MC SE only;',
                  'unlike the gradient gate it does not include the stochastic training-reference',
                  'variance. Do not interpret it as a calibrated two-estimator significance test.',
                  'No scalar-Q diagnostic determines training admission in this protocol.']
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=complete, gradient_certified=certified,
                         audit_count=len(audits), final_seed_count=len(final_pairs))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--plots-only', action='store_true', help='Render curves without replacing the full report')
    args = parser.parse_args()
    analyze(args.run, plots_only=args.plots_only)
