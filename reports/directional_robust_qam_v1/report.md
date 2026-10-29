# Directional Robust QAM v1 — two-dimensional PointObstacle

This is the first terminal-gradient-only robustness experiment. It is **not** a propagation-aware method and is not called PAGR. The full frozen configuration, 42 fixed-N measurements, 50-paired-episode files for each seed, post-training audit, checkpoints, and generated detailed report are retained on Ubuntu in `/hdd/pagr-long-training-20261029/runs/02-qam-point-obstacle-directional-v1-20261029-002`. The local `runs/02-qam-point-obstacle-directional-v1-20261029-002` is a compact mirror; it does not duplicate all checkpoint weights. No GitHub push was performed.

For endpoint $a=X_1$, the method obtains each member's actual $g_m=\nabla_a Q_m(s,a)$, forms the sample covariance $\Sigma_g$, and uses

\[
\tilde g=\left(I+\alpha\frac{\Sigma_g}{\|\bar g\|^2+10^{-12}}\right)^{-1}\bar g,
\qquad p_1=-\beta\tilde g.
\]

Training and tuning do not access a reference gradient. Ordinary QAM's Actor step, TD loss, Base, Lean Adjoint, environment, and reward are unchanged. Alpha was selected *only* from development seed 59000 at N=50 on a 1024-episode held-out bank, using the predeclared success/collision/return ordering over `{0, 0.1, 0.3, 1, 3, 10}`. The frozen choice was **alpha = 3**; its development advantage over alpha 0 was only one success out of 1024 episodes.

## Final N=300 results

Each entry uses the same 4096-episode evaluation bank within a seed.

| Seed | Method | Success % | Collision % | Return |
|---:|---|---:|---:|---:|
| 59001 | Ordinary Learned-QAM | 52.29 | 47.68 | -0.01635 |
| 59001 | Directional Robust QAM | 52.29 | 47.68 | -0.01636 |
| 59002 | Ordinary Learned-QAM | 49.71 | 50.29 | -0.05823 |
| 59002 | Directional Robust QAM | 49.66 | 50.34 | -0.05896 |
| 59003 | Ordinary Learned-QAM | 51.37 | 48.63 | -0.03172 |
| 59003 | Directional Robust QAM | 51.17 | 48.83 | -0.03502 |

Mean success was **51.12% ± 1.31%** for Ordinary and **51.04% ± 1.32%** for Robust (sample SD across three seeds). The mean Robust–Ordinary difference was **−0.081 percentage points**: no seed improved, one tied, two worsened. The Ordinary N=300 metrics exactly reproduced the preceding v2 study. Full N=0/50/100/150/200/250/300 results are in `formal-curves.csv` and the generated run report.

Across 150 paired final physical trajectories (50 per seed), Robust repaired **0** Ordinary failures and introduced **0** new failures; both succeeded on 78 and both failed on 72. Collision→success was **0**. Each seed retains the three requested trajectory plots and per-episode records.

## Why the result is negative

Across formal diagnostic batches, the filtered gradient norm was about **96.4%** of the ensemble-mean gradient norm, with mean directional cosine **0.993**. Mean covariance trace was about **0.000261**. The largest pre-clip Actor parameter-gradient norm was **0.816**, below the fixed clip threshold 5, so clipping did not erase the difference. With three Critic members, covariance reliability itself remains an open question.

The independent post-training MC audit used 8192 rollouts at each of 30 fixed point/checkpoint pairs. Mean gradient error was **0.1045** for the ensemble mean and **0.1033** after filtering; Robust was closer at 19/30 points. But the MC reference standard-error norm averaged **0.0109**, larger than the mean improvement. Mean cosine to the reference slightly *decreased* (0.4950→0.4901). These diagnostics do not establish an accuracy gain and were never used to choose alpha. Covariance trace had approximately zero correlation (−0.058) with original gradient error at these fixed points.

There is no performance gain for a global-scale control to explain, so no formal Global-Scale branch was run. This v1 result does **not** justify claiming success or moving directly to propagation-aware PAGR. Any v2 method changes should be developed on seed 59000 only before another predeclared evaluation.

Research caveat: seeds 59001–59003 had appeared in an earlier reference-gradient study, so they are not wholly unseen. The new alpha was still frozen using seed 59000 before this comparison. An earlier `-001` run is preserved: a tuple/JSON-list manifest comparison bug stopped formal before any seed began. The corrected `-002` reran development and formal from scratch without changing the candidate set or evaluation protocol.
