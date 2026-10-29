# Global guidance-scale confound audit

Frozen Learned-QAM beta: **16**; formal comparison: 300 updates on seeds 59001–59003.
The development seed is historical and the formal seeds were previously studied; all performance results are exploratory, not unseen confirmation.

## Direct answers

1. **Learned beta:** **16**, selected on development seed 59000 at N=300 with no MC/reference labels.
2. **Tuned Learned performance:** see per-seed and aggregate rows below.
3. **Same-beta reference advantage:** computed as reference minus tuned Learned for each seed.
4. **Global-scale absorbable gradient component:** quantified by gradient-space projected residuals at N=0/50/100.
5. **Actor-gradient residual:** see `actor-gradient-scale-residual.csv`.
6. **Adam-step residual:** see `adam-step-scale-residual.csv`.
7. **Scale-matched reference:** compared with the oracle-calibrated best-scaled Learned branch after matching terminal-adjoint RMS at N=0.
8. **MC semantics:** see `oracle_semantic_validation.json`; the current MC estimator is a finite-sample current-policy continuation score estimator, not an exact oracle.
9. **Evidence class:** determined below; one-seed scale-matched result is mechanism evidence only, not a multi-seed claim.
10. **Benchmark recommendation:** determined below; do not generalize beyond PointObstacle.

## Formal tuned-Learned vs same-beta Reference

Success, collision, and discounted return are evaluated on the same 4096-episode bank per seed. The current toy has sparse terminal reward and no separately specified KL/control-energy objective, so `J_SOC` is not defined/reported.

| seed | branch | success | collision | discounted return |
|---:|---|---:|---:|---:|
| 59001 | learned_tuned | 60.303% | 39.697% | 0.121385 |
| 59001 | reference_same_beta | 72.339% | 27.661% | 0.323139 |
| 59002 | learned_tuned | 60.156% | 39.844% | 0.121394 |
| 59002 | reference_same_beta | 74.048% | 25.952% | 0.352435 |
| 59003 | learned_tuned | 66.699% | 33.301% | 0.224692 |
| 59003 | reference_same_beta | 74.634% | 25.366% | 0.361499 |

Three-seed mean ± SD:

| branch | success | collision | discounted return |
|---|---:|---:|---:|
| learned_tuned | 62.39 ± 3.74% | 37.61 ± 3.74% | 0.15582 ± 0.05964 |
| reference_same_beta | 73.67 ± 1.19% | 26.33 ± 1.19% | 0.34569 ± 0.02005 |

## Development beta sweep (seed 59000; N=300; 4096 final episodes)

I deliberately used N=300 rather than the pasted draft's N=100 so beta selection and the primary formal comparison share the same training horizon; this is a protocol clarification, not a hidden change. If strict reproduction of the original N=100 draft is required, the sweep must be rerun at N=100 and its selected beta compared as a sensitivity check.
| beta | success | collision | discounted return | clip fraction | mean Adam step |
|---:|---:|---:|---:|---:|---:|
| 4 | 53.247% | 46.753% | 0.000936 | 0.0% | 0.005026 |
| 8 | 59.253% | 40.747% | 0.101164 | 0.0% | 0.006879 |
| 12 | 62.842% | 37.158% | 0.163466 | 0.0% | 0.007901 |
| 16 | 65.405% | 34.595% | 0.208087 | 0.0% | 0.009191 |
| 24 | 58.179% | 41.821% | 0.105310 | 11.7% | 0.010925 |
| 32 | 48.022% | 51.978% | -0.043422 | 22.0% | 0.011183 |
| 48 | 31.348% | 68.652% | -0.289917 | 35.0% | 0.012142 |

## Actor clipping and realized update scale

Same beta does not imply equal parameter-gradient magnitude; report clipping frequency and realized Adam-step norm across all formal updates.

| branch | clip fraction | mean pre-clip parameter-gradient norm | mean actual Adam-step norm |
|---|---:|---:|---:|
| learned_tuned | 0.0% | 0.7580 | 0.008828 |
| reference_same_beta | 15.2% | 3.4626 | 0.008804 |

Paired same-beta success differences (Reference − Learned, pp):

| seed | delta success |
|---:|---:|
| 59001 | +12.036 |
| 59002 | +13.892 |
| 59003 | +7.935 |

## Gradient / Actor / Adam scale residual

The scalar projection is least-squares on the fixed shared diagnostic batch; residual is orthogonal to the learned vector (up to numerical precision). Cosine is included because positive scaling cannot improve it.

| N | c_g | gradient residual / ref | c_q | Actor-gradient residual / ref | c_Adam | Adam-step residual / ref |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.6660 | 0.7661 | 3.0881 | 0.3959 | 0.7897 | 0.6135 |
| 50 | 1.8883 | 0.8778 | 4.3110 | 0.8351 | 0.9468 | 0.8872 |
| 100 | 1.6584 | 0.8942 | 1.9512 | 0.9534 | 1.2578 | 0.8216 |

## N=100 scale-matched branch (seed 59001 only)

The single global scales were frozen from the N=0 diagnostic gradient batch before the scale-matched branch training; no success/return was used to choose them.

| branch | scale | success | collision | discounted return | clip fraction | mean Adam step |
|---|---:|---:|---:|---:|---:|---:|
| learned_baseline | 1.00000 | 60.840% | 39.160% | 0.128105 | 0.0% | 0.009813 |
| best_scaled_learned | 2.66600 | 65.332% | 34.668% | 0.216954 | 51.0% | 0.014153 |
| scale_matched_reference | 0.64273 | 67.578% | 32.422% | 0.240487 | 4.0% | 0.009892 |
| full_reference | 1.00000 | 69.165% | 30.835% | 0.269310 | 23.0% | 0.010705 |

Formal mean same-beta gap: **+11.287 pp**; all seeds positive: `True`.
Scale-matched N=100 seed-59001 gap: **+2.246 pp** (reference minus best-scaled Learned).
Paired 4096-episode evaluation interval (episode noise only): **[+0.925, +3.567] pp**; it does not include training-seed variation.
The N=0 terminal-adjoint RMS match used `c_g=2.6660` and `alpha=0.6427`, frozen for both 100-update branches.

## Interpretation and stop/continue decision

Mixed/inconclusive — global scale explains a large part of the N=100 gap, while non-scalar residuals remain large; the single-seed performance residual is below the predeclared 4-pp multi-seed bar.

**Recommendation:** Do not develop a new robust-QAM method from this result. If continuing this benchmark, the next justified step is a fresh preregistered scale-matched multi-seed confirmation; otherwise retire PointObstacle as the main benchmark.

## Oracle semantic validation

The semantic audit is a source-level check of first-action conditioning, future current-policy actions, discount/termination, tanh chain rule, critic independence, and stochasticity; endpoint records also retain the current Actor hash passed to the MC copy. Because this is Monte Carlo, it remains a finite-sample reference, not an exact oracle.
The online-vs-target table below is a separate read-only audit of the existing long-MC-v2 seed-59001 snapshots (`runs/02-qam-point-obstacle-long-mc-v2-20261029`), not a retraining or a target-critic version of the beta-16 formal run.

The online-vs-target audit found the EMA target gradient closer to MC on 53.1%, 53.1%, and 54.7% of 64 endpoints at N=0/50/100; their relative RMS errors were nearly equal. This does not support switching to the target Critic as the explanation for the performance gap.

| N | target closer fraction | online relative RMS error | target relative RMS error |
|---:|---:|---:|---:|
| 0 | 0.531 | 0.8660 | 0.8607 |
| 50 | 0.531 | 0.8768 | 0.8688 |
| 100 | 0.547 | 0.8741 | 0.8719 |

## Run provenance

Formal training runner SHA-256: `e73aba59f93c3bafa4258c459cd86c34e7b1f0cf45dd43563d334ec51410d4c0` (captured before training).
Scale-diagnostic runner SHA-256: `eff72e2141cb2c312fc67fea02be7446ab63958619228d5a30ce9eaac0e91e2e`; config SHA-256: `ed7c0bd4a4ee2e7c84f8fb36185d9f88f03d59a499e15b50b75f8908cc22556d`.
Formal inputs were validated against the source run's input hashes. Development seed 59000 lacked historical per-input hashes; its legacy frozen protocol and source-code hashes were checked, and its input hashes were frozen in the new protocol.

## Caveats

- The development and formal seeds have prior exposure; results are exploratory and cannot support an unbiased generalization claim.
- The scale-matched branch is one seed and 100 updates; it is a privileged mechanism diagnostic, not a primary multi-seed performance result.
- A good global scale is fit with MC reference gradients and is therefore an oracle diagnostic, not an implementable Learned-QAM tuning procedure.
- `J_SOC` is omitted because this benchmark's specified objective does not include a control-energy/KL term.
