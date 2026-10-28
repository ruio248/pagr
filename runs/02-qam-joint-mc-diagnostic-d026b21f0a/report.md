# Joint TD+AM current-policy MC gradient diagnostic

This is a **post-hoc, fixed-snapshot exploratory diagnostic**, not a second joint-training run or an algorithm win. Training checkpoints are reused unchanged from `runs/02-qam-td-formal-31c55defef`. No retraining took place.

## Estimator and scope

For the first transition $y_{k+1}=y_k+Ba+\sigma_{env}\xi$, $\nabla_a Q^{\pi_\theta}=B^\top\mathbb E[\xi G]/\sigma_{env}$. All subsequent actions use the **saved current Actor**, not pi_ref. The terminal environment noise is Rao-Blackwellized when k<H-1; k=H-1 uses the sampled indicator. The leave-one-out covariance baseline preserves unbiasedness for the raw score expectation. Each point has 2 independent splits of 8192 rollouts; split differences and estimated standard errors measure reference uncertainty. The estimator is finite-sample, not exact Oracle.

States are selected from held-out evaluation transitions by a fixed RNG, one at k=0/4/7 per seed and checkpoint; the same state is reused at N=0/150/300. The Actor endpoint changes with N. The 3-state microbatch is not the actual B=64 training batch.

## Pointwise Q and action-gradient measurements

The gradient error norm should be compared with its MC split difference and SE; neither is a formal simultaneous confidence bound. At k=7, the exact policy-independent Q/gradient is used solely to calibrate the MC estimator. Fixed pi_ref Q is never used for k<7.

| seed | N | k | q_critic | q_mc | q_mc_se | gradient_critic_minus_mc_norm | gradient_mc_split_difference_norm | gradient_mc_se_l2 | terminal_exact_gradient_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 49001 | 0 | 0 | 0.3522 | 0.51 | 0.002883 | 0.175 | 0.1495 | 0.08053 | nan |
| 49001 | 0 | 4 | 0.08474 | 0.2598 | 0.002705 | 0.3427 | 0.1313 | 0.07522 | nan |
| 49001 | 0 | 7 | 0.1592 | 0.05994 | 0.001855 | 0.6565 | 0.1421 | 0.05595 | 0.04989 |
| 49001 | 150 | 0 | 0.5398 | 0.6763 | 0.002366 | 0.4232 | 0.1598 | 0.06616 | nan |
| 49001 | 150 | 4 | 0.1523 | 0.5761 | 0.002932 | 0.3337 | 0.1191 | 0.08162 | nan |
| 49001 | 150 | 7 | 0.6813 | 0.9902 | 0.0007707 | 0.9199 | 0.03746 | 0.02744 | 0.027 |
| 49001 | 300 | 0 | 0.5286 | 0.6995 | 0.002239 | 0.3748 | 0.1015 | 0.06248 | nan |
| 49001 | 300 | 4 | 0.2163 | 0.5399 | 0.00299 | 0.3587 | 0.1291 | 0.08346 | nan |
| 49001 | 300 | 7 | 0.2872 | 0.94 | 0.001855 | 0.5568 | 0.1326 | 0.05977 | 0.03175 |
| 49002 | 0 | 0 | 0.1774 | 0.286 | 0.002715 | 0.1937 | 0.1129 | 0.07521 | nan |
| 49002 | 0 | 4 | 0.1129 | 0.113 | 0.001912 | 0.2818 | 0.1101 | 0.05334 | nan |
| 49002 | 0 | 7 | -0.1903 | 0.0001831 | 0.0001057 | 0.2818 | 0.007332 | 0.00431 | 0.003792 |
| 49002 | 150 | 0 | 0.3099 | 0.5555 | 0.002746 | 0.2256 | 0.1178 | 0.07681 | nan |
| 49002 | 150 | 4 | 0.1642 | 0.376 | 0.002978 | 0.4437 | 0.1363 | 0.08296 | nan |
| 49002 | 150 | 7 | -0.08689 | 0 | 0 | 0.4151 | 0 | 0 | 6.6e-05 |
| 49002 | 300 | 0 | 0.2482 | 0.5282 | 0.002799 | 0.292 | 0.1943 | 0.07826 | nan |
| 49002 | 300 | 4 | 0.2298 | 0.1592 | 0.002239 | 0.3623 | 0.1097 | 0.06208 | nan |
| 49002 | 300 | 7 | -0.03413 | 0 | 0 | 0.5073 | 0 | 0 | 4.329e-05 |
| 49003 | 0 | 0 | 0.2851 | 0.5954 | 0.002683 | 0.187 | 0.1554 | 0.07511 | nan |
| 49003 | 0 | 4 | 0.1339 | 0.3161 | 0.002885 | 0.4641 | 0.2208 | 0.08031 | nan |
| 49003 | 0 | 7 | 0.1111 | 0 | 0 | 0.1025 | 0 | 0 | 7.019e-09 |
| 49003 | 150 | 0 | 0.2042 | 0.7185 | 0.00212 | 0.1999 | 0.1208 | 0.05914 | nan |
| 49003 | 150 | 4 | 0.2049 | 0.6676 | 0.002729 | 0.3404 | 0.1332 | 0.07647 | nan |
| 49003 | 150 | 7 | 0.2009 | 0 | 0 | 0.1857 | 0 | 0 | 6.449e-08 |
| 49003 | 300 | 0 | 0.3251 | 0.7604 | 0.001826 | 0.2278 | 0.1139 | 0.05099 | nan |
| 49003 | 300 | 4 | 0.2187 | 0.642 | 0.002805 | 0.3201 | 0.18 | 0.07859 | nan |
| 49003 | 300 | 7 | 0.1112 | 0 | 0 | 0.2106 | 0 | 0 | 5.575e-06 |

Across these selected points, critic Q mean absolute error versus MC is 0.2381, while mean MC Q is 0.4063. The action-gradient discrepancy exceeds the two-split gap at 27/27 points. This is **not** the desired clean case of accurate Q values but inaccurate gradients: value errors are also substantial, and the point selection is small.

## Propagation and virtual update

Both gradients enter the same frozen-base Lean adjoint, same Flow paths and same actor optimizer state. The split-reference parameter and Adam differences indicate how much these downstream comparisons themselves move under MC noise. Virtual steps do not alter saved checkpoints.

| seed | N | gradient_error_rms | gradient_mc_split_rms | matching_label_difference_rms | matching_parameter_gradient_difference_norm | matching_parameter_gradient_mc_split_norm | adam_step_difference_norm | adam_step_mc_split_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 49001 | 0 | 0.1098 | 0.03529 | 0.2705 | 0.3174 | 0.1104 | 0.04502 | 0.02645 |
| 49001 | 150 | 0.1539 | 0.02927 | 0.3636 | 0.5109 | 0.08672 | 0.02315 | 0.00456 |
| 49001 | 300 | 0.1098 | 0.03046 | 0.2728 | 0.3491 | 0.09534 | 0.01703 | 0.004767 |
| 49002 | 0 | 0.06395 | 0.02278 | 0.1636 | 0.2363 | 0.08635 | 0.05823 | 0.03302 |
| 49002 | 150 | 0.09355 | 0.02601 | 0.2368 | 0.3376 | 0.1024 | 0.01795 | 0.005272 |
| 49002 | 300 | 0.09936 | 0.0322 | 0.263 | 0.3515 | 0.108 | 0.01987 | 0.005557 |
| 49003 | 0 | 0.07372 | 0.03897 | 0.1862 | 0.2054 | 0.1052 | 0.05492 | 0.03832 |
| 49003 | 150 | 0.06297 | 0.02596 | 0.1693 | 0.1966 | 0.06952 | 0.0109 | 0.003926 |
| 49003 | 300 | 0.06434 | 0.03074 | 0.1729 | 0.1761 | 0.08263 | 0.009748 | 0.004615 |

## Paired closed-loop outcome after one virtual step

The same initial states, Flow noise, and environment noise are shared across the unmodified, learned-step and MC-reference-step actors. Each row is one 3-state microbatch intervention, **not** a full retraining comparison. Delta is MC-reference minus learned; the interval is the uncorrected paired episode mean ±1.96 SE. Episodes are not independent training seeds.

| seed | N | success_before | success_learned | success_mc_reference | delta_success_reference_minus_learned | delta_success_se | eval_episodes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 49001 | 0 | 0.3604 | 0.3647 | 0.3657 | 0.0009766 | 0.002584 | 2048 |
| 49001 | 150 | 0.6592 | 0.6587 | 0.6602 | 0.001465 | 0.001891 | 2048 |
| 49001 | 300 | 0.6675 | 0.667 | 0.6753 | 0.008301 | 0.002335 | 2048 |
| 49002 | 0 | 0.3477 | 0.3511 | 0.355 | 0.003906 | 0.003239 | 2048 |
| 49002 | 150 | 0.6431 | 0.6416 | 0.645 | 0.003418 | 0.001618 | 2048 |
| 49002 | 300 | 0.6558 | 0.6558 | 0.6577 | 0.001953 | 0.002072 | 2048 |
| 49003 | 0 | 0.3613 | 0.3672 | 0.3667 | -0.0004883 | 0.00263 | 2048 |
| 49003 | 150 | 0.6406 | 0.6421 | 0.6431 | 0.0009766 | 0.001381 | 2048 |
| 49003 | 300 | 0.6787 | 0.6792 | 0.6797 | 0.0004883 | 0.00162 | 2048 |

Positive paired point estimates: 8/9; uncorrected intervals excluding zero: 2/9. These are descriptive, with no multiple-testing correction and no claim that the reference is a deployable algorithm. A one-step microbatch intervention is not evidence of better final training.

## Interpretation limits

This shows whether the *measured current-policy* gradient discrepancy survives adjoint propagation and changes an Actor update. It does not by itself establish a repeatable gain in final trained performance. The MC reference needs a simulator and known Gaussian transition, uses many more rollouts than ordinary TD+AM, and can remain noisy for sparse terminal rewards. Baseline TD uses fixed offline transitions, so distribution shift is also possible. No true Q for the evolving current Actor is asserted.
