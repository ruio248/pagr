# TD-Critic terminal-gradient experiment

Profile: **formal 3-seed exploratory**. Fixed task d=16, H=8, sigma_env=0.15, gamma=0.98, beta=10.0, K=64, Actor B=64.

## Scope and references

A: fixed pi_ref, TD learning Q^{pi_ref}, frozen critic, paired Learned/Oracle Lean AM. B: current Actor next-action TD backup plus alternating AM. B has no exact current-policy Oracle. Neither is a full official benchmark reproduction, and no new robustification method is claimed.

Official qam.py `main` resolved to commit `2726d767c9a0a7a46d49693f0391f73dc2cf58ac`; critic_loss uses TD target and ensemble, adj_matching initializes with negative critic action gradient, actor_loss matches the propagated lean label, sample_actions deploys ODE actions with clipping and possible best-of-n. This run instead uses a shared finite-epsilon all-Gaussian SDE sampler, no clipping.

## Offline data and reference value

| seed | episodes | transitions | successes | nonzero_reward_fraction | k0_count | k1_count | k2_count | k3_count | k4_count | k5_count | k6_count | k7_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 49001 | 512 | 4096 | 113 | 0.027588 | 512 | 512 | 512 | 512 | 512 | 512 | 512 | 512 |
| 49002 | 512 | 4096 | 107 | 0.026123 | 512 | 512 | 512 | 512 | 512 | 512 | 512 | 512 |
| 49003 | 512 | 4096 | 108 | 0.026367 | 512 | 512 | 512 | 512 | 512 | 512 | 512 | 512 |

Goal radius 3.00004 was set once from the fixed reference terminal-state 20% quantile before formal seeds. Analytic Q and gradient are verified against Monte Carlo, multiple finite-difference h and the Bellman relation; see oracle-checks.csv. At k<7, immediate r=0 while Q_ref varies with action; see future-value-check.json.

## A: critic diagnostics and fixed-N mechanism

The critic was trained only from observed transitions with TD target and pi_ref next actions; Q_ref was not a target or checkpoint-selection criterion. Final checkpoint is fixed.

| seed | oracle_q_rmse | validation_bellman_rmse | value_out_of_range_fraction | q_mean | q_std | oracle_q_mean |
| --- | --- | --- | --- | --- | --- | --- |
| 49001 | 0.16858 | 0.15115 | 0.10352 | 0.19121 | 0.20946 | 0.19095 |
| 49002 | 0.16725 | 0.16548 | 0.13379 | 0.19341 | 0.2096 | 0.17433 |
| 49003 | 0.16634 | 0.15816 | 0.11914 | 0.18472 | 0.20868 | 0.17672 |

On common paths at N=0/150/300, `delta p_1=-beta(grad Qhat-grad Qref)` was independently propagated and compared with the difference of two full lean recursions. The actual matching-label difference is `nu_j delta p_{j+1}`; the parameter difference was checked against the Actor-Jacobian projection and identical-history virtual Adam steps were cloned. See per-seed NPZ and table.

| seed | N | q_value_rmse | gradient_abs_rms | gradient_rel_rms | gradient_cosine | matching_gradient_difference_norm | adam_step_difference_norm | recurrence_max_error | parameter_projection_max_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 49001 | 0 | 0.18323 | 0.12785 | 0.94618 | 0.59993 | 0.1031 | 0.051598 | 1.6689e-06 | 1.3504e-08 |
| 49001 | 150 | 0.16961 | 0.10498 | 1.1853 | 0.61661 | 0.10694 | 0.0061198 | 1.3113e-06 | 2.1886e-08 |
| 49001 | 300 | 0.17579 | 0.10474 | 0.94059 | 0.61737 | 0.10246 | 0.0061545 | 1.6689e-06 | 1.9092e-08 |
| 49002 | 0 | 0.14613 | 0.1155 | 1.1291 | 0.59896 | 0.10787 | 0.052816 | 1.3113e-06 | 3.2596e-08 |
| 49002 | 150 | 0.27034 | 0.10894 | 0.98705 | 0.61192 | 0.10878 | 0.0069904 | 1.9073e-06 | 2.8871e-08 |
| 49002 | 300 | 0.26662 | 0.10597 | 1.0896 | 0.60617 | 0.11356 | 0.0073587 | 1.9073e-06 | 2.0023e-08 |
| 49003 | 0 | 0.2538 | 0.14538 | 0.98076 | 0.64279 | 0.10208 | 0.050599 | 1.9073e-06 | 1.8568e-08 |
| 49003 | 150 | 0.17679 | 0.11172 | 1.2759 | 0.65344 | 0.10621 | 0.0063915 | 1.4305e-06 | 1.3504e-08 |
| 49003 | 300 | 0.17364 | 0.11059 | 0.98534 | 0.6601 | 0.10245 | 0.006623 | 9.5367e-07 | 1.3504e-08 |

## A: paired fixed-N outcomes

J_extract = mean[path KL - beta Q_ref] on shared independent states; it is not environment return. A closed-loop success is an actual eight-step episode with fresh actions at each k.

| seed | N | method | extract_J | extract_KL | extract_Q_ref | success_rate | discounted_return | eval_episodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 49001 | 0 | learned | -1.732 | 0 | 0.1732 | 0.39844 | 0.34589 | 128 |
| 49001 | 0 | oracle | -1.732 | 0 | 0.1732 | 0.39844 | 0.34589 | 128 |
| 49001 | 50 | learned | -1.8125 | 0.15914 | 0.19716 | 0.5 | 0.43406 | 128 |
| 49001 | 50 | oracle | -1.8767 | 0.13145 | 0.20081 | 0.5625 | 0.48832 | 128 |
| 49001 | 100 | learned | -1.8588 | 0.14949 | 0.20083 | 0.57031 | 0.4951 | 128 |
| 49001 | 100 | oracle | -1.9361 | 0.12954 | 0.20657 | 0.625 | 0.54258 | 128 |
| 49001 | 150 | learned | -1.796 | 0.16156 | 0.19575 | 0.60156 | 0.52223 | 128 |
| 49001 | 150 | oracle | -1.8631 | 0.14927 | 0.20124 | 0.71875 | 0.62397 | 128 |
| 49001 | 200 | learned | -1.8563 | 0.16242 | 0.20188 | 0.53906 | 0.46797 | 128 |
| 49001 | 200 | oracle | -1.9291 | 0.14937 | 0.20785 | 0.60938 | 0.52901 | 128 |
| 49001 | 250 | learned | -1.8851 | 0.15911 | 0.20442 | 0.58594 | 0.50867 | 128 |
| 49001 | 250 | oracle | -1.967 | 0.14129 | 0.21083 | 0.67969 | 0.59005 | 128 |
| 49001 | 300 | learned | -1.8694 | 0.17846 | 0.20478 | 0.57031 | 0.4951 | 128 |
| 49001 | 300 | oracle | -1.965 | 0.12228 | 0.20873 | 0.66406 | 0.57649 | 128 |
| 49002 | 0 | learned | -2.0129 | 0 | 0.20129 | 0.375 | 0.32555 | 128 |
| 49002 | 0 | oracle | -2.0129 | 0 | 0.20129 | 0.375 | 0.32555 | 128 |
| 49002 | 50 | learned | -1.9697 | 0.13675 | 0.21064 | 0.57812 | 0.50189 | 128 |
| 49002 | 50 | oracle | -2.026 | 0.13338 | 0.21594 | 0.66406 | 0.57649 | 128 |
| 49002 | 100 | learned | -1.9634 | 0.15354 | 0.2117 | 0.51562 | 0.44763 | 128 |
| 49002 | 100 | oracle | -2.0303 | 0.12357 | 0.21538 | 0.58594 | 0.50867 | 128 |
| 49002 | 150 | learned | -1.9634 | 0.14901 | 0.21124 | 0.58594 | 0.50867 | 128 |
| 49002 | 150 | oracle | -2.0362 | 0.13682 | 0.2173 | 0.64844 | 0.56293 | 128 |
| 49002 | 200 | learned | -2.1035 | 0.15914 | 0.22626 | 0.55469 | 0.48154 | 128 |
| 49002 | 200 | oracle | -2.1711 | 0.11826 | 0.22893 | 0.63281 | 0.54936 | 128 |
| 49002 | 250 | learned | -1.8945 | 0.17643 | 0.20709 | 0.60156 | 0.52223 | 128 |
| 49002 | 250 | oracle | -1.9668 | 0.15161 | 0.21184 | 0.70312 | 0.6104 | 128 |
| 49002 | 300 | learned | -2.1364 | 0.14967 | 0.22861 | 0.67969 | 0.59005 | 128 |
| 49002 | 300 | oracle | -2.2075 | 0.12362 | 0.23312 | 0.74219 | 0.64431 | 128 |
| 49003 | 0 | learned | -2.1133 | 0 | 0.21133 | 0.38281 | 0.33233 | 128 |
| 49003 | 0 | oracle | -2.1133 | 0 | 0.21133 | 0.38281 | 0.33233 | 128 |
| 49003 | 50 | learned | -2.4311 | 0.14503 | 0.25761 | 0.63281 | 0.54936 | 128 |
| 49003 | 50 | oracle | -2.4861 | 0.12177 | 0.26078 | 0.67969 | 0.59005 | 128 |
| 49003 | 100 | learned | -2.2117 | 0.11965 | 0.23314 | 0.48438 | 0.4205 | 128 |
| 49003 | 100 | oracle | -2.2631 | 0.12358 | 0.23867 | 0.57812 | 0.50189 | 128 |
| 49003 | 150 | learned | -2.2874 | 0.14972 | 0.24372 | 0.58594 | 0.50867 | 128 |
| 49003 | 150 | oracle | -2.3456 | 0.11233 | 0.24579 | 0.63281 | 0.54936 | 128 |
| 49003 | 200 | learned | -2.4921 | 0.12683 | 0.2619 | 0.60156 | 0.52223 | 128 |
| 49003 | 200 | oracle | -2.5489 | 0.10879 | 0.26577 | 0.64062 | 0.55614 | 128 |
| 49003 | 250 | learned | -2.2088 | 0.12654 | 0.23353 | 0.59375 | 0.51545 | 128 |
| 49003 | 250 | oracle | -2.2591 | 0.10977 | 0.23689 | 0.64844 | 0.56293 | 128 |
| 49003 | 300 | learned | -2.348 | 0.13289 | 0.24809 | 0.49219 | 0.42728 | 128 |
| 49003 | 300 | oracle | -2.4183 | 0.11646 | 0.25348 | 0.59375 | 0.51545 | 128 |

Final paired positive deltaJ = Learned J_extract - Oracle J_extract; positive deltaSuccess = Oracle success - Learned success:

| seed | deltaJ | deltaSuccess |
| --- | --- | --- |
| 49001 | 0.095635 | 0.09375 |
| 49002 | 0.0711 | 0.0625 |
| 49003 | 0.070296 | 0.10156 |

Final exploratory mean deltaJ=0.0790102 (3/3 positive); mean deltaSuccess=0.0859375 (3/3 positive). Evaluation episodes are not independent training replicates; seed is the comparison unit. No best-N choice or conclusion of general superiority is made.

## B: minimal joint TD+AM

A-pretrained critic initializes B; every B step uses the **current Actor** to generate its TD next action with the same SDE sampler, updates critic once, then computes a fresh learned-Q action gradient and one Lean-AM Actor update. The following is real closed-loop evaluation only. Q^{pi_ref} is **not** an Oracle for this evolving policy.

| seed | N | success_rate | discounted_return | eval_episodes |
| --- | --- | --- | --- | --- |
| 49001 | 0 | 0.39844 | 0.34589 | 128 |
| 49001 | 50 | 0.57812 | 0.50189 | 128 |
| 49001 | 100 | 0.65625 | 0.56971 | 128 |
| 49001 | 150 | 0.6875 | 0.59684 | 128 |
| 49001 | 200 | 0.60938 | 0.52901 | 128 |
| 49001 | 250 | 0.67969 | 0.59005 | 128 |
| 49001 | 300 | 0.66406 | 0.57649 | 128 |
| 49002 | 0 | 0.375 | 0.32555 | 128 |
| 49002 | 50 | 0.65625 | 0.56971 | 128 |
| 49002 | 100 | 0.60156 | 0.52223 | 128 |
| 49002 | 150 | 0.625 | 0.54258 | 128 |
| 49002 | 200 | 0.60156 | 0.52223 | 128 |
| 49002 | 250 | 0.67188 | 0.58327 | 128 |
| 49002 | 300 | 0.79688 | 0.69179 | 128 |
| 49003 | 0 | 0.38281 | 0.33233 | 128 |
| 49003 | 50 | 0.67969 | 0.59005 | 128 |
| 49003 | 100 | 0.57812 | 0.50189 | 128 |
| 49003 | 150 | 0.63281 | 0.54936 | 128 |
| 49003 | 200 | 0.69531 | 0.60362 | 128 |
| 49003 | 250 | 0.71094 | 0.61718 | 128 |
| 49003 | 300 | 0.61719 | 0.5358 | 128 |

Joint final mean success rate=0.692708; these B results must not be mixed with A paired Oracle contrasts. B demonstrates the TD→Q-gradient→Lean-adjoint call chain, not a Q-gradient-error measurement against an exact current-policy value.

## Limits

Three exploratory seeds, finite data, fixed critic updates and finite-epsilon discretization do not prove a universal bottleneck. Oracle-Qref in A is a fixed-policy evaluator, not Q*. Oracle matching gradient is not generally the full true policy gradient. Critic fit, optimization, rollout distribution and finite Monte Carlo may all limit outcome differences. No robust guard or new algorithm was tested.
