# Same Actor-update budget: terminal-gradient reliability in Lean AM

Primary question: after the same number N of Actor optimizer.step() calls, does Oracle terminal-gradient access yield higher true task reward, lower true stochastic-control J, or both? Oracle is a truth-assisted diagnostic, not a deployable algorithm or theoretical upper bound.

## Provenance and what was executed

This is a read-only reanalysis of `runs/02-flow-am-dim-formal-8q5obfng`. The original three tasks × three seeds × two Actor arms had already completed 300 updates each. No Actor, Base or reward model was retrained; zero new optimizer steps and zero new checkpoint evaluations were performed. The models and data remain in the source run. The new output directory contains analyses, figures, the source manifest and this report.

N is Actor parameter updates; K=64 is Flow EM discretization, B=64 is rollout paths per update. Both arms share N/K/B, Base, scalar data, reward ensemble, initial Actor, initial Adam state and pairable training/evaluation noise. Their state paths differ after policies diverge. The intervention is only the terminal label source. Lean AM, not full adjoint or BPTT, is used in both arms.

The complete learning curve uses N=[0, 50, 100, 150, 200, 250, 300] and 512 common evaluation paths per task-seed (noise role 300000). It was measured during original training. Only N=0,150,300 model/optimizer checkpoints were serialized; N=50,100,200,250 are real curve measurements, not reconstructed models. The independent N=300 final table uses 4096 different paths (noise role 700000); it is never spliced into the 512-path curve.

## Reward versus Actor updates (higher is better)

Each task has unsmoothed Learned/Oracle curves with every seed, their mean and training-seed SD, and a Frozen Base N=0 reference. See `fixed-updates-true-reward.png` and the exact 126 source curve rows.

## True SOC J versus Actor updates (lower is better)

The original objective remains `J=PathKL-beta E[R_true]`, beta=1.5. No post-hoc PathKL matching, actor scaling, checkpoint selection or extra energy loss was performed. See `fixed-updates-true-J.png`.

## Paired checkpoint effects

Define `Delta_R=(R_Oracle-R_Learned)/d`, `Delta_J=(J_Learned-J_Oracle)/d`, `Delta_KL=(KL_Oracle-KL_Learned)/d`. Positive Delta_R means higher Oracle task reward; positive Delta_J means lower Oracle true J. Every row obeys `Delta_J=beta*Delta_R-Delta_KL`. Pair within seed first, then aggregate. The table below is the pre-specified N=50/150/300 summary; `paired-updates-by-seed.csv` and `paired-updates-summary.csv` keep all three seeds and all seven update points.

| task | N | delta_reward_mean | delta_reward_seed_sd | reward_positive_seeds | delta_J_mean | delta_J_seed_sd | J_positive_seeds | delta_KL_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| independent_d2 | 50 | -0.000595679 | 0.00500424 | 1 | 0.00117249 | 0.00280596 | 1 | -0.00206601 |
| independent_d2 | 150 | -0.000921058 | 0.00634317 | 1 | 0.00280961 | 0.00273091 | 3 | -0.0041912 |
| independent_d2 | 300 | -0.00212015 | 0.00499566 | 1 | 0.00167515 | 0.00128695 | 3 | -0.00485537 |
| independent_d16 | 50 | -0.00117982 | 0.00643199 | 1 | 0.00716772 | 0.00208669 | 3 | -0.00893745 |
| independent_d16 | 150 | -0.00586507 | 0.0174841 | 1 | 0.0392393 | 0.0236701 | 3 | -0.0480369 |
| independent_d16 | 300 | -0.00348329 | 0.0143834 | 1 | 0.0471122 | 0.030575 | 3 | -0.0523372 |
| coupled_d16 | 50 | 0.00633778 | 0.00671833 | 3 | 0.0103798 | 0.00265728 | 3 | -0.000873177 |
| coupled_d16 | 150 | 0.0139497 | 0.0197375 | 2 | 0.0278932 | 0.00524061 | 3 | -0.00696869 |
| coupled_d16 | 300 | 0.0144107 | 0.0213006 | 2 | 0.0284243 | 0.0059186 | 3 | -0.0068082 |

Direction counts at the three highlighted checkpoints:

- independent_d2: N=50: reward 1/3, J 1/3 seeds favor Oracle; N=150: reward 1/3, J 3/3 seeds favor Oracle; N=300: reward 1/3, J 3/3 seeds favor Oracle.
- independent_d16: N=50: reward 1/3, J 3/3 seeds favor Oracle; N=150: reward 1/3, J 3/3 seeds favor Oracle; N=300: reward 1/3, J 3/3 seeds favor Oracle.
- coupled_d16: N=50: reward 3/3, J 3/3 seeds favor Oracle; N=150: reward 2/3, J 3/3 seeds favor Oracle; N=300: reward 2/3, J 3/3 seeds favor Oracle.

Direct fixed-N reading of this run:

- independent_d2: mean Delta_J/d is 0.0011725 at N=50 and 0.0016751 at N=300; the N=300 mean reward difference is -0.0021202. A small J advantage does not imply a reward advantage.
- independent_d16: mean Delta_J/d rises from 0.0071677 at N=50 to 0.047112 at N=300, while the mean N=300 Delta_R/d is -0.0034833 and Delta_KL/d is -0.052337. This is earlier and accumulating J separation, not evidence of consistently higher reward or a measured training-speed ratio.
- coupled_d16: mean Delta_R/d and Delta_J/d are 0.0063378 and 0.01038 at N=50, versus 0.014411 and 0.028424 at N=300; final curve directions are 2/3 for reward and 3/3 for J.

## Path KL as explanation, not a matching condition

See `fixed-updates-path-KL.png` and `delta_KL_mean` above. A lower J does not imply higher reward; a higher reward does not imply lower J. Differences in control intensity are allowed and explicitly accounted for by the original J.

## Independent high-sample N=300 final table

| task | seed | learned_reward_per_dim | oracle_reward_per_dim | learned_J_per_dim | oracle_J_per_dim | learned_KL_per_dim | oracle_KL_per_dim | delta_reward_per_dim | delta_J_per_dim | delta_KL_per_dim | final_J_reduction_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| independent_d2 | 38001 | -0.310733 | -0.307324 | 0.802701 | 0.798265 | 0.336602 | 0.337279 | 0.0034083 | 0.00443626 | 0.000676185 | 0.552666 |
| independent_d2 | 38002 | -0.300934 | -0.30278 | 0.824172 | 0.822637 | 0.37277 | 0.368466 | -0.00184601 | 0.00153481 | -0.00430383 | 0.186225 |
| independent_d2 | 38003 | -0.308525 | -0.315427 | 0.83108 | 0.830318 | 0.368292 | 0.357177 | -0.00690136 | 0.000762617 | -0.0111147 | 0.0917622 |
| independent_d16 | 38001 | -0.322585 | -0.331104 | 0.865962 | 0.839446 | 0.382085 | 0.342791 | -0.00851912 | 0.0265157 | -0.0392944 | 3.06199 |
| independent_d16 | 38002 | -0.318482 | -0.332068 | 0.927005 | 0.842774 | 0.449282 | 0.344672 | -0.0135863 | 0.0842307 | -0.10461 | 9.08632 |
| independent_d16 | 38003 | -0.347558 | -0.335889 | 0.899018 | 0.867726 | 0.377681 | 0.363892 | 0.0116692 | 0.0312923 | -0.0137886 | 3.48072 |
| coupled_d16 | 38001 | -0.431922 | -0.39564 | 0.99337 | 0.971532 | 0.345487 | 0.378072 | 0.0362821 | 0.0218378 | 0.0325854 | 2.19835 |
| coupled_d16 | 38002 | -0.405165 | -0.391598 | 0.990432 | 0.961307 | 0.382685 | 0.373911 | 0.0135672 | 0.0291247 | -0.00877387 | 2.94061 |
| coupled_d16 | 38003 | -0.398312 | -0.404147 | 1.01835 | 0.985099 | 0.420884 | 0.378879 | -0.00583512 | 0.0332531 | -0.0420058 | 3.26539 |

The final J reduction percentage is `(J_Learned-J_Oracle)/J_Learned×100` for this fixed checkpoint only. It is not an update-count saving or wall-clock acceleration. Final paired means, seed SD and direction counts:

| task | delta_reward_mean | delta_reward_seed_sd | reward_positive_seeds | delta_J_mean | delta_J_seed_sd | J_positive_seeds | delta_KL_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coupled_d16 | 0.0146714 | 0.0210803 | 2 | 0.0280719 | 0.00578006 | 3 | -0.00606475 |
| independent_d16 | -0.00347875 | 0.0133609 | 1 | 0.0473462 | 0.032032 | 3 | -0.0525644 |
| independent_d2 | -0.00177969 | 0.00515515 | 1 | 0.00224456 | 0.00193694 | 3 | -0.0049141 |

## Compact mechanism readout

| task | terminal_gradient_relative_rms | q_relative_difference | step_relative_difference |
| --- | --- | --- | --- |
| coupled_d16 | 0.306346 | 0.90783 | 0.580438 |
| independent_d16 | 0.404659 | 0.973097 | 0.655054 |
| independent_d2 | 0.113227 | 0.575841 | 0.324589 |

These existing common-path/shared-Adam-history diagnostics describe label, local matching-gradient and one-step optimizer discrepancies. They do not establish the true J policy gradient or equate the two independent training histories. A more accurate terminal gradient need not improve every reward checkpoint.

## Scope of the conclusion

The tables answer fixed-N performance separately for true reward and true J, including stages where their directions differ. Three explored training seeds and 300 steps do not establish a general convergence rate, asymptotic optimum, or causal isolation of critic error from Base quality and evolving state distributions. Future robust methods must use the same data, initial Actor, B, K and actual N optimizer steps; any extra forward/adjoint/critic computation must be logged separately. No such new method is implemented here.

Figures: fixed-updates-true-reward.png, fixed-updates-true-J.png, fixed-updates-path-KL.png, paired-fixed-update-differences.png.
