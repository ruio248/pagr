# Terminal-gradient dimension × generative coupling: executed exploratory run

Fixed tasks: ['independent_d2', 'independent_d16', 'coupled_d16']; seeds [38001, 38002, 38003]. Each task-seed uses 4096 behavior training actions, 512 noisy scalar reward labels, 5 reward models, 2500 FM updates and 300 updates per actor. The three seed numbers are new to this repository but this is exploratory.

## Direct answer

Final learned-checkpoint terminal-gradient relative RMS means: A=0.1132, B=0.4047, C=0.3063. Oracle gain G=(J_learned-J_oracle)/d means: A=0.0022446, B=0.047346, C=0.028072. Positive G favors Oracle on the same true reward-versus-KL objective.

Dimension increase coincides with a larger measured terminal-gradient error in this run.
The coupled setting has a larger measured Base Jacobian off-diagonal fraction than independent d16.
The d16 independent Oracle gain exceeds d2 in this run.
The d16 coupled Oracle gain does not exceed d16 independent in this run.
These comparisons do not isolate adjoint propagation from reward-model fit, state distribution, or pretrained Base quality. Three training seeds support an initial diagnosis, not a general dimension law.

## Every final seed and method

| task | seed | method | cost | cost_per_dim | reward_per_dim | energy_per_dim | kl_identity_max_abs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| independent_d2 | 38001 | Frozen Base | 3.47731 | 1.73866 | -1.1591 | 0 | 0 |
| independent_d2 | 38001 | learned | 1.6054 | 0.802701 | -0.310733 | 0.336602 | 5.72205e-06 |
| independent_d2 | 38001 | oracle | 1.59653 | 0.798265 | -0.307324 | 0.337279 | 5.60284e-06 |
| independent_d2 | 38002 | Frozen Base | 3.69361 | 1.8468 | -1.2312 | 0 | 0 |
| independent_d2 | 38002 | learned | 1.64834 | 0.824172 | -0.300934 | 0.37277 | 6.4373e-06 |
| independent_d2 | 38002 | oracle | 1.64527 | 0.822637 | -0.30278 | 0.368466 | 8.46386e-06 |
| independent_d2 | 38003 | Frozen Base | 3.67069 | 1.83535 | -1.22356 | 0 | 0 |
| independent_d2 | 38003 | learned | 1.66216 | 0.83108 | -0.308525 | 0.368292 | 7.03335e-06 |
| independent_d2 | 38003 | oracle | 1.66064 | 0.830318 | -0.315427 | 0.357177 | 7.03335e-06 |
| independent_d16 | 38001 | Frozen Base | 29.721 | 1.85756 | -1.23838 | 0 | 0 |
| independent_d16 | 38001 | learned | 13.8554 | 0.865962 | -0.322585 | 0.382085 | 1.14441e-05 |
| independent_d16 | 38001 | oracle | 13.4311 | 0.839446 | -0.331104 | 0.342791 | 1.28746e-05 |
| independent_d16 | 38002 | Frozen Base | 29.5629 | 1.84768 | -1.23179 | 0 | 0 |
| independent_d16 | 38002 | learned | 14.8321 | 0.927005 | -0.318482 | 0.449282 | 1.43051e-05 |
| independent_d16 | 38002 | oracle | 13.4844 | 0.842774 | -0.332068 | 0.344672 | 1.04904e-05 |
| independent_d16 | 38003 | Frozen Base | 31.0449 | 1.94031 | -1.29354 | 0 | 0 |
| independent_d16 | 38003 | learned | 14.3843 | 0.899018 | -0.347558 | 0.377681 | 1.43051e-05 |
| independent_d16 | 38003 | oracle | 13.8836 | 0.867726 | -0.335889 | 0.363892 | 1.14441e-05 |
| coupled_d16 | 38001 | Frozen Base | 30.9608 | 1.93505 | -1.29003 | 0 | 0 |
| coupled_d16 | 38001 | learned | 15.8939 | 0.99337 | -0.431922 | 0.345487 | 1.43051e-05 |
| coupled_d16 | 38001 | oracle | 15.5445 | 0.971532 | -0.39564 | 0.378072 | 1.33514e-05 |
| coupled_d16 | 38002 | Frozen Base | 31.5502 | 1.97189 | -1.31459 | 0 | 0 |
| coupled_d16 | 38002 | learned | 15.8469 | 0.990432 | -0.405165 | 0.382685 | 1.23978e-05 |
| coupled_d16 | 38002 | oracle | 15.3809 | 0.961307 | -0.391598 | 0.373911 | 1.04904e-05 |
| coupled_d16 | 38003 | Frozen Base | 31.5477 | 1.97173 | -1.31449 | 0 | 0 |
| coupled_d16 | 38003 | learned | 16.2936 | 1.01835 | -0.398312 | 0.420884 | 1.04904e-05 |
| coupled_d16 | 38003 | oracle | 15.7616 | 0.985099 | -0.404147 | 0.378879 | 1.33514e-05 |

## Task means and training-seed SD

| task | method | cost_mean | cost_std | cost_per_dim_mean | cost_per_dim_std | reward_per_dim_mean | reward_per_dim_std | energy_per_dim_mean | energy_per_dim_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coupled_d16 | Frozen Base | 31.3529 | 0.339593 | 1.95956 | 0.0212246 | -1.30637 | 0.0141497 | 0 | 0 |
| coupled_d16 | learned | 16.0115 | 0.245478 | 1.00072 | 0.0153424 | -0.4118 | 0.0177601 | 0.383019 | 0.0376999 |
| coupled_d16 | oracle | 15.5623 | 0.190963 | 0.972646 | 0.0119352 | -0.397128 | 0.00640581 | 0.376954 | 0.00266613 |
| independent_d16 | Frozen Base | 30.1096 | 0.813841 | 1.88185 | 0.0508651 | -1.25457 | 0.03391 | 0 | 0 |
| independent_d16 | learned | 14.3573 | 0.488904 | 0.897328 | 0.0305565 | -0.329541 | 0.0157373 | 0.403016 | 0.0401283 |
| independent_d16 | oracle | 13.5997 | 0.2473 | 0.849982 | 0.0154563 | -0.33302 | 0.00253082 | 0.350452 | 0.0116777 |
| independent_d2 | Frozen Base | 3.61387 | 0.118816 | 1.80694 | 0.0594082 | -1.20462 | 0.0396055 | 0 | 0 |
| independent_d2 | learned | 1.63864 | 0.029598 | 0.819318 | 0.014799 | -0.306731 | 0.00513976 | 0.359221 | 0.0197163 |
| independent_d2 | oracle | 1.63415 | 0.0334698 | 0.817073 | 0.0167349 | -0.308511 | 0.00640609 | 0.354307 | 0.0157907 |

## Per-seed Oracle gains and Oracle versus Frozen

| task | seed | oracle_gain_J | oracle_gain_J_per_dim | oracle_gain_over_frozen_J |
| --- | --- | --- | --- | --- |
| coupled_d16 | 38001 | 0.349404 | 0.0218378 | 15.4162 |
| coupled_d16 | 38002 | 0.465995 | 0.0291247 | 16.1693 |
| coupled_d16 | 38003 | 0.53205 | 0.0332531 | 15.7861 |
| independent_d16 | 38001 | 0.424251 | 0.0265157 | 16.2899 |
| independent_d16 | 38002 | 1.34769 | 0.0842307 | 16.0785 |
| independent_d16 | 38003 | 0.500677 | 0.0312923 | 17.1613 |
| independent_d2 | 38001 | 0.00887252 | 0.00443626 | 1.88078 |
| independent_d2 | 38002 | 0.00306962 | 0.00153481 | 2.04833 |
| independent_d2 | 38003 | 0.00152523 | 0.000762617 | 2.01006 |

Task mean and seed SD of the gains:

| task | oracle_gain_J_per_dim_mean | oracle_gain_J_per_dim_std | oracle_gain_over_frozen_J_mean | oracle_gain_over_frozen_J_std |
| --- | --- | --- | --- | --- |
| coupled_d16 | 0.0280719 | 0.00578006 | 15.7905 | 0.376532 |
| independent_d16 | 0.0473462 | 0.032032 | 16.5099 | 0.573944 |
| independent_d2 | 0.00224456 | 0.00193694 | 1.97972 | 0.0877976 |

An Oracle reward increase alone is insufficient: the cost column includes the exact finite-chain KL energy. If Oracle cannot improve Frozen on a task, Base quality or finite optimization limits are immediate constraints.

## Reward fit and propagation mechanism

| task | value_rmse_over_true_reward_std | terminal_gradient_relative_rms | q_relative_difference | step_relative_difference |
| --- | --- | --- | --- | --- |
| coupled_d16 | 0.397588 | 0.306346 | 0.90783 | 0.580438 |
| independent_d16 | 0.883652 | 0.404659 | 0.973097 | 0.655054 |
| independent_d2 | 0.130351 | 0.113227 | 0.575841 | 0.324589 |

The full diagnostics CSV also contains gradient cosines/norm ratios, zero-scale flags, raw value RMSE, and 8 separate rollout repeats at the initial and learned-final checkpoints. The adjoint-time CSV reports both absolute error and correct signal RMS at four fixed times. The shared-history virtual Adam step does not predict the entire independent training trajectory.

| task | base_jacobian_offdiag_fraction | propagator_sigma_max | propagator_sigma_min |
| --- | --- | --- | --- |
| coupled_d16 | 0.374569 | 0.0726919 | 0.0161914 |
| independent_d16 | 0.181862 | 0.0465112 | 0.0250888 |
| independent_d2 | 0.0739989 | 0.0350289 | 0.0310359 |

A singular value below one means contraction for that measured direction; no amplification threshold was imposed.

## Pretrained Base quality

| task | base_step0_fm_loss_per_dim | base_heldout_fm_loss_per_dim | ode_random_projection_w2 | sde_random_projection_w2 | ode_adjacent_corr_gap_rms | sde_adjacent_corr_gap_rms |
| --- | --- | --- | --- | --- | --- | --- |
| coupled_d16 | 2.11878 | 1.64387 | 0.064501 | 0.0766385 | 0.032939 | 0.0320641 |
| independent_d16 | 2.01598 | 1.57559 | 0.0683789 | 0.0771563 | 0.0397914 | 0.0365075 |
| independent_d2 | 1.9699 | 1.55559 | 0.0743807 | 0.0824918 | 0.0416425 | 0.0383604 |

Each seed has per-coordinate mean/variance and adjacent-correlation arrays, paired behavior/Base ODE/Base SDE samples, and inverse-transform Gaussian checks. Random-projection W2 and 2D plots are partial fit diagnostics, not full high-dimensional distribution metrics.

## Fixed training and measurement budgets

| task | seed | source | updates | training_rollouts | stochastic_velocity_calls | warmup_velocity_calls | oracle_gradient_points | clipping_count | wall_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| independent_d2 | 38001 | learned | 300 | 19200 | 1.2288e+06 | 307200 | 0 | 1 | 6.0766 |
| independent_d2 | 38001 | oracle | 300 | 19200 | 1.2288e+06 | 307200 | 19200 | 1 | 5.79938 |
| independent_d2 | 38002 | learned | 300 | 19200 | 1.2288e+06 | 307200 | 0 | 3 | 6.50607 |
| independent_d2 | 38002 | oracle | 300 | 19200 | 1.2288e+06 | 307200 | 19200 | 2 | 5.66193 |
| independent_d2 | 38003 | learned | 300 | 19200 | 1.2288e+06 | 307200 | 0 | 1 | 5.72031 |
| independent_d2 | 38003 | oracle | 300 | 19200 | 1.2288e+06 | 307200 | 19200 | 0 | 5.58561 |
| independent_d16 | 38001 | learned | 300 | 19200 | 1.2288e+06 | 307200 | 0 | 0 | 5.87317 |
| independent_d16 | 38001 | oracle | 300 | 19200 | 1.2288e+06 | 307200 | 19200 | 0 | 7.03083 |
| independent_d16 | 38002 | learned | 300 | 19200 | 1.2288e+06 | 307200 | 0 | 0 | 7.48909 |
| independent_d16 | 38002 | oracle | 300 | 19200 | 1.2288e+06 | 307200 | 19200 | 0 | 7.03808 |
| independent_d16 | 38003 | learned | 300 | 19200 | 1.2288e+06 | 307200 | 0 | 0 | 6.09735 |
| independent_d16 | 38003 | oracle | 300 | 19200 | 1.2288e+06 | 307200 | 19200 | 0 | 5.95872 |
| coupled_d16 | 38001 | learned | 300 | 19200 | 1.2288e+06 | 307200 | 0 | 0 | 5.93419 |
| coupled_d16 | 38001 | oracle | 300 | 19200 | 1.2288e+06 | 307200 | 19200 | 0 | 5.82284 |
| coupled_d16 | 38002 | learned | 300 | 19200 | 1.2288e+06 | 307200 | 0 | 0 | 7.32198 |
| coupled_d16 | 38002 | oracle | 300 | 19200 | 1.2288e+06 | 307200 | 19200 | 0 | 7.34489 |
| coupled_d16 | 38003 | learned | 300 | 19200 | 1.2288e+06 | 307200 | 0 | 0 | 6.58901 |
| coupled_d16 | 38003 | oracle | 300 | 19200 | 1.2288e+06 | 307200 | 19200 | 0 | 5.68509 |

FM loss and matching loss are divided by d. Reward and energy are both left in the same true objective J; division by d in summaries does not alter beta. The observed-data noise scale was calibrated on an independent fixed behavior sample for each task-seed before reward fitting. Learned actor updates were run with the truth API blocked.

## Limits and next question

This first round changes both reward-model input distribution and the learned Flow when rho changes. It cannot assign a larger B/C gap uniquely to the adjoint. An optional oracle-assisted equal-error diagnostic is specified in the config but disabled; no kappa intervention was run. The previous four-mode 2D negative result is retained separately.

Figures: behavior-base-projections.png, final-reward-KL-J.png, terminal-gradient-to-Adam.png, oracle-gain-and-propagation.png. All data and checkpoints reside in this unique run.
