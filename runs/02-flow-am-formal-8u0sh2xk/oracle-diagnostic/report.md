# Oracle AM gain-space diagnosis — executed Notebook 2 extension

Existing development seeds [28001, 28002, 28003]; K32, 300 Adam updates, beta=1.5. Each checkpoint/batch uses 32 independent training-noise probes and 1024 fresh common-noise paths per virtual pair. These probes are not independent training seeds or confirmatory trials.

## Direct answer

The clearest observed pattern is an Oracle reward gain accompanied by additional path KL/energy. Its net J effect must be judged separately; the present three examined seeds do not isolate label error from sampling and optimization limitations.

Equal-update paired J oracle gain: A64=-0.000552024 (descriptive 95% t interval [-0.0127068, 0.0116028]); A128=-0.00413983 ([-0.00865019, 0.000370524]). Positive means lower true J for Oracle. Intervals summarize training-seed variation, not 4096 evaluation paths or 32 probes as independent seeds.

Oracle reward effects: B64=0.0122078, B128=0.0192343; energy effects (Oracle minus learned): B64=0.0188638, B128=0.0329913. True J = full finite-chain path KL − beta × true reward.

## Four arms at fixed update 300

| seed | method | reward | reward_mc_se | energy | energy_mc_se | cost | cost_mc_se | terminal_shift_from_frozen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 28001 | AM-learned-B64 | 1.15401 | 0.00850324 | 0.329502 | 0.00278755 | -1.40152 | 0.0143453 | 1.25204 |
| 28001 | AM-oracle-B64 | 1.16614 | 0.00841384 | 0.350895 | 0.00296739 | -1.39831 | 0.014377 | 1.29055 |
| 28001 | AM-learned-B128 | 1.1779 | 0.00837676 | 0.358328 | 0.00312967 | -1.40852 | 0.0142013 | 1.31786 |
| 28001 | AM-oracle-B128 | 1.18458 | 0.00834305 | 0.371012 | 0.00327371 | -1.40585 | 0.0141498 | 1.33317 |
| 28002 | AM-learned-B64 | 1.1896 | 0.00836839 | 0.283905 | 0.00244267 | -1.50049 | 0.0138562 | 1.18185 |
| 28002 | AM-oracle-B64 | 1.2054 | 0.0083962 | 0.302513 | 0.00246852 | -1.50559 | 0.0138374 | 1.19281 |
| 28002 | AM-learned-B128 | 1.21449 | 0.00832237 | 0.314783 | 0.0025337 | -1.50696 | 0.0137629 | 1.25168 |
| 28002 | AM-oracle-B128 | 1.25352 | 0.00819652 | 0.376908 | 0.00305378 | -1.50337 | 0.0137945 | 1.33949 |
| 28003 | AM-learned-B64 | 1.17899 | 0.00856627 | 0.272156 | 0.00236568 | -1.49633 | 0.0140408 | 1.17092 |
| 28003 | AM-oracle-B64 | 1.18768 | 0.00854873 | 0.288746 | 0.00273636 | -1.49278 | 0.0141061 | 1.17745 |
| 28003 | AM-learned-B128 | 1.2048 | 0.0086114 | 0.299737 | 0.0024057 | -1.50747 | 0.0141957 | 1.2362 |
| 28003 | AM-oracle-B128 | 1.2168 | 0.00859108 | 0.323902 | 0.00282144 | -1.5013 | 0.0144196 | 1.23924 |

Frozen Flow reference on the identical SDE evaluation noise:

| seed | reward | energy | cost |
| --- | --- | --- | --- |
| 28001 | 0.748596 | 0 | -1.12289 |
| 28002 | 0.796467 | 0 | -1.1947 |
| 28003 | 0.804802 | 0 | -1.2072 |

## Budget (checkpoint elapsed includes fixed curve evaluations; extra diagnostics separately counted)

| seed | method | train_K | update | training_paths | generated_time_steps | actor_updates | actor_true_gradient_points | checkpoint_elapsed_seconds | extra_diagnostic_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 28001 | AM-learned-B64 | 32 | 0 | 0 | 0 | 0 | 0 | 0.255804 | 0.229351 |
| 28001 | AM-learned-B64 | 32 | 150 | 9600 | 307200 | 150 | 0 | 1.56787 | 0.461947 |
| 28001 | AM-learned-B64 | 32 | 300 | 19200 | 614400 | 300 | 0 | 2.8783 | 0.693176 |
| 28001 | AM-oracle-B64 | 32 | 0 | 0 | 0 | 0 | 0 | 0.0208302 | 0 |
| 28001 | AM-oracle-B64 | 32 | 150 | 9600 | 307200 | 150 | 9600 | 1.07263 | 0 |
| 28001 | AM-oracle-B64 | 32 | 300 | 19200 | 614400 | 300 | 19200 | 2.1143 | 0 |
| 28001 | AM-learned-B128 | 32 | 0 | 0 | 0 | 0 | 0 | 0.0218292 | 0 |
| 28001 | AM-learned-B128 | 32 | 150 | 19200 | 614400 | 150 | 0 | 1.52079 | 0 |
| 28001 | AM-learned-B128 | 32 | 300 | 38400 | 1.2288e+06 | 300 | 0 | 3.02091 | 0 |
| 28001 | AM-oracle-B128 | 32 | 0 | 0 | 0 | 0 | 0 | 0.022591 | 0 |
| 28001 | AM-oracle-B128 | 32 | 150 | 19200 | 614400 | 150 | 19200 | 1.4029 | 0 |
| 28001 | AM-oracle-B128 | 32 | 300 | 38400 | 1.2288e+06 | 300 | 38400 | 2.78443 | 0 |
| 28001 | AM-learned-B64-K64 | 64 | 300 | 19200 | 1.2288e+06 | 300 | 0 | 3.724 | 0 |
| 28001 | AM-oracle-B64-K64 | 64 | 300 | 19200 | 1.2288e+06 | 300 | 19200 | 3.62731 | 0 |
| 28002 | AM-learned-B64 | 32 | 0 | 0 | 0 | 0 | 0 | 0.249057 | 0.228079 |
| 28002 | AM-learned-B64 | 32 | 150 | 9600 | 307200 | 150 | 0 | 1.54059 | 0.464003 |
| 28002 | AM-learned-B64 | 32 | 300 | 19200 | 614400 | 300 | 0 | 2.82583 | 0.695584 |
| 28002 | AM-oracle-B64 | 32 | 0 | 0 | 0 | 0 | 0 | 0.0207138 | 0 |
| 28002 | AM-oracle-B64 | 32 | 150 | 9600 | 307200 | 150 | 9600 | 1.03485 | 0 |
| 28002 | AM-oracle-B64 | 32 | 300 | 19200 | 614400 | 300 | 19200 | 2.05261 | 0 |
| 28002 | AM-learned-B128 | 32 | 0 | 0 | 0 | 0 | 0 | 0.0216457 | 0 |
| 28002 | AM-learned-B128 | 32 | 150 | 19200 | 614400 | 150 | 0 | 1.49891 | 0 |
| 28002 | AM-learned-B128 | 32 | 300 | 38400 | 1.2288e+06 | 300 | 0 | 2.98639 | 0 |
| 28002 | AM-oracle-B128 | 32 | 0 | 0 | 0 | 0 | 0 | 0.0212564 | 0 |
| 28002 | AM-oracle-B128 | 32 | 150 | 19200 | 614400 | 150 | 19200 | 1.40032 | 0 |
| 28002 | AM-oracle-B128 | 32 | 300 | 38400 | 1.2288e+06 | 300 | 38400 | 3.44792 | 0 |
| 28002 | AM-learned-B64-K64 | 64 | 300 | 19200 | 1.2288e+06 | 300 | 0 | 3.66168 | 0 |
| 28002 | AM-oracle-B64-K64 | 64 | 300 | 19200 | 1.2288e+06 | 300 | 19200 | 3.65449 | 0 |
| 28003 | AM-learned-B64 | 32 | 0 | 0 | 0 | 0 | 0 | 0.261978 | 0.238113 |
| 28003 | AM-learned-B64 | 32 | 150 | 9600 | 307200 | 150 | 0 | 1.58025 | 0.469302 |
| 28003 | AM-learned-B64 | 32 | 300 | 19200 | 614400 | 300 | 0 | 2.85341 | 0.698777 |
| 28003 | AM-oracle-B64 | 32 | 0 | 0 | 0 | 0 | 0 | 0.020335 | 0 |
| 28003 | AM-oracle-B64 | 32 | 150 | 9600 | 307200 | 150 | 9600 | 1.02655 | 0 |
| 28003 | AM-oracle-B64 | 32 | 300 | 19200 | 614400 | 300 | 19200 | 2.04944 | 0 |
| 28003 | AM-learned-B128 | 32 | 0 | 0 | 0 | 0 | 0 | 0.0235803 | 0 |
| 28003 | AM-learned-B128 | 32 | 150 | 19200 | 614400 | 150 | 0 | 1.51896 | 0 |
| 28003 | AM-learned-B128 | 32 | 300 | 38400 | 1.2288e+06 | 300 | 0 | 3.00019 | 0 |
| 28003 | AM-oracle-B128 | 32 | 0 | 0 | 0 | 0 | 0 | 0.0244122 | 0 |
| 28003 | AM-oracle-B128 | 32 | 150 | 19200 | 614400 | 150 | 19200 | 1.68393 | 0 |
| 28003 | AM-oracle-B128 | 32 | 300 | 38400 | 1.2288e+06 | 300 | 38400 | 3.34534 | 0 |
| 28003 | AM-learned-B64-K64 | 64 | 300 | 19200 | 1.2288e+06 | 300 | 0 | 4.10767 | 0 |
| 28003 | AM-oracle-B64-K64 | 64 | 300 | 19200 | 1.2288e+06 | 300 | 19200 | 3.58271 | 0 |

B64/300 and B128/150 both use 19200 training paths, but they have different optimizer-step counts and not equal wall-clock or total compute. B128/300 uses twice the trajectories of B64/300; batch effects are not a pure variance estimate.

## Equal-rollout-budget results (B64/300 versus B128/150)

| seed | method | update | reward | energy | cost |
| --- | --- | --- | --- | --- | --- |
| 28001 | AM-learned-B64 | 300 | 1.15401 | 0.329502 | -1.40152 |
| 28001 | AM-oracle-B64 | 300 | 1.16614 | 0.350895 | -1.39831 |
| 28001 | AM-learned-B128 | 150 | 1.06647 | 0.22402 | -1.37569 |
| 28001 | AM-oracle-B128 | 150 | 1.09154 | 0.262625 | -1.37469 |
| 28002 | AM-learned-B64 | 300 | 1.1896 | 0.283905 | -1.50049 |
| 28002 | AM-oracle-B64 | 300 | 1.2054 | 0.302513 | -1.50559 |
| 28002 | AM-learned-B128 | 150 | 1.14481 | 0.243768 | -1.47344 |
| 28002 | AM-oracle-B128 | 150 | 1.15916 | 0.266739 | -1.472 |
| 28003 | AM-learned-B64 | 300 | 1.17899 | 0.272156 | -1.49633 |
| 28003 | AM-oracle-B64 | 300 | 1.18768 | 0.288746 | -1.49278 |
| 28003 | AM-learned-B128 | 150 | 1.13462 | 0.240972 | -1.46095 |
| 28003 | AM-oracle-B128 | 150 | 1.11386 | 0.201967 | -1.46883 |

## Paired 2×2 effects

For J: A_B = J_learned,B − J_oracle,B, S_source = J_source,64 − J_source,128, I=A64−A128=S_learned−S_oracle. For reward, higher-better reverses the first two signs and I=S_learned−S_oracle. The energy table uses lower-better convention.

| schedule | metric | effect | mean | seed_sd | descriptive_ci95_low | descriptive_ci95_high | positive_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| equal_updates | cost | A64 | -0.000552024 | 0.00489297 | -0.0127068 | 0.0116028 | 1 |
| equal_updates | cost | A128 | -0.00413983 | 0.00181566 | -0.00865019 | 0.000370524 | 0 |
| equal_updates | cost | S_learned | 0.0082033 | 0.00255718 | 0.00185092 | 0.0145557 | 3 |
| equal_updates | cost | S_oracle | 0.00461549 | 0.00593142 | -0.010119 | 0.01935 | 2 |
| equal_updates | cost | I | 0.00358781 | 0.00468041 | -0.00803897 | 0.0152146 | 2 |
| equal_rollouts | cost | A64 | -0.000552024 | 0.00489297 | -0.0127068 | 0.0116028 | 1 |
| equal_rollouts | cost | A128 | 0.00181253 | 0.00525479 | -0.0112411 | 0.0148662 | 1 |
| equal_rollouts | cost | S_learned | -0.0294151 | 0.00519485 | -0.0423198 | -0.0165104 | 0 |
| equal_rollouts | cost | S_oracle | -0.0270506 | 0.00565718 | -0.0411038 | -0.0129974 | 0 |
| equal_rollouts | cost | I | -0.00236456 | 0.00897791 | -0.0246669 | 0.0199378 | 1 |
| equal_updates | reward | A64 | 0.0122078 | 0.00355373 | 0.00337988 | 0.0210358 | 3 |
| equal_updates | reward | A128 | 0.0192343 | 0.0173478 | -0.0238599 | 0.0623285 | 3 |
| equal_updates | reward | S_learned | 0.0248657 | 0.000963437 | 0.0224724 | 0.027259 | 3 |
| equal_updates | reward | S_oracle | 0.0318921 | 0.0150364 | -0.00546046 | 0.0692447 | 3 |
| equal_updates | reward | I | -0.00702643 | 0.014697 | -0.0435357 | 0.0294829 | 1 |
| equal_rollouts | reward | A64 | 0.0122078 | 0.00355373 | 0.00337988 | 0.0210358 | 3 |
| equal_rollouts | reward | A128 | 0.00622442 | 0.0239694 | -0.053319 | 0.0657678 | 2 |
| equal_rollouts | reward | S_learned | -0.0588996 | 0.0248023 | -0.120512 | 0.00271281 | 0 |
| equal_rollouts | reward | S_oracle | -0.0648831 | 0.0161538 | -0.105011 | -0.0247549 | 0 |
| equal_rollouts | reward | I | 0.00598343 | 0.0215572 | -0.0475677 | 0.0595345 | 2 |
| equal_updates | energy | A64 | -0.0188638 | 0.00241181 | -0.0248551 | -0.0128725 | 0 |
| equal_updates | energy | A128 | -0.0329913 | 0.025875 | -0.0972684 | 0.0312859 | 0 |
| equal_updates | energy | S_learned | -0.0290953 | 0.00166515 | -0.0332317 | -0.0249588 | 0 |
| equal_updates | energy | S_oracle | -0.0432227 | 0.0280238 | -0.112838 | 0.0263922 | 0 |
| equal_updates | energy | I | 0.0141275 | 0.0267225 | -0.052255 | 0.0805099 | 2 |
| equal_rollouts | energy | A64 | -0.0188638 | 0.00241181 | -0.0248551 | -0.0128725 | 0 |
| equal_rollouts | energy | A128 | -0.0075241 | 0.0410462 | -0.109489 | 0.0944403 | 1 |
| equal_rollouts | energy | S_learned | 0.0589343 | 0.0405592 | -0.0418203 | 0.159689 | 3 |
| equal_rollouts | energy | S_oracle | 0.070274 | 0.0298878 | -0.00397136 | 0.144519 | 3 |
| equal_rollouts | energy | I | -0.0113397 | 0.0388607 | -0.107875 | 0.0851955 | 2 |

Per-seed effects are in paired-effects-by-seed.csv; all paired final evaluation arrays and checkpoint 0/150/300 results are retained.

## Repeated common-checkpoint gradient, Adam and paired virtual-step evidence

| seed | checkpoint_actor | checkpoint_step | batch_size | mean_d_norm | d_norm_repeat_sd | trace_cov_qL | trace_cov_qO | trace_cov_d | trace_cross_cov_qO_d | variance_identity_residual | raw_cosine_mean | q_norm_ratio_mean | projection_c_mean | perpendicular_ratio_mean | clipped_cosine_mean | step_cosine_mean | step_norm_ratio_mean | step_relative_difference_mean | D_J_mean | D_J_repeat_sd | D_J_eval_mc_se_rms | D_R_mean | D_R_repeat_sd | D_R_eval_mc_se_rms | projection_undefined_count | projection_negative_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 28001 | AM-learned | 0 | 64 | 0.677448 | 0.294084 | 10.3962 | 9.87093 | 0.615748 | -0.0452412 | 1.30839e-10 | 0.969491 | 0.955811 | 0.926843 | 0.214678 | 0.969491 | 0.874549 | 0.999999 | 0.494518 | -0.00028015 | 0.0013731 | 0.000381266 | 0.000504467 | 0.00094627 | 0.000254759 | 0 | 0 |
| 28001 | AM-learned | 0 | 128 | 0.62892 | 0.235051 | 4.49138 | 4.80537 | 0.320939 | -0.317467 | -2.88356e-09 | 0.974331 | 0.94036 | 0.916507 | 0.200092 | 0.974331 | 0.884406 | 1 | 0.477476 | 4.21454e-05 | 0.000749014 | 0.000377915 | 0.000293231 | 0.000418003 | 0.000253768 | 0 | 0 |
| 28001 | AM-learned | 150 | 64 | 0.648941 | 0.350354 | 8.65959 | 9.08224 | 0.68227 | -0.552463 | 1.98743e-09 | 0.954296 | 0.895638 | 0.854716 | 0.232154 | 0.954296 | 0.991839 | 0.991162 | 0.124403 | -0.00012007 | 0.000145544 | 7.70743e-05 | 8.73636e-05 | 0.000196202 | 5.44544e-05 | 0 | 0 |
| 28001 | AM-learned | 150 | 128 | 0.656791 | 0.275569 | 4.27919 | 4.51911 | 0.329107 | -0.284515 | -2.18931e-09 | 0.963425 | 0.868639 | 0.839018 | 0.201342 | 0.963425 | 0.994355 | 0.992548 | 0.105079 | -0.000129231 | 0.000105788 | 6.32981e-05 | 8.45148e-05 | 0.000131596 | 4.50042e-05 | 0 | 0 |
| 28001 | AM-learned | 300 | 64 | 0.894469 | 0.323181 | 6.13489 | 6.67106 | 0.477075 | -0.506625 | 1.59447e-09 | 0.915463 | 0.95055 | 0.866304 | 0.360652 | 0.915463 | 0.994265 | 0.999798 | 0.106933 | -2.6451e-05 | 0.00010361 | 8.12677e-05 | 5.33871e-05 | 0.000163443 | 5.79025e-05 | 0 | 0 |
| 28001 | AM-learned | 300 | 128 | 0.816721 | 0.234284 | 3.9375 | 4.37508 | 0.304183 | -0.370879 | -1.398e-09 | 0.920583 | 0.880396 | 0.80903 | 0.314205 | 0.920583 | 0.995295 | 0.996849 | 0.0968888 | -2.38418e-05 | 8.19992e-05 | 7.56788e-05 | 4.92084e-05 | 0.000122191 | 5.3862e-05 | 0 | 0 |
| 28001 | AM-oracle | 300 | 64 | 0.817917 | 0.289495 | 8.1522 | 9.63462 | 0.511848 | -0.997133 | -6.13824e-09 | 0.908765 | 1.00206 | 0.91048 | 0.35649 | 0.908765 | 0.994737 | 1.01524 | 0.102691 | 3.6931e-05 | 0.000120502 | 8.12374e-05 | 6.15948e-05 | 0.000133728 | 5.84304e-05 | 0 | 0 |
| 28001 | AM-oracle | 300 | 128 | 0.806411 | 0.23302 | 3.99572 | 4.65131 | 0.336962 | -0.496275 | 8.7261e-10 | 0.915931 | 1.03131 | 0.945594 | 0.360133 | 0.915931 | 0.995817 | 1.019 | 0.0937808 | 1.23482e-05 | 0.000101738 | 7.50194e-05 | 5.76563e-05 | 0.000104728 | 5.43353e-05 | 0 | 0 |
| 28002 | AM-learned | 0 | 64 | 0.466387 | 0.267349 | 6.59382 | 8.45086 | 0.649029 | -1.25303 | 1.71431e-09 | 0.969011 | 0.914202 | 0.886256 | 0.202691 | 0.969011 | 0.846647 | 0.999983 | 0.538638 | 0.00095072 | 0.00193555 | 0.000479756 | -0.000640074 | 0.00132158 | 0.000320626 | 0 | 0 |
| 28002 | AM-learned | 0 | 128 | 0.433312 | 0.161562 | 2.92821 | 3.74186 | 0.304171 | -0.558914 | -9.45116e-10 | 0.978331 | 0.942936 | 0.922149 | 0.187846 | 0.978331 | 0.862576 | 0.99999 | 0.518444 | 0.000371411 | 0.000845878 | 0.000312022 | -0.000213068 | 0.000514164 | 0.000209021 | 0 | 0 |
| 28002 | AM-learned | 150 | 64 | 0.619508 | 0.402386 | 6.93398 | 9.09312 | 1.01047 | -1.5848 | 2.4658e-09 | 0.942933 | 0.975318 | 0.91779 | 0.303531 | 0.942933 | 0.992641 | 0.99498 | 0.118322 | 5.72548e-05 | 0.000141285 | 0.00010267 | 0.000197268 | 0.000199888 | 6.45392e-05 | 0 | 0 |
| 28002 | AM-learned | 150 | 128 | 0.585214 | 0.229192 | 3.92625 | 5.04074 | 0.472317 | -0.793399 | 3.10979e-09 | 0.945587 | 0.985658 | 0.932098 | 0.302809 | 0.945587 | 0.994276 | 0.989985 | 0.105494 | 5.24667e-05 | 0.00011322 | 8.73373e-05 | 0.000222106 | 0.00014024 | 5.47647e-05 | 0 | 0 |
| 28002 | AM-learned | 300 | 64 | 0.563556 | 0.420721 | 6.63708 | 8.86605 | 1.26582 | -1.7474 | 2.98736e-10 | 0.903294 | 0.894505 | 0.805814 | 0.312267 | 0.903294 | 0.989128 | 0.977175 | 0.14574 | -2.46907e-05 | 0.000318885 | 0.000125296 | 0.000250545 | 0.000230124 | 8.21746e-05 | 0 | 0 |
| 28002 | AM-learned | 300 | 128 | 0.590509 | 0.297985 | 3.05474 | 4.74205 | 0.752503 | -1.21991 | 1.32418e-09 | 0.936956 | 0.895251 | 0.835797 | 0.289464 | 0.936956 | 0.991515 | 0.98635 | 0.131627 | -9.83767e-05 | 0.000269258 | 0.000106072 | 0.000253748 | 0.000193293 | 7.06218e-05 | 0 | 0 |
| 28002 | AM-oracle | 300 | 64 | 0.625656 | 0.536393 | 4.46956 | 6.95939 | 1.27311 | -1.88147 | 2.02351e-09 | 0.911387 | 0.903201 | 0.826345 | 0.320312 | 0.911387 | 0.99148 | 0.991782 | 0.129719 | -6.04963e-05 | 0.000266391 | 0.000104191 | 0.000248226 | 0.000228276 | 7.13814e-05 | 0 | 0 |
| 28002 | AM-oracle | 300 | 128 | 0.599225 | 0.310724 | 1.94932 | 3.08916 | 0.580997 | -0.860419 | -3.05082e-09 | 0.923428 | 0.892365 | 0.823107 | 0.314258 | 0.923428 | 0.993871 | 0.993598 | 0.110534 | -5.72351e-05 | 0.00017868 | 8.90113e-05 | 0.000248492 | 0.000151473 | 5.98651e-05 | 0 | 0 |
| 28003 | AM-learned | 0 | 64 | 0.632295 | 0.363616 | 6.50294 | 9.3804 | 0.813758 | -1.84561 | -2.14277e-09 | 0.961158 | 1.00172 | 0.961607 | 0.248828 | 0.961158 | 0.83284 | 1.00001 | 0.56735 | 0.0005475 | 0.00214882 | 0.000678868 | -0.000754079 | 0.00130941 | 0.00045163 | 0 | 0 |
| 28003 | AM-learned | 0 | 128 | 0.694201 | 0.274635 | 2.43238 | 3.48143 | 0.35067 | -0.699858 | 1.54032e-09 | 0.96811 | 1.04053 | 1.00684 | 0.241401 | 0.96811 | 0.827215 | 1 | 0.579619 | -0.000421601 | 0.00226067 | 0.000611225 | -0.000297352 | 0.00119842 | 0.00040359 | 0 | 0 |
| 28003 | AM-learned | 150 | 64 | 1.3155 | 0.466736 | 6.09273 | 10.4998 | 1.20375 | -2.80539 | 1.07481e-08 | 0.938373 | 0.750401 | 0.704245 | 0.239793 | 0.938373 | 0.988843 | 0.997151 | 0.148994 | -0.000891875 | 0.000479186 | 0.000150626 | 0.000224451 | 0.000176702 | 9.98155e-05 | 0 | 0 |
| 28003 | AM-learned | 150 | 128 | 1.34187 | 0.310467 | 3.04264 | 5.2672 | 0.582653 | -1.40361 | 2.67747e-09 | 0.925435 | 0.729155 | 0.673268 | 0.26113 | 0.925435 | 0.989688 | 1.00578 | 0.144848 | -0.00096657 | 0.000352535 | 0.000139889 | 0.000191195 | 0.000133231 | 9.42658e-05 | 0 | 0 |
| 28003 | AM-learned | 300 | 64 | 1.47672 | 0.546182 | 8.63018 | 15.5329 | 1.53541 | -4.21908 | 7.12994e-11 | 0.953684 | 0.753786 | 0.718814 | 0.217057 | 0.953684 | 0.984653 | 0.954541 | 0.175726 | -0.000608167 | 0.000310759 | 0.000131806 | 6.40977e-05 | 0.000199647 | 9.90456e-05 | 0 | 0 |
| 28003 | AM-learned | 300 | 128 | 1.50185 | 0.48159 | 4.88118 | 8.83246 | 0.883705 | -2.41749 | 2.03303e-09 | 0.954954 | 0.745317 | 0.711471 | 0.218466 | 0.954954 | 0.984553 | 0.942019 | 0.18014 | -0.000743213 | 0.000240427 | 0.000129285 | 8.74212e-05 | 0.000176875 | 9.8433e-05 | 0 | 0 |
| 28003 | AM-oracle | 300 | 64 | 1.48613 | 0.448239 | 8.14572 | 14.1542 | 1.37669 | -3.69257 | -3.17121e-09 | 0.952336 | 0.807464 | 0.768679 | 0.234487 | 0.952336 | 0.987185 | 0.9984 | 0.161004 | -0.000431046 | 0.000300129 | 0.000102954 | 9.98844e-05 | 0.000143197 | 7.90187e-05 | 0 | 0 |
| 28003 | AM-oracle | 300 | 128 | 1.48878 | 0.36646 | 3.69096 | 6.58885 | 0.713408 | -1.80565 | -6.47286e-10 | 0.954662 | 0.807898 | 0.771047 | 0.234154 | 0.954662 | 0.986637 | 0.984954 | 0.163266 | -0.000601182 | 0.000283911 | 9.94387e-05 | 0.000144089 | 0.000121203 | 7.7083e-05 | 0 | 0 |

The mean of d=qL−qO is a finite-repetition estimate, not known population bias. The exact covariance identity includes 2 tr Cov(qO,d); no independence assumption is made. Probe-repetitions.csv gives every raw/clipped/step norm, cosine, projection coefficient/perpendicular ratio, and paired D_J/D_R. Their Monte Carlo SE is computed from per-path differences, whereas repeat SD describes variation across fresh training batches at a fixed actor. Near-zero qO projections are marked undefined and all repeats remain counted.

At these shared-history virtual updates, Adam-step directions are on average more aligned than raw gradients; this does not imply independent long-run learned/oracle histories converge.

## Grid sensitivity

| seed | train_K | eval_K | J_oracle_gain |
| --- | --- | --- | --- |
| 28001 | 32 | 32 | -0.00320338 |
| 28002 | 32 | 32 | 0.0050944 |
| 28003 | 32 | 32 | -0.00354709 |
| 28001 | 32 | 64 | -0.00550728 |
| 28002 | 32 | 64 | 0.00516383 |
| 28003 | 32 | 64 | -0.00462025 |
| 28001 | 64 | 64 | 0.010912 |
| 28002 | 64 | 64 | -0.00101428 |
| 28003 | 64 | 64 | -0.00652939 |

Oracle–learned J signs differ across grid settings for at least one seed; no grid-stable conclusion.

K32 actors are evaluated under both K32 and K64 finite chains; K64 actors were retrained at K64 for all three seeds with B64. Training grid and evaluation grid are distinct variables. K64 does not provide a complete 2×2.

## Provenance and limits

The three original arms and first-seed K64 arms were retrained into a new unique run; original historical outputs were not overwritten or pooled as extra seeds. The historical baseline replay audit compares data, frozen checkpoint tensors, actor and Adam states at 0/150/300, and saved paired evaluation arrays. Oracle-B128 and K64 seeds 2–3 are new training. All repeated-gradient and virtual-step measurements are new diagnostics. Initial actor and frozen-model hashes are audited for every seed.

This is a fixed objective with a learned terminal-label gradient, not an uncertain true terminal cost. No new uncertainty algorithm was added. A small or interval-overlapping effect does not demonstrate equivalence. The 32 probes on shared states isolate local update mechanisms, but cannot identify the causal contribution of independent long-run optimizer histories. Three previously inspected training seeds are an exploratory diagnosis.

Figures: four-arm-learning-curves.png, paired-effects.png, gradient-adam-virtual-step.png, grid-sensitivity.png. Full numerical outputs are in this run directory.
