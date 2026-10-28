# Current-policy Q-gradient reference precision audit — FAIL

The 17 geometry-based audit points were fixed in `configs/reference_audit_points.json`
at commit `af127527efb7ce787f7f6c2e7c66716fb6efc8d1`, before this audit.
This run freezes the **ordinary QAM Actor/Critic at seed 59000, N=0**.
The numerical table evaluates its full current ODE deployment sampler; independent
Monte Carlo uses real simulator transitions and two independent 8192-rollout
streams per point, without reading Critic or table Q. The finite differences use
shared deterministic QMC nodes at h=(0.04, 0.02, 0.01).

## Predeclared numerical study

Nested levels are L0:(41, 8, 128), G1:(61, 8, 128), P1:(61, 16, 128), T1:(61, 16, 256), G2:(81, 16, 256), P2:(81, 32, 256), T2:(81, 32, 512). `G`, `P`, and
`T` steps separately change grid, policy samples, and transition samples.
Pointwise action-gradient quadrature uses (2048, 4096, 8192) Sobol nodes. The
highest two *table* levels P2→T2 differ only in transition integration.

On the 13 nondegenerate points, E_ref = RMS(g_P2−g_T2) = **0.000668707**;
E_critic = RMS(g_critic−g_T2) =
**0.533984**; ratio = **0.001252**
(required ≤ 0.25). This ratio alone cannot prove reference accuracy: the
grid/policy and independent MC/FD checks are additional mandatory gates.
Last-axis drifts: {'grid': 0.01334381516692321, 'policy': 0.012576553224394448, 'transition': 0.0006690765186812851, 'gradient_quadrature': 0.0003671852419470719}. Max-axis/critic ratio =
0.02499. Near-zero gradients use absolute error;
the affected point IDs are ['A06', 'A09', 'A13', 'A14'].

Independent MC Q z RMS/max (positive-SE points only) = 2.75/
7.01; gradient z RMS/max =
1.1/2.42.
Zero-SE MC entries use fixed absolute fallbacks (Q ≤ 0.001, gradient
component ≤ 0.01), not meaningless relative/z errors: Q points
['A14'], gradient components
['A14:x', 'A14:y'].
MC gradient SE RMS / critic gap = 0.0306;
the two MC splits' gradient differences are stored per point. The h=0.01
finite-difference gradient RMS/max-coordinate error is
0.00828/0.0365.
Opposite table-vs-MC directions at nondegenerate points: 0.

Automatic gate results: `{"all_axis_drift": true, "convergence_trend": true, "direction": true, "finite_difference": true, "mc_gradient_consistency": true, "mc_gradient_precision": true, "mc_q_consistency": false, "reference_to_critic_ratio": true}`.

## Decision

Reference precision failed. Formal Oracle-gap experiment not executed.

No formal 3-seed training, 50-pair trajectory test, or causal Q-gradient
performance-gap conclusion is reported here. Those conditional downstream
steps remain blocked unless the reference is certified for updated policies.
The original single-seed smoke and all earlier research artifacts are untouched.

## Post-failure, non-confirmatory diagnosis

The two 8192-rollout MC splits agree with each other: their worst Q-mean
difference is 1.64 combined standard errors. The primary table fails the
predeclared Q–MC rule at A01, A02, and A16 (respectively about 4.35, 4.05,
and 7.01 MC standard errors). A14 had exactly zero MC variance: after the
first run exposed the undefined z-score, the runner was corrected to use an
explicit absolute fallback rather than dividing by a tiny epsilon. This
correction is recorded in `runner-revision-history.json`; it does not turn the
failed gate into a pass.

We **did not change the gate** after observing this failure. Separate
single-axis sensitivity probes, using the same fixed Actor and points, show:

| Post-failure table | Mean absolute Q–MC error | Q z RMS | Maximum Q z | Table build time |
|---|---:|---:|---:|---:|
| 81² grid, 32 policy, 256 transition (primary P2) | 0.00987 | 2.71 | 6.92 | 192 s |
| 81² grid, 64 policy, 256 transition | 0.01062 | 3.08 | 6.15 | 477 s |
| 121² grid, 32 policy, 256 transition | 0.00610 | 1.63 | 3.54 | 409 s |

The 121² table's pointwise gradient also agrees with independent MC at
gradient-z RMS/max 0.96/2.73, and with h=0.01 finite differences at RMS
absolute error 0.00841. Its 81²→121² nondegenerate gradient drift / Critic
gap is 0.0191. These are useful *development* observations, not a retroactive
PASS: the 121² setting was selected after the primary failed, and only the
smoke seed's `N=0` Actor has been examined. The result suggests spatial-grid
interpolation near the obstacle/goal and other sharp value geometry is more
important here than merely doubling policy samples; it does not prove a unique
error cause. The [64-policy probe](followup-g81-p64-t256.json) and
[121-grid probe](followup-g121-p32-t256.json) retain all 17 points.

At roughly 409 seconds per 121² current-policy table, recomputing one before
each of 300 reference Actor updates for each of three seeds would require
about **102 hours of table construction alone on this machine**. A credible
formal protocol therefore needs a separately validated faster reference
estimator or numerical acceleration, plus precision checks on updated Actors;
using the failed 81² table or stale tables would violate the fixed protocol.
