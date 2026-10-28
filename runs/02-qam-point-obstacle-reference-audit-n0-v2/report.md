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

## Post-failure numerical diagnosis

The two independent 8192-rollout MC splits agree with each other: worst Q
mean split discrepancy is 1.64 combined standard errors. Primary table Q is
inconsistent with MC at A01, A02, A16 (about 4.35, 4.05, 7.01 standard errors).
The zero-variance MC point A14 uses an explicit absolute-error fallback added
after the first run exposed an undefined z-score; that runner correction is
logged and is **not** the cause of the remaining failure. The high→higher
transition-quadrature ratio of 0.00125 does
not cover grid error: the last grid-only gradient drift is 0.01334, about 20×
the high→higher transition drift 0.000669.

After failure, fixed-point *diagnostic* probes changed only one axis at a
time. Doubling policy samples 32→64 on the 81² grid did not improve Q–MC
agreement (Q z RMS 2.71→3.08); raising the grid 81²→121² at 32 policy samples
did (2.71→1.63, maximum 6.92→3.54). The latter table's gradient agrees with
independent MC (z RMS/max 0.96/2.73) and h=0.01 finite differences (RMS
absolute error 0.00841), but 121² was chosen *after seeing failure* and was
tested only on the smoke seed's `N=0` Actor. The likely dominant error is
spatial discretization/interpolation around sharp collision/goal geometry;
that is an inference, not a proven unique source. See the
[full precision report](reference-audit/precision_report.md) and both
[policy-sampling](reference-audit/followup-g81-p64-t256.csv) and
[grid](reference-audit/followup-g121-p32-t256.csv) follow-up CSVs.

## Decision questions

1. **Did the current-policy numerical reference pass?** No. The original
   Q–MC acceptance thresholds were retained; the primary precision gate is
   `reference_validated=false`, solely because
   numerical Q fails independent MC Q consistency. We did not relax a threshold
   or retroactively promote the post-failure 121² probe.
2. **How large is reference self-error relative to Critic discrepancy?** On
   13 nondegenerate points, the highest two table levels have
   `E_ref=0.000669`, `E_critic=0.533984`, ratio `0.001252`; the largest last-axis
   drift ratio is `0.02499`. These attractive *gradient* numbers do not cure
   Q-value bias or validate every updated policy.
3. **Is there a stable full 3-seed, fixed-N performance gap?** Unknown: formal
   training was not executed because the gate failed. The earlier one-seed
   `N=10` smoke is not a formal result.
4. **Does any gap come from collision, timeout, or shorter paths?** Unknown
   without formal trajectories; no causal component is assigned.
5. **How many of 50 paired failures were repaired?** Not measured. No formal
   50-pair test bank or trajectory claims were created.
6. **Does true Critic gradient error reach adjoint, matching, and Adam step?**
   The old smoke probe demonstrated a *difference between two labels* through
   those code paths, but its numerical label was unvalidated. A formal
   same-Ordinary-Actor mechanism diagnosis at `N=0/150/300` remains undone.
7. **Is a recoverable Q-gradient-related performance gap established?** No.
   Neither reference accuracy across training nor the formal performance
   comparison has been established.
8. **What remains unverified?** Precision for updated Actors and their endpoint
   distribution; a fast validated current-policy reference; formal 3-seed
   fixed-update comparison; final independent evaluation; 50 paired episodes;
   full checkpoint mechanism and ensemble/propagation diagnostics.
9. **Can PAGR robustness development begin on this evidence?** Not as a
   validated algorithm study. The existing identity transform is only an
   interface placeholder; no new PAGR method was designed or claimed.

The 121² table takes about 409 seconds per frozen Actor on this machine;
rebuilding it before every reference update for 300 updates × 3 seeds is
roughly **102 hours of table construction alone**. A faster, separately
validated local current-policy estimator or numerical acceleration, and
updated-Actor checks, are prerequisites. **Reference precision failed.
Formal Oracle-gap experiment not executed.**
