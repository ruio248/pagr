# Two-dimensional obstacle QAM — design smoke, **not** formal evidence

This unique run is a **single-seed N=10 smoke** (`seed=59000`). It proves that the
new environment, offline transitions, conditional Flow, ordinary TD/Lean-AM
training, a current-policy numerical-table branch, paired evaluation, and a
behavior animation run end to end. It does **not** establish a high-precision
reference gradient or a repeatable performance advantage. Formal training is
blocked in code until a separate reference-precision audit passes.

The map is `[0,1]^2`, with a left-start rectangle, goal `(0.85,0.50)` radius
`0.06`, and obstacle `(0.50,0.50)` radius `0.16`. At each of 28 environment
steps, the raw Flow action `a∈R²` produces `p'=p+0.06 tanh(a)+0.012 ξ`.
The segment from `p` to `p'` is checked for collision before a goal hit;
collision and out-of-bounds terminate with `−1`, goal terminates with `+1`,
and horizon timeout gives `0`. There is no distance shaping or hidden clamp.

The behavior data mix upper/lower routes (inferred from the initial side of
`y=0.5`) and 5% exploration. Training and validation are independent complete
episodes. The 2400-episode training set had 1727 successes, 640 collisions,
and 33 timeouts; both upper and lower routes had successful trajectories.
The pretrained conditional FM Base is shared and frozen. Both branches start
from the same Actor/Critic weights; each step has one current-Actor TD backup
and one Lean-AM Actor optimizer step. The only terminal-label difference is
`∇ₐ Q_φ` versus the current-Actor numerical-table estimate. Both environmental
deployment and TD use the same 8-step ODE action sampler; the AM matching
path is the 8-step finite-chain SDE. This is a frozen-Base QAM core study, not
the complete official QAM stack. The four source functions were checked at
[official commit 2726d767](https://github.com/ColinQiyangLi/qam/blob/2726d767c9a0a7a46d49693f0391f73dc2cf58ac/agents/qam.py).

## Behavioral smoke outcome

| Evaluation | Method | N=0 success | N=5 success | N=10 success | N=10 collision | N=10 discounted return |
|---|---|---:|---:|---:|---:|---:|
| Curve, 256 paired episodes | ordinary TD-Critic | 0.5156 | 0.6016 | 0.5820 | 0.4180 | 0.0840 |
| Curve, 256 paired episodes | numerical-table terminal | 0.5156 | 0.6836 | 0.6992 | 0.3008 | 0.2793 |
| Independent final, 512 paired episodes | ordinary TD-Critic | — | — | 0.4980 | 0.5020 | −0.0560 |
| Independent final, 512 paired episodes | numerical-table terminal | — | — | 0.6152 | 0.3848 | 0.1385 |

The 512 final episodes are a different evaluation stream and are **not**
appended to the 256-episode learning curve. Six demonstration starts were
predeclared before rendering; all six, including failures, appear in the
[physical trajectory plot](showcase-trajectories.png) and [GIF](showcase.gif).
`ffmpeg` was unavailable, so no MP4 was generated. The [curve](learning-curves.png)
and [per-run CSV](curves.csv) report all checks. No third algorithm is defined;
the transformation interface is the identity and is not represented as PAGR.

## Reference precision gate: **failed / not established**

The numerical table performs finite-horizon policy evaluation, not `max_a`:
it averages the complete current ODE sampler's actions at each grid point,
then applies Gaussian transition integration with the simulator's exact
collision/goal/horizon masks. A separate 2048-point Gaussian Sobol integral
computes each terminal action gradient through the score identity including
`0.06 sech²(a)`. Table parameters in this smoke were grid `41×41`, eight
policy samples, and 128 Gaussian transition samples per policy action.

An independent environment-MC audit at five preselected geometries (start,
near obstacle, past obstacle, near goal, near boundary) used 4096 closed-loop
rollouts per point. For the shared initial Actor, the `k=0` start point's MC
value was about `0.3033 ± 0.0116` (one MC standard error), while the smoke
table gave `0.3419`. A `61×61` / 16-policy-sample table gave `0.2612`, an
`81×81` / 32-policy-sample table `0.2721`, an `81×81` / 64-policy-sample table
`0.2574`, and a `121×121` / 32-policy-sample / 32-transition-sample table
`0.2890`. The sequence is **not converged** in grid and policy integration;
at the near-obstacle point the corresponding values also moved substantially.
The `121×121` table took about 74 seconds, and the `81×81` / 64-policy table
about 203 seconds for a **single Actor snapshot** on this machine. The
required recalculation before every reference Actor update would be expensive
even for N=10, let alone the suggested N=300 and three seeds. Independent MC
score-gradient and center-difference checks are exploratory and do not cure
the global value-table bias.

Therefore the apparent N=10 smoke gap **cannot** be attributed to accurate
terminal-gradient substitution. The [six-point mechanism probe](mechanism-smoke.json)
is explicitly marked `reference_validated=false`; its `Q` MAE, propagated
label difference, and virtual Adam-step difference show code-path behavior,
not validated physical error. No full formal three-seed comparison, calibrated
reference, or novel suppression method has been run. The 16D and earlier
analytic results remain untouched.

The run's `provenance.json` stores the Python-source hash at training start.
Subsequent edits added visualization, an explicit formal-run gate, and corrected
the smoke mechanism label statistic; thus the current source hash differs from
that training-start hash. The saved data, checkpoints, metrics and plots were
not overwritten. This is another reason not to treat this design run as a
frozen formal protocol.
