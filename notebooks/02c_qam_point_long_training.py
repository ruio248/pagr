# %% [markdown]
# # Notebook 2 extension: does the QAM gap persist with longer training?
#
# This extends the existing two-dimensional obstacle task to three seeds and
# 300 Actor optimizer updates per method. The environment, offline data recipe,
# frozen Base, deployment sampler, Lean Adjoint, Adam, and TD implementation are
# inherited unchanged from `02_qam_point_obstacle.py`.
#
# Ordinary QAM uses `-beta * grad_a Q_critic` as the terminal adjoint. The
# privileged reference branch uses a direct current-policy MC score gradient
# with 8192 simulator trajectories per endpoint. Both branches still perform
# one TD update per Actor update. A GPU accelerates only Actor forward calls;
# the independent environment transitions use the original NumPy simulator.
# No new PAGR method is introduced.
#
# The new user-approved protocol checks gradient precision, not scalar Q as a
# veto. Independent MC uses the original NumPy simulator and separate IID
# trajectories. Scalar Q disagreement is retained as a diagnostic. Snapshot
# checks include actual training endpoints; they do not certify every point.
#
# The old N=10 smoke and original audit FAIL are separate historical artifacts.
# A first GPU-table extension paused at seed 59001, N=50 because a near-goal
# gradient failed independent MC. Table and quadrature refinement did not cure
# the bias. The MC extension restarts Actor training at N=0 with unchanged
# thresholds. Seed 59001 was seen in numerical development, so conclusions are
# explicitly exploratory.

# %% [markdown]
# ## What changed, and what did not?
#
# The environment has position $p_k\in[0,1]^2$ and action $a_k\in\mathbb R^2$:
#
# $$p_{k+1}=p_k+\Delta\tanh(a_k)+\sigma\xi_k,\qquad \xi_k\sim\mathcal N(0,I).$$
#
# The goal gives +1, segment collision / leaving the square gives -1, and
# timeout gives 0. Subsequent actions come from the **frozen current Actor**,
# not a best-action maximization or a reference Critic. For a fixed first
# state/action, let $G$ be the discounted return of that continuation. Then
#
# $$\nabla_a Q^\pi(k,p,a)=\frac{\Delta(1-\tanh^2(a))}{\sigma}
#       \odot\mathbb E[\xi_k G].$$
#
# The implementation uses the unbiased sample covariance, which removes a
# constant return baseline without changing this expectation:
#
# $$\widehat{\nabla_a Q^\pi}=\frac{\Delta(1-\tanh^2(a))}{\sigma}
#  \odot\frac{1}{M-1}\sum_{m=1}^{M}(\xi^{(m)}-\bar\xi)(G^{(m)}-\bar G),
#  \quad M=8192.$$
#
# Ordinary QAM supplies $-\beta\nabla_a Q_{\rm critic}$ to the lean-adjoint
# terminal condition; Reference-QAM supplies $-\beta\widehat{\nabla_a Q^\pi}$.
# The rest of the matching objective and the Actor optimizer are unchanged.
# Both branches update a TD Critic, but the reference branch does **not** use
# that Critic to supply its Actor's terminal gradient. It is a privileged
# diagnostic, not a simulator-free replacement algorithm.
#
# This experiment can test a finite-update performance gap. It cannot by
# itself establish that Q values are already accurate, that the effect is
# caused only by adjoint amplification, or that a cheap new algorithm wins.
# Success/return also do not equal the full KL/energy-regularized AM objective;
# this extension does not claim to have measured improvement in that objective.

# %%
import json
import os
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd()
if not (ROOT/'configs/point_long_training_mc.json').exists():
    ROOT = ROOT.parent
protocol = json.loads((ROOT/'configs/point_long_training_mc.json').read_text())
display(pd.DataFrame([dict(seeds=protocol['seeds'], actor_updates=protocol['actor_updates'],
                          curve_updates=protocol['curve_updates'],
                          audit_updates=protocol['audit_updates'],
                          final_episodes=protocol['final_episodes'])]))

# %% [markdown]
# Training implementation: `scripts/run_point_long_training.py`.
# Local MC reference: `MonteCarloReference` in the training script.
# Historical accelerated table: `scripts/point_reference_accelerated.py`.
# Analysis: `scripts/analyze_point_long_training.py`.
#
# Launch explicitly from the repository root (GPU environment):
#
# ```bash
# .venv/bin/python -u scripts/run_point_long_training.py \
#   --protocol configs/point_long_training_mc.json \
#   --output runs/02-qam-point-obstacle-long-mc-v2-20261029 --device cuda
# ```
#
# Checkpoints preserve Actor, Critic, target, both Adam states, training RNG,
# and source/protocol hashes. Repeating the command resumes the same protocol;
# a changed scientific source or configuration is rejected.
#
# Comparison uses fixed checkpoints, independent final evaluation, paired test
# noise, and all three seeds. Extra reference compute is reported separately.
# Success thresholds 60/70/80 percent are fixed before observing the curves.

# %%
run = Path(os.environ.get('PAGR_LONG_RUN',
           ROOT/'runs/02-qam-point-obstacle-long-mc-v2-20261029'))
if (run/'protocol.json').exists():
    display(Markdown((run/'report.md').read_text()) if (run/'report.md').exists()
            else Markdown('Training has started; the aggregate report is pending.'))
else:
    display(Markdown('No local result directory yet. Training can run on Ubuntu; '
                     'set `PAGR_LONG_RUN` to a synced result directory to inspect it.'))

# %%
for filename in ('success-gap-aggregate.csv', 'final-independent.csv',
                 'sample-efficiency.csv', 'gradient-audits.csv', 'training-cost.csv'):
    if (run/filename).exists():
        display(Markdown(f'**{filename}**'))
        display(pd.read_csv(run/filename))
for filename in ('learning-curves.png', 'success-gap.png'):
    if (run/filename).exists():
        display(Image(filename=str(run/filename)))

# %% [markdown]
# Interpret early and late gaps separately. An early advantage followed by
# convergence can support an update-efficiency benefit. A persistent advantage
# can support a finite-budget performance benefit. Neither pattern should be
# assumed in advance, and three training seeds remain exploratory evidence.
